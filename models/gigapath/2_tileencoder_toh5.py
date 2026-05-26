"""
This script extracts tile-level GigaPath embeddings from pre-tiled whole-slide
images and saves them as one HDF5 file per slide. For each NDPI slide, the script
reads the corresponding tiling output dataset.csv, resolves the tile PNG paths,
loads the tile images, and passes them through the pretrained GigaPath tile
encoder.
In addition to tile embeddings and tile coordinates, the script optionally parses
the corresponding NDPA circle annotation and converts the ROI position from
Hamamatsu stage coordinates into level-0 pixel coordinates. Based on this ROI,
it saves distance-to-ROI information for each tile, which can later be used for
ROI-guided sampling during slide-level model training.

Usage
-----
Run from the project root directory after the preprocessing/tiling step:

    python models/gigapath/2_tileencoder_toh5.py \
        --slides-dir /mnt/d/BMM_LVI \
        --tiling-root /home/student/.cache/outputs/preprocessing \
        --h5-out-root /mnt/d/tile_encoder_h5files \
        --level 1 \
        --tile-size 1024 \
        --batch-size 128 \
        --num-workers 4

Outputs
-------
Output folder for HDF5 files. For each slide, the script writes:
    <h5_out_root>/<slide_id>.tile_embeds.h5

Each H5 file contains:
    - tile_embeds
    - coords
    - dist_to_roi
    - is_roi_near
    - slide-level metadata as H5 attributes

Notes
-----
The tiling parameters, especially --tile-size and --level, should match the
previous tiling step.
"""

import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pandas as pd
from PIL import Image

import openslide
import timm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# These variables are set from CLI arguments in main().
slides_dir = Path("/mnt/d/BMM_LVI")
tiling_root = Path("/home/student/.cache/outputs/preprocessing")
h5_out_root = Path("/mnt/d/tile_encoder_h5files")
tile_size = 1024
level = 1
batch_size = 128
num_workers = 4


# -----------------------------
# Helper functions
# -----------------------------
def parse_circle_roi_ndpa(ndpa_path: Path):
    """Parse the first circle ROI from an NDPA file. Return (cx_nm, cy_nm, r_nm) or None."""
    if not ndpa_path.exists():
        return None
    tree = ET.parse(str(ndpa_path))
    root = tree.getroot()
    circ = root.find(".//annotation[@type='circle']")
    if circ is None:
        return None
    cx_nm = float(circ.findtext("x"))
    cy_nm = float(circ.findtext("y"))
    r_nm = float(circ.findtext("radius"))
    return cx_nm, cy_nm, r_nm


def get_mpp(slide: openslide.OpenSlide):
    """Read microns-per-pixel from OpenSlide properties."""
    props = slide.properties
    mpp_x = props.get("openslide.mpp-x", None)
    mpp_y = props.get("openslide.mpp-y", None)
    if mpp_x is None or mpp_y is None:
        raise ValueError("Cannot find openslide.mpp-x / openslide.mpp-y in slide properties.")
    return float(mpp_x), float(mpp_y)


def ndpa_stage_nm_to_level0_px(cx_stage_nm, cy_stage_nm, r_nm, slide: openslide.OpenSlide):
    """
    Convert Hamamatsu NDPA circle (stage coordinates in nm) to level-0 pixel coordinates.

    Uses Hamamatsu offsets:
      - hamamatsu.XOffsetFromSlideCentre
      - hamamatsu.YOffsetFromSlideCentre
    """
    props = slide.properties
    mpp_x = float(props["openslide.mpp-x"])
    mpp_y = float(props["openslide.mpp-y"])

    # Hamamatsu center offsets (nm)
    x0_nm = float(props["hamamatsu.XOffsetFromSlideCentre"])
    y0_nm = float(props["hamamatsu.YOffsetFromSlideCentre"])

    W0, H0 = slide.dimensions  # level-0 pixels

    # Convert slide size to nm
    W_nm = W0 * mpp_x * 1000.0
    H_nm = H0 * mpp_y * 1000.0

    # Stage nm -> relative-to-centre nm
    x_rel = cx_stage_nm - x0_nm
    y_rel = cy_stage_nm - y0_nm

    # Relative-to-centre nm -> image nm (top-left origin, y-down)
    x_img_nm = x_rel + (W_nm / 2.0)
    y_img_nm = y_rel + (H_nm / 2.0)

    # nm -> level-0 pixels
    x_px0 = x_img_nm / (mpp_x * 1000.0)
    y_px0 = y_img_nm / (mpp_y * 1000.0)

    # radius nm -> pixels
    r_px0 = r_nm / (mpp_x * 1000.0)

    return float(x_px0), float(y_px0), float(r_px0)




class TilePngDataset(Dataset):
    """Loads tile PNGs from disk and returns (image_tensor, coords_tensor)."""
    def __init__(self, image_paths, coords, transform):
        self.image_paths = image_paths
        self.coords = coords
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB")
        img = self.transform(img)
        c = self.coords[idx]
        return img, torch.from_numpy(c)


def resolve_tile_paths(df: pd.DataFrame, slide_out_dir: Path):
    """
    Resolve absolute tile PNG paths.
    """
    image_paths = []
    for rel in df["image"].tolist():
        rel_path = Path(rel)

        p = slide_out_dir.parent / rel_path
        if p.exists():
            image_paths.append(str(p))
            continue

        raise FileNotFoundError(
            f"Cannot resolve tile path for rel='{rel}'."
        )
    return image_paths


def process_one_slide(slide_path: Path, tile_encoder, transform, device):
    """
    Build H5 for one NDPI slide:
      - read dataset.csv from tiling output
      - run tile encoder on all tile PNGs
      - compute dist_to_roi from NDPA circle ROI (if exists)
      - write H5 to h5_out_root
    """
    slide_id = slide_path.name  # e.g. "00PH05780.ndpi"
    slide_out_dir = tiling_root / "output" / slide_id
    dataset_csv = slide_out_dir / "dataset.csv"

    if not dataset_csv.exists():
        print(f"[SKIP] dataset.csv not found for {slide_id}: {dataset_csv}")
        return

    df = pd.read_csv(dataset_csv)

    # Optional: filter out negative coords (border noise)
    # df = df[(df["tile_x"] >= 0) & (df["tile_y"] >= 0)].reset_index(drop=True)

    coords = df[["tile_x", "tile_y"]].to_numpy(dtype=np.float32)
    image_paths = resolve_tile_paths(df, slide_out_dir)

    # Build dataloader
    dl = DataLoader(
        TilePngDataset(image_paths, coords, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Encode tiles
    all_embeds, all_coords = [], []
    tile_encoder.eval()
    with torch.no_grad():
        for imgs, cs in dl:
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast(device_type='cuda', enabled=(device.type == "cuda"), dtype=torch.float16):
                emb = tile_encoder(imgs)  # [B, D]
            all_embeds.append(emb.detach().cpu().half())  # float16 on disk
            all_coords.append(cs.cpu().float())

    tile_embeds = torch.cat(all_embeds, dim=0).numpy()
    coords_out = torch.cat(all_coords, dim=0).numpy()

    # ROI distances
    ndpa_path = Path(str(slide_path) + ".ndpa")  # "xx.ndpi.ndpa"
    roi = parse_circle_roi_ndpa(ndpa_path)

    dist_to_roi = np.full((coords_out.shape[0],), np.nan, dtype=np.float32)
    is_roi_near = np.zeros((coords_out.shape[0],), dtype=np.uint8)

    if roi is not None:
        cx_nm, cy_nm, r_nm = roi
        slide = openslide.OpenSlide(str(slide_path))
        mpp_x, mpp_y = get_mpp(slide)
        ds = float(slide.level_downsamples[level])
        slide.close()

        slide = openslide.OpenSlide(str(slide_path))
        roi_x0, roi_y0, roi_r0 = ndpa_stage_nm_to_level0_px(cx_nm, cy_nm, r_nm, slide)

        slide.close()

        # compute distances in LEVEL-0 coords
        tile_centers = coords_out + np.array([tile_size/2, tile_size/2], dtype=np.float32)
        d = np.sqrt((tile_centers[:,0] - roi_x0)**2 + (tile_centers[:,1] - roi_y0)**2).astype(np.float32)
        dist_to_roi = d

        margin = float(tile_size) * 1.0
        is_roi_near = (d <= (roi_r0 + margin)).astype(np.uint8)
        idx = np.argsort(dist_to_roi)[:10]
        idx = np.argsort(dist_to_roi)[:10]


    # Save H5
    h5_path = h5_out_root / f"{slide_id}.tile_embeds.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("tile_embeds", data=tile_embeds, compression="gzip")
        f.create_dataset("coords", data=coords_out, compression="gzip")
        f.create_dataset("dist_to_roi", data=dist_to_roi, compression="gzip")
        f.create_dataset("is_roi_near", data=is_roi_near, compression="gzip")

        f.attrs["slide_id"] = slide_id
        f.attrs["level"] = int(level)
        f.attrs["tile_size"] = int(tile_size)
        f.attrs["ndpa_path"] = str(ndpa_path)

    print(f"[Done] {slide_id} -> {h5_path.name}  tiles={coords_out.shape[0]}")




def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract GigaPath tile embeddings and save one H5 file per slide."
    )
    parser.add_argument(
        "--slides-dir",
        type=Path,
        default=Path("/mnt/d/BMM_LVI"),
        help="Folder containing .ndpi slides and optional .ndpi.ndpa files. Default: /mnt/d/BMM_LVI",
    )
    parser.add_argument(
        "--tiling-root",
        type=Path,
        default=Path("/home/student/.cache/outputs/preprocessing"),
        help="Root folder from tile_one_slide preprocessing. Default: /home/student/.cache/outputs/preprocessing",
    )
    parser.add_argument(
        "--h5-out-root",
        type=Path,
        default=Path("/mnt/d/tile_encoder_h5files"),
        help="Output folder for H5 files. Default: /mnt/d/tile_encoder_h5files",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        help="Tiling level used during preprocessing. Must match dataset coordinates. Default: 1",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Tile size used during preprocessing. Default: 1024",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for tile encoder inference. Default: 128",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers. Default: 4",
    )
    return parser.parse_args()


def main():
    global slides_dir, tiling_root, h5_out_root, tile_size, level, batch_size, num_workers

    args = parse_args()

    slides_dir = args.slides_dir
    tiling_root = args.tiling_root
    h5_out_root = args.h5_out_root
    tile_size = args.tile_size
    level = args.level
    batch_size = args.batch_size
    num_workers = args.num_workers

    if not slides_dir.exists():
        raise FileNotFoundError(f"slides_dir does not exist: {slides_dir}")
    if not tiling_root.exists():
        raise FileNotFoundError(f"tiling_root does not exist: {tiling_root}")

    h5_out_root.mkdir(parents=True, exist_ok=True)

    print("slides_dir:", slides_dir)
    print("tiling_root:", tiling_root)
    print("h5_out_root:", h5_out_root)
    print("level:", level)
    print("tile_size:", tile_size)
    print("batch_size:", batch_size)
    print("num_workers:", num_workers)

    # GigaPath tile encoder expects 224x224 with this preprocessing.
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tile_encoder = tile_encoder.to(device)

    ndpi_files = sorted(slides_dir.glob("*.ndpi"))
    print("Found NDPI files:", len(ndpi_files))

    for slide_path in ndpi_files:
        try:
            process_one_slide(slide_path, tile_encoder, transform, device)
        except Exception as e:
            print(f"[FAIL] {slide_path.name}: {repr(e)}")


if __name__ == "__main__":
    main()
