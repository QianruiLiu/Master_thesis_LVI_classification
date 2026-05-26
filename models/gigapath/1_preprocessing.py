"""
Tile whole-slide NDPI files using the Prov-GigaPath preprocessing pipeline.

Usage
-----
Run from the project root directory:

    python models/gigapath/1_preprocessing.py \
        --in-dir /mnt/d/BMM_LVI \
        --save-dir /home/student/.cache/outputs/preprocessing \
        --level 1 \
        --tile-size 1024

Outputs
-------
Output directory used by gigapath.pipeline.tile_one_slide. The expected
 downstream structure is typically:
    <save_dir>/output/<slide_id>/dataset.csv
    <save_dir>/output/<slide_id>/...tile PNGs

"""

import argparse
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

from gigapath.pipeline import tile_one_slide


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tile NDPI whole-slide images using the Prov-GigaPath preprocessing pipeline."
    )
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=Path("/mnt/d/BMM_LVI"),
        help="Directory containing input .ndpi files. Default: /mnt/d/BMM_LVI",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path(os.path.expanduser("~")) / ".cache" / "outputs" / "preprocessing",
        help="Directory where tiling outputs will be saved. Default: ~/.cache/outputs/preprocessing",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        help="OpenSlide level used for tiling. Default: 1",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Tile size in pixels. Default: 1024",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    assert "HF_TOKEN" in os.environ, (
        "Please set the HF_TOKEN environment variable to your Hugging Face API token"
    )

    if not args.in_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.in_dir}")

    args.save_dir.mkdir(parents=True, exist_ok=True)

    ndpi_files = sorted(args.in_dir.glob("*.ndpi"))
    print("Found NDPI:", len(ndpi_files))
    print("Input directory:", args.in_dir)
    print("Save directory:", args.save_dir)
    print("Level:", args.level)
    print("Tile size:", args.tile_size)

    for slide_path in ndpi_files:
        slide_path_str = str(slide_path)
        print("\n=== Processing:", slide_path_str, "===")
        print(
            "NOTE: Prov-GigaPath is trained with 0.5 mpp preprocessed slides. "
            "Please make sure to use the appropriate level for the 0.5 MPP"
        )

        try:
            tile_one_slide(
                slide_path_str,
                save_dir=str(args.save_dir),
                level=args.level,
                tile_size=args.tile_size,
            )
            print("Done:", slide_path_str)
        except Exception as e:
            print("FAILED:", slide_path_str, "->", repr(e))

    print("\nAll done.")


if __name__ == "__main__":
    main()
