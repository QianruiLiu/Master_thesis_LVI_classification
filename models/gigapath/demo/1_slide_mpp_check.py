import huggingface_hub
import os
from gigapath.preprocessing.data.slide_utils import find_level_for_target_mpp

assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"

local_dir = os.path.join(os.path.expanduser("~"), ".cache/")

slide_path = r"/mnt/d/BMM_LVI/00PH07791.ndpi"

print("NOTE: Prov-GigaPath is trained with 0.5 mpp preprocessed slides")
target_mpp = 0.5
level = find_level_for_target_mpp(slide_path, target_mpp)
if level is not None:
    print(f"Found level: {level}")
else:
    print("No suitable level found.")
