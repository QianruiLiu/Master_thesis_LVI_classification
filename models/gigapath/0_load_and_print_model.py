import os
import math
import json
import random
import zlib
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import h5py

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score

import openslide
import xml.etree.ElementTree as ET

import gigapath.slide_encoder as slide_encoder

def load_slide_encoder(global_pool: bool = True) -> nn.Module:
    """Load pretrained GigaPath slide encoder from HF hub."""
    # NOTE: CLS token not trained during pretraining; global_pool=True recommended.
    model = slide_encoder.create_model(
        "hf_hub:prov-gigapath/prov-gigapath",
        "gigapath_slide_enc12l768d",
        1536,
        global_pool=global_pool,
    )
    return model

model = load_slide_encoder(global_pool=True)
print("===== MODEL STRUCTURE =====")
print(model)

print("\n===== NAMED MODULES =====")
for name, module in model.named_modules():
    print(name, type(module))

print("\n===== NAMED PARAMETERS =====")
for name, param in model.named_parameters():
    print(name, param.shape, param.requires_grad)