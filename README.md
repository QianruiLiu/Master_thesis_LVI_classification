# Identifying lymphovascular invasion in breast cancer by deep learning on histopathological slides
The github repository from master thesis of BINP51

Author: Qianrui Liu

Github Link: https://github.com/QianruiLiu/Master_thesis_LVI_classification

**Workflow overview**
> 0.Environment setup

> 1.Dataset structure and label

> 2.WSI preprocessing

> 3.Tile-level feature extraction

> 4.Model development

> 5.Independent test evaluation

> 6.Robustness analysis across random seeds

> 7.Clinical analysis

*For an easier management of the installed programs and their dependencies, all analyses were performed in a Conda environment.

## 0) Environment setup

In this study, three environments need to be set up. First of all, clone this repository to your computer with

```bash
git clone https://github.com/QianruiLiu/Master_thesis_LVI_classification.git
```

### Gigapath

* **Install gigapath**

Download gigapath repository provided in https://github.com/prov-gigapath/prov-gigapath and open it by

```bash
git clone https://github.com/prov-gigapath/prov-gigapath
cd prov-gigapath
```

* **Build Gigapath environment**

```bash
conda env create -f ../Master_thesis_LVI_classification/envs/gigapath.yml
conda activate gigapath
pip install -e .  # Install the local GigaPath repository as an editable Python package
```

  *Notice: All the scripts in model/gigapath should be run in this environment
  
### CLAM

```bash
cd ~/Master_thesis_LVI_classification
conda env create -f envs/clam.yml
conda activate clam_latest
```

The original CLAM repository can be found in https://github.com/mahmoodlab/CLAM.git.

  *Notice: All the scripts in model/CLAM should be run in this environment

### Survival_analysis

```bash
cd ~/Master_thesis_LVI_classification
conda env create -f envs/clinical_analysis.yml
conda activate lvi_clinical_analysis
```

  *Notice: All the scripts in clinical_analysis/ should be run in this environment

## 1) Dataset structure and label

### Dataset overview of WSIs
This study based on 488 breast cancer H&E WSIs with slide-level LVI labels. The slides are in `.ndpi` format, each accompanied by an `.ndpa` annotation file containing pathologist-provided LVI-related annotations, such as ROI circles or pins.

The figure shows an example of the slides with ROI circles indicating LVI area：

<p align="center">
  <img src="dataset/figure.png" width="700">
</p>

### Slide-level LVI labels
* **Pathologist B's slide-level LVI labels**were used as the main ground-truth labels for model training and evaluation.

* **Pathologist A's LVI-positive slides with ROI circle annotations**were used only for ROI-guided sampling during training and for qualitative model interpretation.

### Stratified dataset splitting
Dataset splitting was stratified by the main slide-level LVI ground-truth label, which was divided into a development set and an independent test set at a 2:1 ratio. This can be done by the codes below:

```bash
import pandas as pd
from sklearn.model_selection import train_test_split

# Read file
df = pd.read_csv("LVI_lable.tsv", sep="\t")

# Remove rows without LVI label
df = df.dropna(subset=["LVI"])

# Stratified split by LVI, 2:1 = development : independent test
dev_df, test_df = train_test_split(
    df,
    test_size=1/3,
    stratify=df["LVI"],
    random_state=42
)

# Save files
dev_df.to_csv("labels_development.tsv", sep="\t", index=False)
test_df.to_csv("labels_independent_test.tsv", sep="\t", index=False)
```
The codes should be run in gigapath environment. The paths of files should be the real path in your computer. The example divided labels used in this study can be found in **labels_and_sheets/labels_develop.tsv** and **labels_and_sheets/labels_independent_test.tsv**.
