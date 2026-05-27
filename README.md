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

This study based on 488 breast cancer H&E WSIs with slide-level LVI labels. The slides are in `.ndpi` format, each accompanied by an `.ndpa` annotation file containing pathologist-provided LVI-related annotations, such as ROI circles or pins.

<p align="center">
  <img src="dataset/figure.png" width="700">
</p>
