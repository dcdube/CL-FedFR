
# Curriculum Learning for FedFR using CR-FIQA Scores
This repository presents a curriculum learning framework for Federated Face Recognition (FedFR), leveraging **CR-FIQA** scores to guide the training process.

Paper: [CR-FIQA paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Boutros_CR-FIQA_Face_Image_Quality_Assessment_by_Learning_Sample_Relative_Classifiability_CVPR_2023_paper.pdf)
Code: [CR-FIQA repository](https://github.com/fdbtrs/CR-FIQA)

<img src="CRFIQA_sort.png"> 

## Overview

We propose a two-stage curriculum learning strategy based on **CR-FIQA** (Classifiability Ranking for Face Image Quality Assessment) scores [13]. The training begins with high-quality (easy) images and progressively incorporates more difficult samples.

### Key Components

- **CR-FIQA Framework**: Used to estimate the **Face Image Quality (FIQ)** of each sample. This framework learns to predict the **relative classifiability** of images.
- **Image Sorting**: Samples are scored using CR-FIQA and sorted in **descending order** based on quality.  
  - **High scores** → Easy-to-classify images  
  - **Low scores** → Hard-to-classify images

### Training Strategy

1. **Stage 1**: Train on the **top 50%** of images (those with the highest CR-FIQA scores).
2. **Stage 2**: Use the model parameters from Stage 1 as initialization and train on **all** images.

This approach is illustrated in:
- **Figure 1**: Image sorting by CR-FIQA scores  
- **Figure 3**: Two-stage curriculum learning process  
- **Algorithm 1**: Training procedure

## Citation

If you use this work, please cite the original [CR-FIQA](https://openaccess.thecvf.com/content/CVPR2023/papers/Boutros_CR-FIQA_Face_Image_Quality_Assessment_by_Learning_Sample_Relative_Classifiability_CVPR_2023_paper.pdf) paper.

---
