# Coarse-to-Fine-Grained-Contrastive-Learning

This repository contains the implementation of the method described in our paper, "Leveraging Coarse-to-Fine Grained Representations in Contrastive Learning for Differential Medical Visual Question Answering".

![Overview](images%2FFigure_1_overview_v2.png)
---
## 🔥 Environment Setup

1. Creating conda environment

```
conda create -n your_env_name python=3.8
conda activate your_env_name
```

2. Install the required packages
```
pip install -r requirements.txt
```


## 📚 Dataset
The Medical-Diff-VQA dataset, a derivative of the MIMIC-CXR dataset, consists of questions categorized into seven categories: abnormality (145,421), location (84,193), type (27,478), level (67,296), view (56,265), presence (155,726), and difference(164,324). The 'difference' questions are specifically for comparing two images. In total, the Medical-Diff-VQA dataset contains 700,703 question-answer pairs derived from 164,324 pairs of main and reference images.

More details can be found in this [paper](https://dl.acm.org/doi/abs/10.1145/3580305.3599819) and the generation of dataset can be found in this [project](https://github.com/Holipori/MIMIC-Diff-VQA).

**MIMIC-CXR:**
   - Download from [Physionet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).
   - **Note:** This dataset requires authorization.

## 🛠  ️Getting Started
This  repository contains the code of two main folders, feature extraction and model. The feature extraction folder contains the code for extracting the image features and generating the dynamic graphs. The model folder contains the code for training and test our model. Please refer README.md in each folder for more details.

## Acknowledgement

If you find our work useful, please consider citing our paper:
```
Comming soon
```



This project is build upon [EKAID](https://github.com/Holipori/EKAID/tree/main), [MCCFormer](https://github.com/doiken23/mccformers.pytorch), [GLoRIA](https://github.com/marshuang80/gloria) and [BLIP](https://github.com/salesforce/BLIP). Thanks to the contributors of these great codebases.
