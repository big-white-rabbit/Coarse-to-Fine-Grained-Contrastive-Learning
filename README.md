# Coarse-to-Fine-Grained-Contrastive-Learning
## Abstract
Chest X-ray Differential Medical Visual Question Answering (Diff-MedVQA) is a novel multi-modal task designed to answer 
questions about diseases, especially their differences, based on a main image and a reference image. Compared to the 
widely explored visual question answering in the general domain, Diff-MedVQA presents two unique issues: 
(1) variations in medical images are often subtle, and 
(2) it is impossible for two chest X-rays taken at different times to be at exactly the same view. 
These issues significantly hinder the ability to answer questions about medical image differences accurately. 
To address this, we introduce a novel method named Coarse-to-Fine Granularity Contrastive Learning. Specifically, this 
method first leverages MCCFormer to extract global difference features between two chest X-rays, and utilizes the 
organ-disease scene graph method to extract fine-grained anatomical difference features within the X-rays, thus 
mitigating issues related to changes in imaging views. Subsequently, a cross-attention module is employed to capture 
question-related coarse-to-fine difference features, and contrastive learning is used to optimize the similarity between
these features and the corresponding symptom/difference answer text features. Finally, the aligned difference features 
are used for decoding to generate answers. Our proposed method significantly outperforms the baseline MCCFormer on the 
public dataset MIMIC-CXR-Diff and achieves state-of-the-art performance.

![structure](structure.png)


This  repository contains the code of two main folders, feature extraction and model. The feature extraction folder contains the code for extracting the image features and generating the dynamic graphs. The model folder contains the code for training and test our model. Please refer README.md in each folder for more details.
# Dataset
The Medical-Diff-VQA dataset, a derivative of the MIMIC-CXR dataset, consists of questions categorized into seven categories: abnormality (145,421), location (84,193), type (27,478), level (67,296), view (56,265), presence (155,726), and difference(164,324). The 'difference' questions are specifically for comparing two images. In total, the Medical-Diff-VQA dataset contains 700,703 question-answer pairs derived from 164,324 pairs of main and reference images.

More details can be found in this [paper](https://dl.acm.org/doi/abs/10.1145/3580305.3599819) and the generation of dataset can be found in this [project](https://github.com/Holipori/MIMIC-Diff-VQA).
# Acknowledgement
Thanks to Xiao Liang, Di Wang, and ... for their kind support.

This project is build upon [EKAID](https://github.com/Holipori/EKAID/tree/main), [MCCFormer](https://github.com/doiken23/mccformers.pytorch), [GLoRIA](https://github.com/marshuang80/gloria) and [BLIP](https://github.com/salesforce/BLIP). Thanks to the contributors of these great codebases.