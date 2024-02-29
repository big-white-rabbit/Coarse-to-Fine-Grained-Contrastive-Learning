# Feature extraction
This folder provides the code for image features extraction and dynamic graph generation.

## Required data
[Chest ImaGenome Dataset](https://physionet.org/content/chest-imagenome/1.0.0/)

## Data preparation
1. Convert MIMIC-CXR-JPG dataset into 1024×1024 PNG images and generate the mapping from dicom id to split id.

```angular2html
python convert.py -p <<path to mimic-cxr-jpg dataset>> -o <<path to output png dataset>>
```

2. Generate the annotation file for split dataset

```angular2html
python get_split_csv.py
```

## Model training
1. train the model for anatomy detection and save the checkpoint.

```angular2html
python train_mimic_ana.py
```

2. train the model for disease classification and save the checkpoint.

```angular2html
python train_mimic_disease.py
```

## Feature extraction and graph generation

After model training, we use the saved checkpoints to extract image features and generate dynamic graphs for each image.

```angular2html
python get_mimic_graph_feats.py
```
