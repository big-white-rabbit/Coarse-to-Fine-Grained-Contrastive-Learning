# Model training

## Data

1. Download the annotation file of MIMIC-Diff-VQA dataset from [Physio](https://physionet.org/content/medical-diff-vqa/1.0.0/)
and place all files into `./data`. Make sure there are `mimic_all.csv`, `all_diseases.json`, `mimic_pair_questions.csv` in the folder.
Besides, the folder also should contain `study2dicom.pkl` and `keywords.json`.
2. Transform annotation file `mimic_pair_question.csv` to sequence file `VQA_mimic_dataset.h5`. And generate the split file and COCOformat ground truth files.

```angular2html
python dataset_preparation.py -t -c
```
The followings are generated files

```angular2html
./data/mimic_gt_captions_test.json
./data/mimic_gt_captions_val.json
./data/mimic_gt_captions_train.json
./data/VQA_mimic_dataset.h5
./data/vocab_mimic_VQA.json
./data/splits_mimic_VQA.json
```

## Training

```angular2html
python pretrain.py
```

--pretrain the model and save the checkpoint.

```angular2html
python train_vqa.py --resume True --resume_fold <path of saved checkpoint>
```

--train VQA with the pretrained model and evaluate the results.