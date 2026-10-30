# Trained models

The trained model files are not stored in this repository because of their size. All 8 files are in one Google Drive folder:

**[Download the models (Google Drive)](https://drive.google.com/drive/folders/1ieqirqQBijlwfUlEa2RiS9SG0vEWbqXL?usp=sharing)**

## What is in the folder

| Module | File | Framework | What it does | Result |
|---|---|---|---|---|
| Clinical | `catboost_liver_model.cbm` | CatBoost | Liver cancer risk from 13 clinical fields | 98% accuracy, 99% ROC-AUC |
| CT | `model.weights.h5` | Keras | 3D Residual U-Net, tumor segmentation | Dice 0.9094 |
| MRI segmentation 1 | `best_seg_model1_DWI.pth` | PyTorch | Initial tumor mask from DWI | Val Dice 0.453 |
| MRI segmentation 2 | `best_seg_model2_T2WI.pth` | PyTorch | Refines the mask using T2WI and mask 1 | Val Dice 0.429 |
| MRI segmentation 3 | `best_seg_model3_cascade.pth` | PyTorch | Final mask from DWI and mask 2 | Val Dice 0.456 |
| MRI classification 1 | `best_cls_model1_malignant.pth` | PyTorch | Malignant vs Benign | 62.5% val, 65.3% test |
| MRI classification 2 | `best_cls_model2_malignant_type.pth` | PyTorch | Type of malignant lesion | 70.6% val, 40.8% test |
| MRI classification 3 | `best_cls_model3_benign_type.pth` | PyTorch | Type of benign lesion | 59.4% val, 50.0% test |

## Where each file goes

Each service looks for its files next to `app.py`:

```
clinical/catboost_liver_model.cbm
ct/config.json              (model architecture, already in the repo)
ct/model.weights.h5
mri/models/best_seg_model1_DWI.pth
mri/models/best_seg_model2_T2WI.pth
mri/models/best_seg_model3_cascade.pth
mri/models/best_cls_model1_malignant.pth
mri/models/best_cls_model2_malignant_type.pth
mri/models/best_cls_model3_benign_type.pth
```

## Model details

**CT.** Input is a single volume of 96×96×48 voxels (one channel). The architecture is stored in `config.json` and the weights in `model.weights.h5`; both are needed to rebuild the model. The model was saved with Keras 3.

**MRI.** All volumes are resized to 32×96×96 and normalized per volume (z-score).

- Segmentation models: model 1 takes 1 channel (DWI), models 2 and 3 take 2 channels (a scan plus the previous mask).
- Classification models: 9 input channels (the 8 MRI phases plus the final mask).
- Each `.pth` file is a checkpoint dictionary. The weights are under the `model_state` key, next to the training history.

Output classes:

| Model | Class index | Meaning |
|---|---|---|
| Classification 1 | 0 / 1 | Benign / Malignant |
| Classification 2 | 0 / 1 / 2 | Intrahepatic Cholangiocarcinoma / Hepatic Metastasis / Hepatocellular Carcinoma |
| Classification 3 | 0 / 1 / 2 / 3 | Hepatic Hemangioma / Hepatic Abscess / Hepatic Cyst / Focal Nodular Hyperplasia |

## A note on loading

The MRI service loads the checkpoints with `torch.load(..., weights_only=False)`, which can run code stored inside a file. Only load checkpoints from this project's Drive folder, never from a source you do not trust.
