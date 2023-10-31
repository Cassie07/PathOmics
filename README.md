# PathOmics: Pathology-and-genomics Multimodal Transformer for Survival Outcome Prediction
The official code of "Pathology-and-genomics Multimodal Transformer for Survival Outcome Prediction" (Accepted to MICCAI2023, top 9%).

<b> Paper </b> [[Link](https://link.springer.com/chapter/10.1007/978-3-031-43987-2_60)]

## Workflow overview of the PathOmics
<p align="center">
  <img src="https://github.com/Cassie07/PathOmics/blob/main/Figures/Figure1.png" width="674.1" height="368.3" title="Figure1">
</p>

Workflow overview of the pathology-and-genomics multimodal transformer (PathOmics) for survival prediction. In (a), we show the pipeline of extracting image and genomics feature embedding via an unsupervised pretraining towards multimodal data fusion. In (b) and (c), our supervised finetuning scheme could flexibly handle mul- tiple types of data for prognostic prediction. With the multimodal pretrained model backbones, both multi- or single-modal data can be applicable for our model fine-tuning.

## Citation
```
@inproceedings{ding2023pathology,
  title={Pathology-and-genomics multimodal transformer for survival outcome prediction},
  author={Ding, Kexin and Zhou, Mu and Metaxas, Dimitris N and Zhang, Shaoting},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={622--631},
  year={2023},
  organization={Springer}
}
```

## Prerequisites
```
python 3.8.18
Pytorch 2.0.1
pytorch-cuda 11.8
Torchvision 0.15.2
Pillow 9.4.0
numpy 1.24.3
pandas 2.0.3
scikit-survival 0.21.0 
scikit-learn 1.2.0
h5py 2.8.0
```
## Usage
### Data prerpocessing
```
1. Download WSIs from TCGA-COAD and TCGA-READ.
2. Download genomics data from CbioPortal and move the downloaded folder into "PathOmics" folder.
* "coadread_tcga_pan_can_atlas_2018" in `bash_main.py` and `bash_main_read.py` is the downloaded folder, please download it before you run the code.
3. Split WSIs into patches and only keep the foreground patches.
4. Extract patch features via pretrained models (e.g., ImageNet-pretrained ResNet101). For more details, please check the code in split_tiles_utils/helper.py
5. Save patch features as .npz files. (For each slide, we generate one .npz file to save patch features).
```

### Run code on TCGA-COAD only
Model will be pretrained and finetuned on theTCGA-COAD training set (4-fold cross-validation).
The finetuned model will be evaluated on the TCGA-COAD hold-out set.

```
python bash_main.py --pretrain_loss 'MSE' --save_model_folder_name 'reproduce_experiments' --experiment_folder_name 'COAD_reproduce' --omic_modal 'miRNA' --kfold_split_seed 42 --pretrain_epochs 25 --finetune_epochs 25 --model_type 'PathOmics' --model_fusion_type 'concat' --model_pretrain_fusion_type 'concat' --cuda_device '2' --experiment_id '1' --use_GAP_in_pretrain_flag --seperate_test
```
### Run code on TCGA-COAD and TCGA-READ
Model will be pretrained on TCGA-COAD (5-fold cross-validation).
Model will be finetuned, validated, and evaluated on the TCGA-READ dataset.

```
python bash_main_read.py --k_fold 5 --fusion_mode 'concat' --prev_fusion_mode 'concat' --pretrain_loss 'MSE' --save_model_folder_name 'reproduce_experiments' --experiment_folder_name 'READ_reproduce' --omic_modal 'miRNA' --kfold_split_seed 42 --pretrain_epochs 25 --finetune_epochs 25 --model_type 'PathOmics' --cuda_device '2' --experiment_id '1' --use_GAP_in_pretrain_flag
```

If you want to use TCGA-COAD pretrain weights and skip the pretraining stage, please add `--load_model_finetune` into your script.
Please modify the code to ensure your pretrain weights saving directory is correct.

### Use data-efficient mode in finetuning stage
Please add `--less_data` into your script and set `--finetune_test_ratio` as your preferred ratio for indicating the ratio of data used for model finetuning.


