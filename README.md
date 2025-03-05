# PathOmics: Pathology-and-genomics Multimodal Transformer for Survival Outcome Prediction

The official code of "Pathology-and-genomics Multimodal Transformer for Survival Outcome Prediction" (Accepted to MICCAI2023, top 9%).

### <b> Our Paper </b> [[Link](https://rdcu.be/dnwKf)]

### <b> [2025.03 New Update!!!] We updated the [paper list](https://github.com/Cassie07/PathOmics#literature-reviews-of-pathology-and-genomics-multimodal-analysis-approaches-in-healthcare) of pathology-and-genomics multimodal analysis approaches in healthcare at the end of this repo. </b>

## Workflow overview of the PathOmics
<p align="center">
  <img src="https://github.com/Cassie07/PathOmics/blob/main/Figures/Figure1.png" width="674.1" height="368.3" title="Figure1">
</p>

<b> Workflow overview of the pathology-and-genomics multimodal transformer (PathOmics) for survival prediction. </b> In (a), we show the pipeline of extracting image and genomics feature embedding via an unsupervised pretraining towards multimodal data fusion. In (b) and (c), our supervised finetuning scheme could flexibly handle multiple types of data for prognostic prediction. With the multimodal pretrained model backbones, both multi- or single-modal data can be applicable for our model fine-tuning.

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
4. Extract patch features via pretrained models (e.g., ImageNet-pretrained ResNet50, ResNet101, etc).
5. Save patch features as .npz files. (For each slide, we generate one .npz file to save patch features).
```
For more details about extracting feature, please check [Issue 1](https://github.com/Cassie07/PathOmics/issues/1) and the code in split_tiles_utils/helper.py

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

## Literature reviews of pathology-and-genomics multimodal analysis approaches in healthcare.
|Publish Date|Title|Paper Link|Code|
|---|---|---|---|
|2025.01|Histo-Genomic Knowledge Association For Cancer Prognosis From Histopathology Whole Slide Images|[TMI](https://ieeexplore.ieee.org/abstract/document/10830530)|[Code](https://github.com/ZacharyWang-007/G-HANet)|
|2024.10|MoME: Mixture of Multimodal Experts for Cancer Survival Prediction|[MICCAI 2024](https://papers.miccai.org/miccai-2024/paper/2168_paper.pdf)|[Code](https://github.com/BearCleverProud/MoME)|
|2024.10|PG-MLIF: Multimodal Low-rank Interaction Fusion Framework Integrating Pathological Images and Genomic Data for Cancer Prognosis Prediction|[MICCAI 2024](https://papers.miccai.org/miccai-2024/paper/1221_paper.pdf)|[Code](https://github.com/panxipeng/PG-MLIF)|
|2024.10|Multimodal Cross-Task Interaction for Survival Analysis in Whole Slide Pathological Images|[MICCAI 2024](https://papers.miccai.org/miccai-2024/paper/1280_paper.pdf)|[Code](https://github.com/jsh0792/MCTI)|
|2024.10|HistGen: Histopathology Report Generation via Local-Global Feature Encoding and Cross-modal Context Interaction|[MICCAI 2024](https://papers.miccai.org/miccai-2024/paper/0796_paper.pdf)|NA|
|2024.07|A Multimodal Knowledge-enhanced Whole-slide Pathology Foundation Model|[ArXiv](https://arxiv.org/abs/2407.15362)|NA|
|2024.06|Transcriptomics-guided Slide Representation Learning in Computational Pathology|[CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Jaume_Transcriptomics-guided_Slide_Representation_Learning_in_Computational_Pathology_CVPR_2024_paper.html)|[Code](https://github.com/mahmoodlab/TANGLE)|
|2024.06|Modeling Dense Multimodal Interactions Between Biological Pathways and Histology for Survival Prediction|[CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Jaume_Modeling_Dense_Multimodal_Interactions_Between_Biological_Pathways_and_Histology_for_CVPR_2024_paper.pdf)|[Code](https://github.com/mahmoodlab/SurvPath)|
|2023.10|Multimodal Optimal Transport-based Co-Attention Transformer with Global Structure Consistency for Survival Prediction|[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_Multimodal_Optimal_Transport-based_Co-Attention_Transformer_with_Global_Structure_Consistency_for_ICCV_2023_paper.pdf)|[Code](https://github.com/Innse/MOTCat)|
|2023.10|Pathology-and-genomics Multimodal Transformer for Survival Outcome Prediction|[MICCAI 2023](https://rdcu.be/dnwKf)|[Code](https://github.com/Cassie07/PathOmics)|
|2023.10|Gene-induced Multimodal Pre-training for Image-omic Classification|[MICCAI 2023](https://link.springer.com/epdf/10.1007/978-3-031-43987-2_49?sharing_token=o-LUpRggl7nGR4-4H-Carve4RwlQNchNByi7wbcMAY7jxNo0bliUewITgRTD3ZK5Fj6WT7MCkvR1cQgUKw8y56vn3M_rNLfRgMpLN1Ln7rytbyfCglj7k-ImPaGGbBOVSA9qHdi0XnhwS27mLQ9ueSU11llzx5ZGz7eglf8kIjc%3D)|[Code](https://github.com/huangwudiduan/GIMP)|
|2023.10|Cross-Modal Translation and Alignment for Survival Analysis|[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_Cross-Modal_Translation_and_Alignment_for_Survival_Analysis_ICCV_2023_paper.pdf)|[Code](https://github.com/FT-ZHOU-ZZZ/CMTA)|
|2023.07|Assessment of emerging pretraining strategies in interpretable multimodal deep learning for cancer prognostication|[BioData Mining](https://link.springer.com/article/10.1186/s13040-023-00338-w)|NA|
|2023.04|Multimodal data fusion for cancer biomarker discovery with deep learning|[Nature Machine Intelligence](https://link.springer.com/article/10.1186/s13040-023-00338-w)|NA|
|2023.03|Hierarchical multimodal fusion framework based on noisy label learning and attention mechanism for cancer classification with pathology and genomic features|[Computerized Medical Imaging and Graphics](https://www.sciencedirect.com/science/article/pii/S089561112200146X)|NA|
|2023.03|Hybrid Graph Convolutional Network With Online Masked Autoencoder for Robust Multimodal Cancer Survival Prediction|[TMI](https://ieeexplore.ieee.org/abstract/document/10061470)|[Code](https://github.com/lin-lcx/HGCN)|
|2023.01|Multimodal deep learning to predict prognosis in adult and pediatric brain tumors|[Communications Medicine](https://www.nature.com/articles/s43856-023-00276-y)|[Code](https://github.com/gevaertlab/MultiModalBrainSurvival)|
|2023.01|Survival Prediction for Gastric Cancer via Multimodal Learning of Whole Slide Images and Gene Expression|[BIBM 2022](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9995480)|[Code](https://github.com/constantjxyz/GC-SPLeM)|
|2023.01|Deep Biological Pathway Informed Pathology-Genomic Multimodal Survival Prediction|[arXiv](https://arxiv.org/pdf/2301.02383.pdf)|NA|
|2022.09|Survival Prediction of Brain Cancer with Incomplete Radiology, Pathology, Genomic, and Demographic Data|[MICCAI 2022](https://rdcu.be/cVRze)|[Code](https://github.com/cuicathy/MMD_SurvivalPrediction)|
|2022.09|Discrepancy and gradient guided multi-modal knowledge distillation for pathological glioma grading|[MICCAI 2022](https://rdcu.be/cVRzf)|[Code](https://github.com/CityU-AIM-Group/MultiModal-learning)|
|2022.08|Multimodal integration of radiology, pathology and genomics for prediction of response to PD-(L)1 blockade in patients with non-small cell lung cancer|[Nature Cancer](https://www.nature.com/articles/s43018-022-00416-8)|[Code](https://github.com/msk-mind/luna/)|
|2022.08|Pan-cancer integrative histology-genomic analysis via multimodal deep learning|[Cancer Cell](https://www.sciencedirect.com/science/article/pii/S1535610822003178?via%3Dihub)|[Code](https://github.com/mahmoodlab/PORPOISE)|
|2022.03|HFBSurv: hierarchical multimodal fusion with factorized bilinear models for cancer survival prediction|[Bioinformatics](https://academic.oup.com/bioinformatics/article/38/9/2587/6533437#401769436)|[Code](https://github.com/Liruiqing-ustc/HFBSurv)|
|2021.10|Multimodal co-attention transformer for survival prediction in gigapixel whole slide images|[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Multimodal_Co-Attention_Transformer_for_Survival_Prediction_in_Gigapixel_Whole_Slide_ICCV_2021_paper.html?ref=https://githubhelp.com)|[Code](https://github.com/mahmoodlab/MCAT)|
|2021.09|Deep Orthogonal Fusion: Multimodal Prognostic Biomarker Discovery Integrating Radiology, Pathology, Genomic, and Clinical Data|[MICCAI 2021](https://link.springer.com/chapter/10.1007/978-3-030-87240-3_64)|NA|
|2020.09|Pathomic fusion: an integrated framework for fusing histopathology and genomic features for cancer diagnosis and prognosis|[TMI](https://ieeexplore.ieee.org/abstract/document/9186053)|[Code](https://github.com/mahmoodlab/PathomicFusion)|
|2020.08|Applying Machine Learning for Integration of Multi-Modal Genomics Data and Imaging Data to Quantify Heterogeneity in Tumour Tissues|[Artificial Neural Networks](https://link.springer.com/protocol/10.1007/978-1-0716-0826-5_10)|NA|
|2019.09|Multimodal Multitask Representation Learning for Pathology Biobank Metadata Prediction|[arXiv](https://arxiv.org/pdf/1909.07846.pdf)|NA|
|2019.07|Deep learning with multimodal representation for pancancer prognosis prediction|[Bioinformatics](https://academic.oup.com/bioinformatics/article/35/14/i446/5529139)|[Code](https://github.com/gevaertlab/MultimodalPrognosis)|
|2019.06|Integrative Analysis of Pathological Images and Multi-Dimensional Genomic Data for Early-Stage Cancer Prognosis|[TMI](https://ieeexplore.ieee.org/abstract/document/8727966)|[Code](https://github.com/gevaertlab/MultimodalPrognosis)|
