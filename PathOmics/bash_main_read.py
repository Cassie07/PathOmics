import sys
from data_loading_utils.load_files import *
from data_loading_utils.Customized_Dataset import *
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import model_and_training_utils
from model_and_training_utils.PathOmics_Survival_model import *
from model_and_training_utils.train_and_eval_utils_core import *
from model_and_training_utils.Customized_Loss import *
from model_and_training_utils.help_utils import *
import torch.optim as optim
import torch
import json
from sklearn.model_selection import KFold, train_test_split
import time
import copy
import random
import argparse

parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')

parser.add_argument('--omic_modal',  choices=['miRNA', 'CNA', 'Methylation'], default='miRNA' ,type=str, help='omic model')
parser.add_argument('--kfold_split_seed', default=42, type=int, help='random seed')
parser.add_argument('--feature_dimension', choices=[256, 512, 1024, 2048], default=1024, type=int, help='image patch feature dimension')
parser.add_argument('--k_fold', default=5, type=int, help='k fold COAD for pretraining')
parser.add_argument('--n_bins', default=4, type=int, help='n survival event intervals')
parser.add_argument('--eps', default=1e-6, type=float, help='calculate bins')
# parser.add_argument('--k_fold', default=5, type=int, help='k fold')
parser.add_argument('--finetune_epochs', default=20, type=int, help='finetuning epochs')
parser.add_argument('--pretrain_epochs', default=20, type=int, help='pretrain epochs')
parser.add_argument('--pretrain_lr', default=1e-4, type=float, help='pretraining learning rate')
parser.add_argument('--finetune_lr', default=5e-5, type=float, help='finetuningg learning rate')
parser.add_argument('--wd', default=0, type=float, help='weight decay')
parser.add_argument('--wd_ft', default=0, type=float, help='weight decay')
parser.add_argument('--gradient_accumulation_step', default=32, type=int, help='Gradient Accumulation Step for MCAT')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--bag_loss', choices=['ce_surv', 'nll_surv', 'cox_surv'], default='nll_surv', type=str, help='supervised loss')
parser.add_argument('--alpha_surv', default=0.0, type=float, help='How much to weigh uncensored patients')
parser.add_argument('--reg_type', choices=['None', 'omic', 'pathomic'], default='None', help='Which network submodules to apply L1-Regularization (default: None)')
parser.add_argument('--model_type', choices=['PathOmics', 'MCAT'], default='PathOmics', help='model name')
parser.add_argument('--fusion_mode', choices=['concat','bilinear','image','omics'], default='concat', help='finetune model fusion type for finetuning (no pretraining and finetuning in baseline models on same dataset setting (e.g., TCGA-COAD experiments))')
parser.add_argument('--prev_fusion_mode', choices=['concat','bilinear','image','omics'], default='concat', help='pretrain model fusion type for finetuning (no pretraining and finetuning in baseline models on same dataset setting (e.g., TCGA-COAD experiments))')
parser.add_argument('--less_data_ratio', default=0.1, type=float, help='less data ratio')
parser.add_argument('--results_string', help='pretraining results (using for loading pretraining model). Example: c-index value_variance among folds') # '0.5333_0.0146'
parser.add_argument('--cuda_device', default='2', type=str)
parser.add_argument('--gene_family_info_path', default ='gene_family',help='gene family info')
parser.add_argument('--coad_feature_folder', default='/home/kding1/projects/2023_PathOmics/Upload_to_server/TCGA_COAD/Extracted_feature', help='extracted feature folder path')
parser.add_argument('--read_feature_folder', default='/home/kding1/projects/2023_PathOmics/Upload_to_server/TCGA_READ/Extracted_feature', help='extracted feature folder path')
parser.add_argument('--clinical_path', default='coadread_tcga_pan_can_atlas_2018/data_clinical_patient_modified.csv', help='clinical info folder path')
parser.add_argument('--proj_ratio', default=1, type=int, help='Model architecture related param: 1: (256, 256), 0.5: (256, 128)')
parser.add_argument('--image_group_method', choices=['random', 'kmeans_cluster'], default='random',help='type of image group method')
parser.add_argument('--omic_bag_type', choices=['Attention', 'SNN', 'SNN_Attention'],default='SNN', help='Architecture for omics data feature extraction')
parser.add_argument('--save_model_folder_name', default='COAD_model_checkpoint',help='Folder name')
parser.add_argument('--experiment_folder_name', default='main_experiments',help='Experiment name')
parser.add_argument('--experiment_id', default='1', type=str)
parser.add_argument('--finetune_test_ratio', default=0.2, type=float, help='finetune data ratio')
parser.add_argument('--pretrain_loss', choices=['MSE','Cosine'], default='MSE', type=str)

# store_true: add '--a' in command --> True, not add '--a' in command --> False
parser.add_argument('--load_model_finetune', action='store_true', help='whether in finetuning mode, if so, no need for running pretraining code') # default True (add)
parser.add_argument('--less_data', action='store_true', help='whether use fewer data for model finetuning(no pretraining and finetuning in baseline models on same dataset setting (e.g., TCGA-COAD experiments))') # default False (not add)
parser.add_argument('--parameter_reset', action='store_true', help='During finetuning, whether reset pretrained parameters or not')# default False (not add)
parser.add_argument('--use_GAP_in_pretrain_flag',  action='store_true', help='Whether use GAP in pretrain or not') # default True (add)

def main():
#    params = parser.parse_args()
    params = parser.parse_args()

    os.environ["params.cuda_device_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = params.cuda_device

    torch.manual_seed(params.kfold_split_seed)
    torch.cuda.manual_seed(params.kfold_split_seed)
    np.random.seed(params.kfold_split_seed)
    random.seed(params.kfold_split_seed)

    start = time.time()
    
    if params.omic_modal == 'Methylation':
        model_size_omic = 'Methylation'
    else:
        model_size_omic = 'small'

    if params.omic_modal == 'miRNA':
        z_score_path = 'coadread_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt'
    elif params.omic_modal == 'CNA':
        z_score_path = 'coadread_tcga_pan_can_atlas_2018/data_cna.txt'
    else:
        z_score_path = 'coadread_tcga_pan_can_atlas_2018/data_methylation_hm27_hm450_merged.txt'#


    load_path = '/home/kding1/projects/2023_PathOmics/Upload_to_server/2023_Multimodal_gene_image/{}/{}/{}_model_{}_OmicBag_{}_FusionType_{}_OmicType_{}'.format(params.save_model_folder_name, params.experiment_folder_name, params.results_string, params.model_type, params.omic_bag_type, params.prev_fusion_mode, params.omic_modal)

    if params.load_model_finetune:
        save_model_mode = False
        save_path = load_path
    else:
        save_model_mode = True # whether save the best model for reusing
        save_path = '/home/kding1/projects/2023_PathOmics/Upload_to_server/PathOmics/{}/{}/'.format(params.save_model_folder_name, params.experiment_folder_name)
        create_directory(save_path)
        save_path += '/{}_model_{}_OmicBag_{}_FusionType_{}_OmicType_{}'.format(params.experiment_id, params.model_type, params.omic_bag_type, params.fusion_mode, params.omic_modal)
        create_directory(save_path)

    # save_model_mode = True # whether save the best model for reusing

    if params.bag_loss == 'ce_surv':
        loss_fn = CrossEntropySurvLoss(alpha = params.alpha_surv)
    elif params.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha = params.alpha_surv)
    elif params.bag_loss == 'cox_surv':
        loss_fn = CoxSurvLoss()
    else:
        raise NotImplementedError

    if params.reg_type == 'omic':
        reg_fn = l1_reg_all
    elif params.reg_type == 'pathomic':
        reg_fn = l1_reg_modules
    else:
        reg_fn = None




    print()
#    params.k_fold = 5
    print()
    # patient_id_list = ['TCGA-F4-6461']
    patient_id_list = get_overlapped_patient(params.coad_feature_folder, params.clinical_path, z_score_path)
    read_patient_id_list = get_overlapped_patient(params.read_feature_folder, params.clinical_path, z_score_path)
    # patient_id_list.remove('TCGA-QG-A5Z2')
    # patient_id_list.remove('TCGA-3L-AA1B')
    # patient_id_list.remove('TCGA-A6-2676')
    # number of patch too small (e.g., 1)
    patient_id_list.remove('TCGA-A6-2675')
    patient_id_list.remove('TCGA-5M-AAT5')
    patient_id_list.remove('TCGA-AA-3521')
    if params.omic_modal != 'miRNA':
        patient_id_list.remove('TCGA-AA-3558')
        read_patient_id_list.remove('TCGA-AF-2689')

    read_patient_id_list.remove('TCGA-F5-6810')


    # patient_id_list.remove('TCGA-CM-4744')
    # patient_id_list.remove('TCGA-AY-6197')
    # patient_id_list.remove('TCGA-D5-6539')
    # patient_id_list.remove('TCGA-G4-6306')
    
    #patient_id_list = patient_id_list[0:50]
    #read_patient_id_list = read_patient_id_list[0:100]
    print('patient_id sample in coad: {}'.format(patient_id_list[0]))
    print('Number of unique patient in coad: {}'.format(len(patient_id_list)))
    print()
    print('patient_id sample in read: {}'.format(read_patient_id_list[0]))
    print('Number of unique patient in read: {}'.format(len(read_patient_id_list)))
    print()

    '''

     Dataset preparation

    '''

    # load genomics z-score
    if params.omic_modal == 'miRNA':
        df_gene = pd.read_csv(z_score_path, delimiter = "\t")
        df_gene = df_gene.drop(['Entrez_Gene_Id'], axis=1)
        df_gene = df_gene[df_gene['Hugo_Symbol'].notna()].dropna()
        df_gene = df_gene.set_index('Hugo_Symbol')
    elif params.omic_modal == 'CNA':
        # load CNA z-score
        df_cna = pd.read_csv(z_score_path, delimiter = "\t")
        df_cna = df_cna.drop(['Entrez_Gene_Id'], axis=1)
        df_cna = df_cna[df_cna['Hugo_Symbol'].notna()].dropna()
        df_cna = df_cna.set_index('Hugo_Symbol')
        df_gene = df_cna
    else:
        # load methylation
        df_methylation = pd.read_csv(z_score_path, delimiter = "\t")
        df_methylation = df_methylation.drop(['ENTITY_STABLE_ID'], axis=1)
        df_methylation = df_methylation.drop(['DESCRIPTION'], axis=1)
        df_methylation = df_methylation.drop(['TRANSCRIPT_ID'], axis=1)
        df_methylation = df_methylation[df_methylation['NAME'].notna()].dropna()
        df_methylation = df_methylation.set_index('NAME')
        df_gene = df_methylation

    # load gene family and genes info
    dict_family_genes = load_gene_family_info('gene_family')

    # load clinical info
    df_clinical = pd.read_csv(params.clinical_path)

    '''
    disc_labels: https://github.com/mahmoodlab/MCAT/blob/b9cca63be83c67de7f95308d54a58f80b78b0da1/datasets/dataset_survival.py
    '''

    df_uncensored = df_clinical[df_clinical.PFS_STATUS == '1:PROGRESSION']
    disc_labels, q_bins = pd.qcut(df_uncensored['OS_MONTHS'], q = params.n_bins, retbins=True, labels=False)
    q_bins[-1] = df_clinical['OS_MONTHS'].max() + params.eps
    q_bins[0] = df_clinical['OS_MONTHS'].min() - params.eps

    disc_labels, q_bins = pd.cut(df_clinical['OS_MONTHS'], bins = q_bins, retbins=True, labels=False, right=False, include_lowest=True)
    df_clinical.insert(2, 'label', disc_labels.values.astype(int))



    '''
    If params.k_fold == 1, we don't use k-fold cross-validation. The entire dataset will be splited into train, validation, and test set. Test set is used for model evaluation.


    If params.k_fold > 1, we use k0fold cross-validation. The entire dataset will be splited into k folds. We used the cross-validated concordance index (average c-Index) to measure the predictive performance of correctly ranking the predicted patient risk scores with respect to overall survival.

    '''

    best_c_index_list = []

    kf = KFold(n_splits = params.k_fold, random_state = params.kfold_split_seed, shuffle = True)

    read_train_val_patient_id, read_test_patient_id = train_test_split(read_patient_id_list, test_size=params.finetune_test_ratio, random_state = params.kfold_split_seed)

    read_train_patient_id, read_val_patient_id = train_test_split(read_train_val_patient_id, test_size=0.25, random_state = params.kfold_split_seed) # val: 0.8 * 0.25 = 0.2

    if not params.less_data:
        finetune_train_dataset = CustomizedDataset(read_train_patient_id, df_gene, df_clinical, dict_family_genes, params.feature_dimension, params.read_feature_folder)
    else:
        finetune_train_idx = copy.deepcopy([i for i in range(len(read_train_patient_id))])
        print(finetune_train_idx)
        random.seed(params.kfold_split_seed)
        finetune_train_idx = list(finetune_train_idx)
        num_data = int(len(finetune_train_idx) * params.less_data_ratio)
        finetune_train_idx = random.sample(finetune_train_idx, k=num_data)
        finetune_train_idx = np.array(finetune_train_idx)
        finetune_train_patient_id = list(np.array(read_patient_id_list)[finetune_train_idx])
        finetune_train_dataset = CustomizedDataset(finetune_train_patient_id, df_gene, df_clinical, dict_family_genes, params.feature_dimension, params.read_feature_folder)


    finetune_train_loader = DataLoader(finetune_train_dataset, batch_size = params.batch_size, shuffle=True, num_workers=12)

    finetune_val_dataset = CustomizedDataset(read_val_patient_id, df_gene, df_clinical, dict_family_genes, params.feature_dimension, params.read_feature_folder)

    finetune_val_loader = DataLoader(finetune_val_dataset, batch_size = params.batch_size, shuffle=False, num_workers=12)

    finetune_test_dataset = CustomizedDataset(read_test_patient_id, df_gene, df_clinical, dict_family_genes, params.feature_dimension, params.read_feature_folder)

    finetune_test_loader = DataLoader(finetune_test_dataset, batch_size = params.batch_size, shuffle=False, num_workers=12)

    finetune_result_by_fold = {i: [] for i in range(params.k_fold)}

    for fold, (train_idx, test_idx) in enumerate(kf.split(patient_id_list)):
        print()
        print('【 Fold {} 】'.format(fold + 1))

        train_patient_id = list(np.array(patient_id_list)[train_idx])
        test_patient_id = list(np.array(patient_id_list)[test_idx])
        print('Number of patient for model training : {}'.format(len(train_patient_id)))
        print('Number of patient for model testing : {}'.format(len(test_patient_id)))

        train_dataset = CustomizedDataset(train_patient_id, df_gene, df_clinical, dict_family_genes, params.feature_dimension, params.coad_feature_folder)
        train_loader = DataLoader(train_dataset, batch_size = params.batch_size, shuffle=True, num_workers=12)

        test_dataset = CustomizedDataset(test_patient_id, df_gene, df_clinical, dict_family_genes, params.feature_dimension, params.coad_feature_folder)
        test_loader = DataLoader(test_dataset, batch_size = params.batch_size, shuffle=False, num_workers=12)

        x_path, x_omics, censorship, survival_months, label, patient_id = next(iter(test_loader))

        # print(x_path.shape)
        print(x_omics[0].shape)
        # print(cen  sorship)
        # print(survival_months)
        # print(label)


        '''
        Model and hyper-parameter
        '''

        # Model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if params.model_type == 'MCAT':
            model = MCAT_Surv_modified(fusion = params.fusion_mode, omic_sizes=[i.shape[1] for i in x_omics])
        elif params.model_type == 'PathOmics':
            model = PathOmics_Surv(device = device, fusion = params.fusion_mode, omic_sizes=[i.shape[1] for i in x_omics], omic_bag = params.omic_bag_type, use_GAP_in_pretrain = params.use_GAP_in_pretrain_flag, proj_ratio = params.proj_ratio, image_group_method = params.image_group_method)


        if not params.load_model_finetune:
            print()
            print('[Train from scratch]')
            # print([i.shape[1] for i in x_omics])
        #     device = "cuda" if torch.cuda.is_available() else "cpu"
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                model = torch.nn.DataParallel(model)
            model.to(device)

            # Define hyper-parameters
            pretrain_optimizer = optim.Adam(model.parameters(), lr = params.pretrain_lr, weight_decay = params.wd)#, betas=(0.9,0.98), params.eps=1e-6,

            # pretrain
            if params.pretrain_loss == 'MSE':
                loss_fn_pretrain = torch.nn.MSELoss()#torch.nn.MSELoss() # torch.nn.CosineEmbeddingLoss()##ContrastiveLoss()#torch.nn.CrossEntropyLoss()
            elif params.pretrain_loss == 'Cosine':
                loss_fn_pretrain = torch.nn.CosineEmbeddingLoss()

            #loss_fn_pretrain = torch.nn.MSELoss()#torch.nn.MSELoss() # torch.nn.CosineEmbeddingLoss()##ContrastiveLoss()#torch.nn.CrossEntropyLoss()
            logs, best_pretrain_model = train_and_evaluate(fold, params.pretrain_epochs, model, [train_loader, test_loader], pretrain_optimizer, loss_fn_pretrain, reg_fn, device, save_path, lambda_reg=0., gc = params.gradient_accumulation_step, save_model = False, model_mode = 'pretrain', fold_mode = 'k_fold')

            if save_model_mode:
                torch.save({
                'model_state_dict': best_pretrain_model.state_dict(),
                }, save_path + '/' + 'fold_{}_pretrain_model.pt'.format(fold))
        else:
            print()
            print('[Load pretrained model]')
            model_dict = model.state_dict()
            checkpoint = torch.load(load_path + '/fold_{}_pretrain_model.pt'.format(fold))
            pretrained_dict = checkpoint['model_state_dict']

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.load_state_dict(pretrained_dict)

            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                best_pretrain_model = torch.nn.DataParallel(model)
            best_pretrain_model = model
            best_pretrain_model.to(device)


        # Finetune
        print()
        print('Finetune')
        if params.less_data:
            print('Number of patient for model finetune-training : {}'.format(len(finetune_train_patient_id)))
            print('Number of patient for model finetune-testing : {}'.format(len(read_test_patient_id)))
        else:
            print('Number of patient for model finetune-training : {}'.format(len(read_train_patient_id)))
            print('Number of patient for model finetune-testing : {}'.format(len(read_test_patient_id)))
        print()

        names = []
        for (name, param), layer in zip(best_pretrain_model.named_parameters(), best_pretrain_model.children()):
            names.append(name)
            if name in ['path_attention_head.attention_a.0.weight', 'path_attention_head.attention_a.0.bias', 'path_attention_head.attention_b.0.weight', 'path_attention_head.attention_b.0.bias', 'path_attention_head.attention_c.weight', 'path_attention_head.attention_c.bias', 'path_rho.0.weight', 'path_rho.0.bias','omic_attention_head.attention_a.0.weight', 'omic_attention_head.attention_a.0.bias', 'omic_attention_head.attention_b.0.weight', 'omic_attention_head.attention_b.0.bias', 'omic_attention_head.attention_c.weight', 'omic_attention_head.attention_c.bias', 'omic_rho.0.weight', 'omic_rho.0.bias','mm.0.weight', 'mm.0.bias', 'mm.2.weight', 'mm.2.bias', 'classifier.weight', 'classifier.bias', 'path_proj.weight', 'path_proj.bias', 'omic_proj.weight', 'omic_proj.bias','logit_scale']:
                param.requires_grad = True
                if params.parameter_reset:
                    if name in ['path_attention_head.attention_a.0.weight', 'path_attention_head.attention_a.0.bias', 'path_attention_head.attention_b.0.weight', 'path_attention_head.attention_b.0.bias', 'path_attention_head.attention_c.weight', 'path_attention_head.attention_c.bias', 'path_rho.0.weight', 'path_rho.0.bias','omic_attention_head.attention_a.0.weight', 'omic_attention_head.attention_a.0.bias', 'omic_attention_head.attention_b.0.weight', 'omic_attention_head.attention_b.0.bias', 'omic_attention_head.attention_c.weight', 'omic_attention_head.attention_c.bias', 'omic_rho.0.weight', 'omic_rho.0.bias','mm.0.weight', 'mm.0.bias', 'mm.2.weight', 'mm.2.bias']:
                        layer.reset_parameters()

            else:
                param.requires_grad = False


        finetune_optimizer = optim.Adam(best_pretrain_model.parameters(), lr = params.finetune_lr, weight_decay = params.wd_ft)
        logs, best_test_c_index, best_epoch = train_and_evaluate(fold, params.finetune_epochs, best_pretrain_model, [finetune_train_loader, finetune_val_loader, finetune_test_loader], finetune_optimizer, loss_fn, reg_fn, device, save_path, lambda_reg=0., gc = params.gradient_accumulation_step, save_model = False, seperate_test_mode = True, model_mode = 'Finetune', fold_mode = 'train_val_test')

        print('【 in fold {}, {} epoch 】 The highest test c-index: {}'.format(fold + 1, best_epoch, best_test_c_index))
        best_c_index_list.append(best_test_c_index)
        finetune_result_by_fold[fold].append(best_epoch)
        finetune_result_by_fold[fold].append(best_test_c_index)

        save = pd.DataFrame.from_dict(logs)
        if not params.load_model_finetune:
            save.to_csv(save_path + '/fold{}_Train_test_logs.csv'.format(fold + 1), header=True)
        else:
            if params.less_data:
                save.to_csv(save_path + '/{}_fold{}_Train_test_logs.csv'.format(params.less_data_ratio, fold + 1), header=True)
            if params.fusion_mode != 'concat':
                save.to_csv(save_path + '/{}_fold{}_Train_test_logs.csv'.format(params.fusion_mode, fold + 1), header=True)

    print()
    print()
    print_result_as_table(finetune_result_by_fold, params.k_fold)
    print('【 The cross-validated c-index (average c-index across folds) 】: {}'.format(sum(best_c_index_list)/len(best_c_index_list)))
    print('【Experiment id】: {}'.format(params.experiment_id))
    print()
    print('【 The standard deviation among folds 】: {}'.format(np.std(best_c_index_list)))
    print()

    print('finish')
    stop = time.time()
    hr = (stop - start)/3600
    print('training time : {} hrs'.format(hr))


if __name__ == "__main__":
    main()
