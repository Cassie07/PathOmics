import numpy as np
import torch
from data_loading_utils.load_files import *#load_gene_family_info, load_feature, load_genomics_z_score, load_clinical
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class CustomizedDataset(Dataset):
    def __init__(self, patient_id_list, df_gene, df_clinical, dict_family_genes, feature_dimension, feature_folder):#, train = True):

        self.patient_id_list = patient_id_list
        self.df_gene = df_gene
        self.df_clinical = df_clinical
        self.dict_family_genes = dict_family_genes
        self.feature_dimension = feature_dimension
        self.feature_folder = feature_folder

    def __len__(self):
        return len(self.patient_id_list)

    def __getitem__(self, idx):

        patient_id = self.patient_id_list[idx]
        
        x_omics = load_genomics_z_score(self.df_gene, patient_id, self.dict_family_genes)
        x_path = load_feature(self.feature_folder, patient_id, self.feature_dimension) 
        survival_months, censorship, label = load_clinical(self.df_clinical, patient_id)
        return x_path, x_omics, censorship, survival_months, label, patient_id
    
    
class CustomizedDataset_multi_omics(Dataset):
    def __init__(self, patient_id_list, df_list, df_clinical, feature_dimension, feature_folder, dict_family_genes):#, train = True):

        self.patient_id_list = patient_id_list
        self.df_list = df_list
        self.df_clinical = df_clinical
        self.feature_dimension = feature_dimension
        self.feature_folder = feature_folder
        self.dict_family_genes = dict_family_genes

    def __len__(self):
        return len(self.patient_id_list)

    def __getitem__(self, idx):

        patient_id = self.patient_id_list[idx]
        
        x_omics = load_multi_omics_z_score_by_family(self.df_list, patient_id, self.dict_family_genes)
        x_omics = [torch.stack(x_omics)]
        #load_multi_omics_z_score(self.df_list, patient_id, self.dict_family_genes)
        x_path = load_feature(self.feature_folder, patient_id, self.feature_dimension) 
        survival_months, censorship, label = load_clinical(self.df_clinical, patient_id)
        
        return x_path, x_omics, censorship, survival_months, label, patient_id