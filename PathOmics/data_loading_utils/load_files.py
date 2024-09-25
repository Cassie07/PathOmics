import collections
import os
import numpy as np
import pandas as pd
import torch


def create_directory(save_path):
    isExist = os.path.exists(save_path)
    if not isExist:
        os.mkdir(save_path)
        
'''
Load genes in gene families

input: gene_family_folder_path
output: {family:[genes]}

'''
def load_gene_family_info(gene_family_folder_path):
    dir_list = os.listdir(gene_family_folder_path)
    dir_list = [gene_family_folder_path + '/' + i for i in dir_list]
    
    all_gene = []
    dict_family_genes = {}
    for p in dir_list:
        # opening the file in read mode
        my_file = open(p, "r")

        # reading the file
        data = my_file.read()
        # replacing end splitting the text 
        # when newline ('\n') is seen.
        data_into_list = data.split("\n")
        all_gene += data_into_list
        gene_family = p.split('/')[-1].split('.')[0]
        print(gene_family + ' : ' + str(len(set(data_into_list))))
        dict_family_genes[gene_family] = data_into_list
        my_file.close()
    print()
    print('Total number of genes : ' + str(len(set(all_gene))))
    print()
#     print(dict_family_genes['tumor suppressors'])
    return dict_family_genes

# dict_family_genes = load_gene_family_info('gene_family')


'''
Load image features

input: feature_folder, patient id, feature dimension
output: feature (numpy array)

'''

def load_feature(feature_folder, patient_id, feature_dimension):
    file_path = feature_folder + '/' + patient_id + '-01Z-00-DX1_{}.npy'.format(feature_dimension)
    feature = np.load(file_path)
    return torch.tensor(feature)

# feature_folder = '/newDisk/users/dingkexin/upload/TCGA_COAD/Extracted_feature'
# patient_id = 'TCGA-F4-6461-01'
# feature_dimension = 1024
# x_path = load_feature(feature_folder, patient_id, feature_dimension)   
# print(x_path.shape)

'''
Load genomics (rna-seq z-score) data

input: path
output: gene family list : x_omics = [torch.shape(100), ..., torch.shape(500)]

'''

def load_genomics_z_score(df_gene, patient_id, dict_family_genes):

    x_omics = []
    
    patient_id = patient_id + '-01'
    
    for family, gene_list in dict_family_genes.items():
        genes = list(df_gene.index)
        gene_list = [i for i in gene_list if i in genes]
        gene_list = [i for i in gene_list if ';' not in i]
        gene_z_scores = df_gene[patient_id].loc[gene_list]
        gene_z_scores = gene_z_scores[gene_z_scores.notna()].dropna()
        x_omics.append(torch.tensor(np.array(gene_z_scores).astype(np.float32)))
    return x_omics

# z_score_path = 'coadread_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt'
# df_clean = pd.read_csv(z_score_path, delimiter = "\t")
# df_clean = df_clean.drop(['Entrez_Gene_Id'], axis=1)
# df_clean = df_clean[df_clean['Hugo_Symbol'].notna()].dropna()
# df_clean = df_clean.set_index('Hugo_Symbol')

# patient_id = 'TCGA-F4-6461-01'
# dict_family_genes = load_gene_family_info('gene_family')
# x_omics = load_genomics_z_score(df_clean, patient_id, dict_family_genes)


'''
Load patient clinical info

input: df_clinical, patient id
output: survival time, censorship, label

'''

def load_clinical(df_clinical, patient_id):
    
    df_patient = df_clinical[df_clinical.PATIENT_ID == patient_id[0:12]]
    censorship = df_patient.OS_STATUS.values[0]
    survival_event = df_patient.OS_MONTHS.values[0]
    label = df_patient.label.values[0]
    
    return survival_event, 0 if censorship == '1:DECEASED' else 1, label

# clinical_path = 'coadread_tcga_pan_can_atlas_2018/data_clinical_patient_modified.csv'
# df_clinical = pd.read_csv(clinical_path)

# patient_id = 'TCGA-F4-6461-01'
# survival_event, censorship, label = load_clinical(df_clinical, patient_id, 4)
# print(survival_event, censorship, label)

'''
Get patient list
Filtering condition:
1. Have genomics information (cbioportal)
2. Have WSI (TCGA)
3. Have clinical information (TCGA)

Patient id : 'TCGA-F4-6461'

'''

def get_overlapped_patient(path_img, path_clinical, path_genonmic):
    patient_list = []
    patient_img = os.listdir(path_img)
    patient_img = list(set([i[0:12] for i in patient_img]))

    df = pd.read_csv(path_clinical)
    patient_clinical = list(set(list(df.PATIENT_ID)))
    patient_list = [i for i in patient_img if i in patient_clinical]
    
    df = pd.read_csv(path_genonmic, delimiter = '\t')
    patient_genmonic = list(set([i[0:12] for i in list(df.columns)[2:]]))
    patient_list = [i for i in patient_list if i in patient_genmonic]
    return patient_list

# path_img = '/newDisk/users/dingkexin/upload/TCGA_COAD/Extracted_feature'    
# path_clinical = 'coadread_tcga_pan_can_atlas_2018/data_clinical_patient_modified.csv'
# path_genonmic= 'coadread_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt'

# patient_list = get_overlapped_patient(path_img, path_clinical, path_genonmic)


'''
Load omics data: rna-seq, cna, methylation

input: path
output: gene family list : x_omics = [torch.shape(100), ..., torch.shape(500)]

'''

def load_multi_omics_z_score(multi_omics_df_list, patient_id, dict_family_genes):

    x_omics = []
    
    gene_list = []
    for tmp_gene_list in dict_family_genes.values():
        gene_list += list(tmp_gene_list)
    
    patient_id = patient_id + '-01'
    for df_gene in multi_omics_df_list:
        genes = list(df_gene.index)
        gene_list = [i for i in gene_list if i in genes]
        gene_z_scores = df_gene[patient_id].loc[gene_list]
        gene_z_scores = gene_z_scores[gene_z_scores.notna()].dropna()
        x_omics.append(torch.tensor(np.array(gene_z_scores).astype(np.float32)))
    return x_omics



'''
Load omics (rna-seq z-score) data + gene family

input: path
output: gene family list : x_omics = [torch.shape(100), ..., torch.shape(500)]

'''

def load_multi_omics_z_score_by_family(multi_omics_df_list, patient_id, dict_family_genes):

    x_omics = []
    
    patient_id = patient_id + '-01'
    
    for family, gene_list in dict_family_genes.items():
        
        all_omics_gene_z_scores = pd.DataFrame()
        
        for df_gene in multi_omics_df_list:
            genes = list(df_gene.index)
            gene_list = [i for i in gene_list if i in genes]
            gene_list = [i for i in gene_list if ';' not in i]
            gene_z_scores = df_gene[patient_id].loc[gene_list]
            gene_z_scores = gene_z_scores[gene_z_scores.notna()].dropna()
            all_omics_gene_z_scores = pd.concat([all_omics_gene_z_scores, gene_z_scores], ignore_index=True)
        all_omics_gene_z_scores = torch.tensor(np.array(all_omics_gene_z_scores).astype(np.float32))
        x_omics.append(all_omics_gene_z_scores.squeeze(-1))
    return x_omics


