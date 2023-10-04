import numpy as np
from tqdm import tqdm
import os 
import torch
from sksurv.metrics import concordance_index_censored
# from lifelines.utils import concordance_index
import json

def create_directory(save_path):
    isExist = os.path.exists(save_path)
    if not isExist:
        os.mkdir(save_path)
        
def train_and_evaluate(fold, epochs, model, loader_list, optimizer, loss_fn, reg_fn, device, save_path, lambda_reg=0., gc = 16, save_model = False, seperate_test_mode = False, model_mode = 'pretrain', fold_mode = 'train_val_test'):
    
    if fold_mode == 'train_val_test':
        if model_mode != 'pretrain':
            logs = {'train_loss':[], 'train_c_index':[], 'test_loss':[], 'test_c_index':[]}
        else:
            logs = {'train_loss':[],'test_loss':[]}
    else:
        if model_mode != 'pretrain':
            logs = {'train_loss':[], 'train_c_index':[], 'test_loss':[], 'test_c_index':[]}
        else:
            logs = {'train_loss':[], 'test_loss':[]}
            
    best_val_loss = float('inf')
    best_val_c_index = float('-inf')
    best_test_c_index = float('-inf')
    best_epoch = 0
    prev_test_loss = float('inf')
    
    if fold_mode == 'train_val_test':
        train_loader, val_loader, test_loader = loader_list
    elif fold_mode == 'k_fold' and model_mode != 'pretrain' and seperate_test_mode:
        train_loader, val_loader, test_loader = loader_list
    else:
        train_loader, test_loader = loader_list
        
    for epoch in range(epochs):
        print()

        if model_mode == 'pretrain':
            train_loss = train_loop(model, train_loader, optimizer, loss_fn, reg_fn, device, epoch, lambda_reg=lambda_reg, gc = gc, model_mode = model_mode)
            print('Epoch: {}, train_loss: {:.4f}'.format(epoch, train_loss))
            logs['train_loss'].append(train_loss)
            test_loss = test_loop(model, test_loader, loss_fn, reg_fn, device, epoch, lambda_reg=lambda_reg, gc = gc, model_mode = model_mode)
            if abs(test_loss) < abs(prev_test_loss):
                print('Epoch: {}, test_loss: {:.4f}'.format(epoch, test_loss))
                prev_test_loss = test_loss
                best_epoch = epoch
                best_pretrain_model = model
                print("Save a new model")
                if save_model:
                    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            }, save_path + '/' + 'fold_{}_model_{}.pt'.format(fold, epoch))
                logs['test_loss'].append(test_loss)
            else:
                logs['test_loss'].append(0)
       
        else:
            train_loss_surv, train_loss, train_c_index = train_loop(model, train_loader, optimizer, loss_fn, reg_fn, device, epoch, lambda_reg=0., gc = gc, model_mode = model_mode)
            print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss, train_c_index))

            logs['train_loss'].append(train_loss)
            logs['train_c_index'].append(train_c_index)
            
            if not seperate_test_mode:
                test_loss_surv, test_loss, test_c_index = test_loop(model, test_loader, loss_fn, reg_fn, device, epoch, lambda_reg=lambda_reg, gc = gc, model_mode = model_mode)
                if test_c_index > best_test_c_index:
                    print('Epoch: {}, test_loss_surv: {:.4f}, test_c_index: {:.4f}'.format(epoch, test_loss, test_c_index))
                    best_test_c_index = test_c_index
                    best_epoch = epoch
                    print("Save a new model")
                    if save_model:
                        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                }, save_path + '/' + 'fold_{}_finetune_model_{}.pt'.format(fold, epoch))
                    logs['test_loss'].append(test_loss)
                    logs['test_c_index'].append(test_c_index)
                else:
                    logs['test_loss'].append(0)
                    logs['test_c_index'].append(0)
            else:
                val_loss_surv, val_loss, val_c_index = test_loop(model, val_loader, loss_fn, reg_fn, device, epoch, lambda_reg=lambda_reg, gc = gc, model_mode = model_mode)
                if val_c_index > best_val_c_index:
                    print('Epoch: {}, val_loss_surv: {:.4f}, val_c_index: {:.4f}'.format(epoch, val_loss, val_c_index))
                    test_loss_surv, test_loss, test_c_index = test_loop(model, test_loader, loss_fn, reg_fn, device, epoch, lambda_reg=lambda_reg, gc = gc, model_mode = model_mode)
                    
                    if test_c_index > best_test_c_index:
                        print('Epoch: {}, test_loss_surv: {:.4f}, test_c_index: {:.4f}'.format(epoch, test_loss, test_c_index))
                        best_test_c_index = test_c_index
                        best_epoch = epoch
                        print("Save a new model")
                        if save_model:
                            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    }, save_path + '/' + 'fold_{}_finetune_model_{}.pt'.format(fold, epoch))
                        logs['test_loss'].append(test_loss)
                        logs['test_c_index'].append(test_c_index)
                    else:
                        logs['test_loss'].append(0)
                        logs['test_c_index'].append(0)
                        
                else:
                    logs['test_loss'].append(0)
                    logs['test_c_index'].append(0)

    if model_mode == 'pretrain':
        return logs, best_pretrain_model
    else:
        return logs, best_test_c_index, best_epoch

            
def train_loop(model, loader, optimizer, loss_fn, reg_fn, device, epoch, lambda_reg=1e-4, gc = 16, model_mode = 'pretrain'):
    


    model.train()
#    adjust_lr(optimizer, epoch, num_epochs, init_lr)
    train_loss_surv, train_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    
    for batch_idx,(data_WSI, data_omic, c, event_time, label, patient_id) in enumerate(tqdm(loader)):
        
        data_WSI = data_WSI.squeeze().to(device)
        data_omic = [i.squeeze().to(device) for i in data_omic]
        label = label.to(device)
        c = c.to(device)
        target = torch.tensor([1])
        target = target.to(device)
        
        # try:
        #     f = open('/home/kding1/projects/2023_PathOmics/Upload_to_server/TCGA_COAD/feature_cluster_record/cluster_1024.json')
        #     cluster_record = json.load(f)
        #     patient_cluster = cluster_record[patient_id[0]]
        # except:
        #     f = open('/home/kding1/projects/2023_PathOmics/Upload_to_server/TCGA_READ/feature_cluster_record/cluster_1024.json')
        #     cluster_record = json.load(f)
        #     patient_cluster = cluster_record[patient_id[0]]

        patient_cluster = []
        if model_mode == 'pretrain':
            path_embedding, omic_embedding = model(x_path=data_WSI, x_omic=data_omic, x_cluster = patient_cluster, mode = model_mode)
#             loss_path = loss_fn(path_embedding,target)
#             loss_omic = loss_fn(omic_embedding,target)
#             loss = (loss_path + loss_omic)/2

            loss = loss_fn(path_embedding, omic_embedding)#, target) # similarity loss
#            loss = loss_fn(path_embedding.unsqueeze(0), omic_embedding.unsqueeze(0), target)
#             MSE = torch.nn.MSELoss()
#             loss_mse = MSE(path_embedding, omic_embedding)
#             loss += loss_mse
            train_loss += loss.item()
            #loss = loss/gc

            
        else:
            hazards, S, Y_hat, _, _ = model(x_path=data_WSI, x_omic=data_omic, x_cluster = patient_cluster, mode = model_mode)
        
            loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
            loss_value = loss.item()

            if reg_fn is None:
                loss_reg = 0
            else:
                loss_reg = reg_fn(model) * lambda_reg
                loss_reg = loss_reg.item()

            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time

            train_loss_surv += loss_value
            train_loss += loss_value + loss_reg

            loss = loss / gc + loss_reg
            
        # backward pass    
        loss.backward()

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()
            
    if model_mode == 'pretrain': 
        return train_loss/len(loader)
    else:
        # calculate loss and error for epoch
        train_loss_surv /= len(loader)
        train_loss /= len(loader)

        # c_index = concordance_index(all_event_times, all_risk_scores, event_observed=1-all_censorships) 
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    #     print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))


        return train_loss_surv, train_loss, c_index

def test_loop(model, loader, loss_fn, reg_fn, device, epoch, lambda_reg=0., gc = 16, model_mode = 'pretrain'):

    
    model.eval()
#    adjust_lr(optimizer, epoch, num_epochs, init_lr)
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    
    for batch_idx,(data_WSI, data_omic, c, event_time, label, patient_id) in enumerate(tqdm(loader)):

        data_WSI = data_WSI.squeeze().to(device)
        data_omic = [i.squeeze().to(device) for i in data_omic]
        label = label.to(device)
        c = c.to(device)
        target = torch.tensor([1])
        target = target.to(device)
        
        # try:
        #     # '/home/kding1/projects/2023_PathOmics/Upload_to_server/TCGA_COAD/feature_cluster_record_3cluster/cluster_1024.json'
        #     f = open('/home/kding1/projects/2023_PathOmics/Upload_to_server/TCGA_COAD/feature_cluster_record/cluster_1024.json')
        #     cluster_record = json.load(f)
        #     patient_cluster = cluster_record[patient_id[0]]
        # except:
        #     f = open('/home/kding1/projects/2023_PathOmics/Upload_to_server/TCGA_READ/feature_cluster_record/cluster_1024.json')
        #     cluster_record = json.load(f)
        #     patient_cluster = cluster_record[patient_id[0]]        
        
        patient_cluster = []
        with torch.no_grad():
            
            if model_mode == 'pretrain':
                path_embedding, omic_embedding = model(x_path=data_WSI, x_omic=data_omic, x_cluster = patient_cluster, mode = model_mode)
            else:
                hazards, S, Y_hat, _, _ = model(x_path=data_WSI, x_omic=data_omic, x_cluster = patient_cluster, mode = model_mode)
                
        if model_mode == 'pretrain':
#             loss_path = loss_fn(path_embedding,target)
#             loss_omic = loss_fn(omic_embedding,target)
#             loss = (loss_path + loss_omic)/2

            loss = loss_fn(path_embedding, omic_embedding)#, target)
#            loss = loss_fn(path_embedding.unsqueeze(0), omic_embedding.unsqueeze(0), target)
#             MSE = torch.nn.MSELoss()
#             loss_mse = MSE(path_embedding, omic_embedding)
#             loss += loss_mse
            
            val_loss += loss.item()

        else:
            loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
            loss_value = loss.item()

            if reg_fn is None:
                loss_reg = 0
            else:
                loss_reg = reg_fn(model) * lambda_reg
                loss_reg = loss_reg.item()

            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.cpu().numpy()
            all_event_times[batch_idx] = event_time

            val_loss_surv += loss_value
            val_loss += loss_value + loss_reg
    
    if model_mode == 'pretrain':
        return val_loss/len(loader)
    else:
        # calculate loss and error for epoch
        val_loss_surv /= len(loader)
        val_loss /= len(loader)

        # c_index = concordance_index(all_event_times, all_risk_scores, event_observed=1-all_censorships) 
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    #     print('Epoch: {}, val_loss_surv: {:.4f}, val_loss: {:.4f}, val_c_index: {:.4f}'.format(epoch, val_loss_surv, val_loss, c_index))


        return val_loss_surv, val_loss, c_index
                           
                
