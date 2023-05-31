import wandb
import time
import torch
import torch.nn as nn
from torch import optim
from torchvision import models
import sys
import os
sys.path.insert(0, os.getcwd()+'/MIMM/src')
import config.Load_Parameter
from SaveWandbRuns.initWandb import initWandb, saveWandbRun
from Models.FeatureEncoder import FeatureEncoderNetwork, FeatureSelectionNetworkExtended
from Models.ClassificationHeads import PTModel, SCModel
from Models.MINE import MIComputer
from Training.Metrics_Utils import compute_metrics


def train_mi_convergence(trainLoader, testLoader,fv_modelPathToFile, pt_modelPathToFile,sc_modelPathToFile, params_dict, paramsFile, saveFile, syncFile):

    # Get the hyperparameters saved in yaml.
    
    torch.manual_seed(params_dict["randomSeed"]["value"])

    config, run = initWandb(params_dict, paramsFile, testType = "MI_CONVERGENCE_")

    torch.cuda.empty_cache()
    training_dataset = params_dict["training_dataset"]["value"]
   
    # Set up model
    if "MorphoMNIST".casefold() in training_dataset.casefold():
        fv_model = FeatureEncoderNetwork().cuda()
    elif "FashionMNIST".casefold() in training_dataset.casefold():
        fv_model = FeatureSelectionNetworkExtended().cuda()
    else:
        raise ValueError('No valid trainType selected. No model found')

    fv_model.load_state_dict(torch.load(fv_modelPathToFile))
    fv_model.cuda()
    fv_model.eval()

    pt_model = PTModel()
    pt_model.load_state_dict(torch.load(pt_modelPathToFile))
    pt_model.cuda()
    pt_model.eval()

    sc_model = SCModel()
    sc_model.load_state_dict(torch.load(sc_modelPathToFile))
    sc_model.cuda()
    sc_model.eval()

    mi_model = MIComputer()
    mi_model.cuda()
    mi_model.train()

    mi_optimizer = optim.Adam(mi_model.parameters(), lr=1e-5)


    # Training
    start_time = time.time()
    num_epochs = params_dict["mi_epochs"]["value"]

    for epoch in range(1,num_epochs+1):
    
        # Loop over training set
        for batch, (X_train, y_train_pt, y_train_sc) in enumerate(trainLoader):

            # Forward to get feature vector (fv)
            fv = fv_model(X_train)

            # Split fv (feature partioning)
            fv_len_2 = int(fv.size(1)/2)
            fv_pt = fv[:, :fv_len_2]
            fv_sc = fv[:, fv_len_2:]

            # Forward splitted part to model_pt and model_sc
            y_pred_pt = pt_model(fv_pt)
            y_pred_sc = sc_model(fv_sc)

            mi, mi_loss = mi_model(fv_pt.detach(), fv_sc.detach())

            # Compute metrics
            accuracy_pt = compute_metrics(y_pred_pt, y_train_pt)
            accuracy_sc = compute_metrics(y_pred_sc, y_train_sc)

            # Log loss and metrics
            wandb.log({
                'epoch': epoch, 'batch': batch, 'accuracy_pt': accuracy_pt, 'accuracy_sc': accuracy_sc, 'mi_convergence': mi.item(), 'mi_convergence_loss':mi_loss.item()})          
            
            # Backprop
            fv_model.zero_grad()
            pt_model.zero_grad()
            sc_model.zero_grad()
            mi_model.zero_grad()

            # Update mi model        
            (-mi_loss).backward()
            mi_optimizer.step()
            # Training of one epoch done.

        # Validation after one epoch
        fv_model.eval()
        pt_model.eval()
        sc_model.eval()
        mi_model.eval()

        with torch.no_grad():
            y_val_pred_pt = []
            y_val_true_pt = []
            y_val_true_sc = []
            y_val_pred_sc = []
            val_mis = 0
            val_mi_losses = 0
            for batch, (X_val, y_val_pt, y_val_sc) in enumerate(testLoader):
                # Extract features
                fv_val = fv_model(X_val)
                fv_val_len_2 = int(fv_val.size(1)/2)
                fv_val_pt_2 = fv_val[:, :fv_val_len_2]
                fv_val_sc_2 = fv_val[:, fv_val_len_2:]

                # Predict CL and CF
                y_val_pred_pt.append(pt_model(fv_val[:, :fv_val_len_2]))
                y_val_pred_sc.append(sc_model(fv_val[:, fv_val_len_2:]))
                y_val_true_pt.append(y_val_pt)
                y_val_true_sc.append(y_val_sc)

                # Compute MI 
                val_mi, val_mi_loss = mi_model(fv_val_pt_2, fv_val_sc_2)
                val_mis += val_mi
                val_mi_losses += val_mi_loss

            y_val_pred_pt = torch.cat(y_val_pred_pt)
            y_val_pred_sc = torch.cat(y_val_pred_sc)
            y_val_true_pt = torch.cat(y_val_true_pt)
            y_val_true_sc = torch.cat(y_val_true_sc)

            # Compute average MI and MI_loss
            y_val_mi_mean = (val_mis * testLoader.batch_size) / \
                len(testLoader.dataset.dataset)
            y_val_mi_mean_loss = (
                val_mi_losses * testLoader.batch_size)/len(testLoader.dataset.dataset)

        
            # Compute val metrics of CL, CF
            val_accuracy_pt = compute_metrics(
                y_val_pred_pt, y_val_true_pt)
            val_accuracy_sc = compute_metrics(
                y_val_pred_sc, y_val_true_sc)

            # Log val loss and metrics
            wandb.log({
                'epoch': epoch, 'val_mi': y_val_mi_mean.item(), 'val_mi_loss': y_val_mi_mean_loss.item(), 'val_accuracy_pt': val_accuracy_pt, 'val_accuracy_sc': val_accuracy_sc})
            print(
                f'Validation epoch: {epoch:2}/{num_epochs} [{10*batch:6}/{len(trainLoader.dataset.dataset)}] val_acc_pt: {val_accuracy_pt:.3f} val_acc_sc: {val_accuracy_sc:.3f} val_MI: {y_val_mi_mean.item():.3f}')
    
        mi_model.train()

    # Report
    training_time = round((time.time() - start_time) / 60)  # in minutes
    print(f'Training done in {training_time} minutes')
    saveWandbRun(run, saveFile, syncFile, params_dict)
    wandb.finish()