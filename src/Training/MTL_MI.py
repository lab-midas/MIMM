import os
import sys
sys.path.insert(0, os.getcwd()+'/MIMM/src')


import wandb
import time
import torch
import torch.nn as nn
from torch import optim
from torchvision import models

import config.Load_Parameter
from Models.FeatureEncoder import FeatureEncoderNetwork
from Models.ClassificationHeads import PTModel, SCModel
from Models.MINE import MIComputer
from Training.Loss_Utils import crossEntropyLoss_for_MTL
from Training.Metrics_Utils import compute_metrics
from SaveWandbRuns.initWandb import create_experiments_model_folder, save_best_models_in_wandb

def mtl_mi_train(trainLoader, valLoader):

    # Get the hyperparameters saved in yaml.
    params = config.Load_Parameter.params
    torch.manual_seed(params.randomSeed)

    # Set up model
    if "MTL".casefold() in params.trainType.casefold():
        fv_model = FeatureEncoderNetwork().cuda()
    else:
        raise ValueError('No valid trainType selected. No model found.')
    
    pt_model = PTModel().cuda()
    sc_model = SCModel().cuda()
    mi_model = MIComputer().cuda() 

    fv_model.train()
    pt_model.train()
    sc_model.train()
    mi_model.train()

    # Set up optimizers
    fv_params = [p for p in fv_model.parameters()]
    pt_params = [p for p in pt_model.parameters()]
    sc_params = [p for p in sc_model.parameters()]

    optimizer = optim.Adam(fv_params+pt_params+sc_params, lr=1e-5, weight_decay=1e-4)
    mi_optimizer = optim.Adam(mi_model.parameters(), lr=1e-5)
    nr_batches_mi = params.nr_batches_MI_train

    # Training
    start_time = time.time()
    num_epochs = params.epochs * nr_batches_mi
    best_accuracy_pt = 0
    best_accuracy_sc = 0
    best_val_loss = 1000
    best_epoch = 0
    best_mi = 0

    for epoch in range(1,num_epochs+1):

        # Record learning rate
        wandb.log({'epoch': epoch, 'lr': optimizer.param_groups[0]['lr']}, commit=False)

        # Loop over training set
        for batch, (X_train, y_train_pt, y_train_sc) in enumerate(trainLoader):
            optimizer.zero_grad()
            mi_optimizer.zero_grad()

            # Join labels to list
            y_train = [y_train_pt, y_train_sc]

            # Forward to get feature vector (fv)
            fv = fv_model(X_train)

            # Split fv (feature partioning)
            fv_len_2 = int(fv.size(1)/2)
            fv_pt = fv[:, :fv_len_2]
            fv_sc = fv[:, fv_len_2:]

            # Forward splitted part to model_pt and model_sc
            y_pred_pt = pt_model(fv_pt)
            y_pred_sc = sc_model(fv_sc)

            # Join predictions to list
            y_pred = [y_pred_pt, y_pred_sc]

            if (batch % nr_batches_mi) == 0:
                # feature extractor loss
                pt_sc_loss = crossEntropyLoss_for_MTL(y_pred, y_train)
                mi, mi_loss = mi_model(fv_pt, fv_sc)
                loss = pt_sc_loss + params.mi_lambda * mi
            else:
                # mi loss
                mi, mi_loss = mi_model(fv_pt.detach(), fv_sc.detach())

            # Compute metrics
            accuracy_pt = compute_metrics(y_pred_pt, y_train_pt)
            accuracy_sc = compute_metrics(y_pred_sc, y_train_sc)

            # Log loss and metrics
            wandb.log({
                'epoch': epoch, 'batch': batch, 'pt_sc_loss': loss.item(), 'loss': loss.item(), 'accuracy_pt': accuracy_pt, 'accuracy_sc': accuracy_sc, 'mi': mi.item(), 'mi_loss':mi_loss.item()})

            # Check for divergence
            if torch.isnan(loss) or torch.isinf(loss):
                wandb.run.summary['diverged'] = True
                wandb.run.summary['best_epoch'] = best_epoch
                wandb.run.finish()
                raise ValueError('Training loss diverged')
            
            # Backprop
            fv_model.zero_grad()
            pt_model.zero_grad()
            sc_model.zero_grad()
            mi_model.zero_grad()

            if (batch % nr_batches_mi) == 0:
                mi_loss.detach()
                mi.detach()
                loss.backward()
                optimizer.step()

            # Update mi model
            else:
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
            for batch, (X_val, y_val_pt, y_val_sc) in enumerate(valLoader):
                # Extract features
                fv_val = fv_model(X_val)
                fv_val_len_2 = int(fv_val.size(1)/2)
                fv_val_pt_2 = fv_val[:, :fv_val_len_2]
                fv_val_sc_2 = fv_val[:, fv_val_len_2:]

                # Predict PT and SC
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
            len_val = len(valLoader.dataset.dataset) if "ColouredMNIST".casefold() not in params.training_dataset.casefold() else len(valLoader.dataset)
            len_train = len(trainLoader.dataset.dataset) if "ColouredMNIST".casefold() not in params.training_dataset.casefold() else len(valLoader.dataset)

            y_val_mi_mean = (val_mis * valLoader.batch_size) / len_val
            y_val_mi_mean_loss = (
                val_mi_losses * valLoader.batch_size)/len_val

           
            # Compute val loss
            y_val_preds = [y_val_pred_pt, y_val_pred_sc]
            y_val_trues = [y_val_true_pt.to('cuda'), y_val_true_sc.to('cuda')]
            val_pt_sc_loss = crossEntropyLoss_for_MTL(y_val_preds, y_val_trues)
            val_loss = val_pt_sc_loss + params.mi_lambda * y_val_mi_mean

            # Compute val metrics of PT, SC
            val_accuracy_pt = compute_metrics(
                y_val_pred_pt, y_val_true_pt)
            val_accuracy_sc = compute_metrics(
                y_val_pred_sc, y_val_true_sc)

            # Log val loss and metrics
            wandb.log({
                'epoch': epoch, 'val_loss': val_loss.item(), 'val_pt_sc_loss': val_pt_sc_loss, 'val_mi': y_val_mi_mean.item(), 'val_mi_loss': y_val_mi_mean_loss.item(), 'val_accuracy_pt': val_accuracy_pt, 'val_accuracy_sc': val_accuracy_sc})
            print(
                f'Validation epoch: {epoch:2}/{num_epochs} [{10*batch:6}/{len_train}] Validation loss:  {val_loss.item():.3f} val_acc_pt: {val_accuracy_pt:.3f} val_acc_sc: {val_accuracy_sc:.3f} val_MI: {y_val_mi_mean.item():.3f}')

            if params.training_dataset == "ColouredMNIST":
                if val_loss < best_val_loss:
                    best_epoch = epoch
                    best_accuracy_pt = val_accuracy_pt
                    best_accuracy_sc = val_accuracy_sc
                    best_mi = y_val_mi_mean.item()
                    wandb.log({'best_epoch': best_epoch, 'best_acc_pt': val_accuracy_pt,
                            'best_acc_sc': val_accuracy_sc, 'best_mi': best_mi})

                    fv_modelPathToFile, pt_modelPathToFile, sc_modelPathToFile, mi_modelPathToFile= create_experiments_model_folder(params, fv_model, pt_model, sc_model, mi_model)
            else:

                if (val_accuracy_pt > best_accuracy_pt) & (val_accuracy_sc > best_accuracy_sc):
                    best_epoch = epoch
                    best_accuracy_pt = val_accuracy_pt
                    best_accuracy_sc = val_accuracy_sc
                    best_mi = y_val_mi_mean.item()
                    wandb.log({'best_epoch': best_epoch, 'best_acc_pt': val_accuracy_pt,
                            'best_acc_sc': val_accuracy_sc, 'best_mi': best_mi})

                    fv_modelPathToFile, pt_modelPathToFile, sc_modelPathToFile, mi_modelPathToFile= create_experiments_model_folder(params, fv_model, pt_model, sc_model, mi_model)
                    
    
        fv_model.train()
        pt_model.train()
        sc_model.train()
        mi_model.train()

    # Report
    save_best_models_in_wandb(params.trainType, fv_modelPathToFile, pt_modelPathToFile, sc_modelPathToFile, mi_modelPathToFile)
    training_time = round((time.time() - start_time) / 60)  # in minutes
    print(f'Training done in {training_time} minutes')
    wandb.finish()

    return fv_modelPathToFile, pt_modelPathToFile, sc_modelPathToFile, mi_modelPathToFile
