from Models.FeatureEncoder import FeatureEncoderNetwork, FeatureSelectionNetworkExtended
from Models.ClassificationHeads import PTModel, SCModel
from SaveWandbRuns.initWandb import initWandb, saveWandbRun
from Training.Metrics_Utils import compute_metrics
from torchvision import models
import torch.nn as nn
import torch
import wandb
import os
import sys
sys.path.insert(0, os.getcwd()+'/MIMM/src')


def evaluation_mtl(valLoader, testLoader, testEqualLoader, fv_modelPathToFile, pt_modelPathToFile, sc_modelPathToFile, params_dict, paramsFile, saveFile, syncFile):

    dataLoaders = valLoader + testLoader + testEqualLoader*2
    testTypes = ["Val"] + ["Test"] * \
        len(testLoader) + ["Test_Equal"] + ["Test_Switched"]
    training_dataset = params_dict["training_dataset"]["value"]

    torch.cuda.empty_cache()
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

    for testType, dataLoader in zip(testTypes, dataLoaders):

        params, run = initWandb(params_dict, paramsFile, testType=testType+"_")

        # Save all predictions of all batches for an overall evaluation.
        y_val_pt_trues_as_list = []
        y_val_pt_preds_as_list = []
        y_val_sc_trues_as_list = []
        y_val_sc_preds_as_list = []

        # Evaluation
        with torch.no_grad():

            # Batchwise evaluation
            for batch, (X_val, y_val_pt, y_val_sc) in enumerate(dataLoader):

                # Estimated CL and SC for one batch
                if "Switched" in testType:
                    init_y_val_pt = y_val_pt[:]
                    y_val_pt = y_val_sc
                    y_val_sc = init_y_val_pt

                fv = fv_model(X_val)
                len_fv = int(fv.size(1)/2)
                fv_pt = fv[:, :len_fv]
                fv_sc = fv[:, len_fv:]

                # Estimation/Predict on both MTL classes
                y_val_pt_preds_as_list.append(pt_model(fv_pt))
                y_val_sc_preds_as_list.append(sc_model(fv_sc))
                y_val_pt_trues_as_list.append(y_val_pt)
                y_val_sc_trues_as_list.append(y_val_sc)

        # Evaluation of all batches
        y_val_pt_preds = torch.cat(y_val_pt_preds_as_list)
        y_val_sc_preds = torch.cat(y_val_sc_preds_as_list)
        y_val_pt_trues = torch.cat(y_val_pt_trues_as_list)
        y_val_sc_trues = torch.cat(y_val_sc_trues_as_list)

        # Compute val metrics
        val_accuracy_pt = compute_metrics(
            y_val_pt_preds, y_val_pt_trues)
        val_accuracy_sc = compute_metrics(
            y_val_sc_preds, y_val_sc_trues)

        # Log the evaluation
        wandb.log({'val_accuracy_pt': val_accuracy_pt,
                  'val_accuracy_sc': val_accuracy_sc})
        wandb.finish()
        print(testType + " val_acc CL " + str(val_accuracy_pt))
        print(testType + " val_acc SC " + str(val_accuracy_sc))
        saveWandbRun(run, saveFile, syncFile, params_dict)

    print("done")
