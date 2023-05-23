import os
import sys
import torch
from Training.MTL_MI import mtl_mi_train
from Evaluation.Evaluation_MTL import evaluation_mtl
from Dataset.Create_DataLoader import init_morphoMNIST_training
from config.Load_Parameter import load_preset_parameters
from SaveWandbRuns.loadPathsToSaveRun import loadPaths
from SaveWandbRuns.initWandb import initWandb, saveWandbRun, upload_wandb

def main():

    os.environ['WANDB_MODE'] = 'offline'

    # Load parameter file
    if len(sys.argv) == 2:
        params_file = sys.argv[1]
    else:
        params_file = "config.yml"

    # Save parameters as dict
    params_dict = load_preset_parameters(params_file)
    training_dataset = params_dict["training_dataset"]["value"]
    trainType = params_dict["trainType"]["value"]

    #  Create dataloader
    if training_dataset == "MorphoMNIST":
        trainLoader, valLoader, testLoaders, testEqualLoader = init_morphoMNIST_training(params_dict, training_dataset)
    else:
        raise ValueError('No valid dataset selected.')
    

    # Initialize and start training 
    if "MTL_MI" in trainType:
        # train MIMM
        syncFile=set_mtl_mi_train(trainLoader, valLoader, testLoaders, testEqualLoader, params_dict, params_file)
    else:
        raise ValueError('No valid training selected')
    
    # Upload offline runs to wandb
    upload_wandb(syncFile)

def set_mtl_mi_train(trainLoader, valLoader, testLoaders, testEqualLoader, params_dict, params_file):

    # Load paths to save runs and wandb path
    saveFile, syncFile = loadPaths(params_dict)
    for mi_batches in params_dict["nr_batches_MI_train_multiple"]["value"]:
        for mi in params_dict["mi_lambdas"]["value"]:
            
            for k in range(5):
                # Init wandb
                params_dict["mi_lambda"]["value"] = mi
                params_dict["nr_batches_MI_train"]["value"] = mi_batches

                config, run  = initWandb(params_dict, params_file)
                config.update({"mi_lambda" : mi}, allow_val_change=True)
                config.update({"nr_batches_MI_train" : mi_batches}, allow_val_change=True)
                config.update({"k" : k}, allow_val_change=True)
               
                # Start training
                fv_modelPathToFile, pt_modelPathToFile, sc_modelPathToFile, _ = mtl_mi_train(trainLoader, valLoader)
                torch.cuda.empty_cache()

                saveWandbRun(run, saveFile, syncFile, params_dict)
                evaluation_mtl([valLoader], testLoaders, [testEqualLoader], fv_modelPathToFile, pt_modelPathToFile, sc_modelPathToFile, params_dict, params_file, saveFile, syncFile)
                torch.cuda.empty_cache()
    return syncFile

if __name__ == '__main__':
    main()