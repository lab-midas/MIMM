import os
import wandb
import torch
from config.Load_Parameter import save_parameters_as_global_dict
from datetime import datetime


def initWandb(params_dict, params_file, testType = "", mi_batches = 0, mi= 0):
    project = set_project_for_wandb(params_dict)
    group = set_group_for_wandb(params_dict, testType)
    namerun = set_name_run_for_wandb(params_dict, testType=testType, mi_batches=mi_batches, mi=mi)
    # init wandb
    wandb.login()
    run = wandb.init(project=project,
                     config=os.path.expanduser('~')+'/MIMM/src/config/' + params_file, name=namerun, group=group)
    config = wandb.config
    save_parameters_as_global_dict(config)
    return config, run

def upload_wandb(syncFile):
    with open(syncFile, 'r') as f:
        for line in f:
            os.system(line)

def create_experiments_model_folder(params, fv_model, pt_model, sc_model="", mi_model=""):
    modelPath = os.path.expanduser('~')+"/MIMM/src/" + datetime.now().strftime('Experiments/' +
                                                                 str(params.k)+"_"+params.trainType+'/%H_%M_%d_%m_%Y')
    if not os.path.exists(modelPath):
        os.chdir(os.path.expanduser('~')+'/MIMM/src')
        os.makedirs(modelPath)

    if "MTL_MI" in params.trainType:
        fv_modelPathToFile = modelPath+'/' + params.trainType+'_fv_model.pth'
        pt_modelPathToFile = modelPath+'/' + params.trainType+'_pt_model.pth'
        sc_modelPathToFile = modelPath+'/' + params.trainType+'_sc_model.pth'
        mi_modelPathToFile = modelPath+'/' + params.trainType+'_mi_model.pth'

        run = wandb.init(job_type='val')
        torch.save(fv_model.state_dict(), fv_modelPathToFile)
        torch.save(pt_model.state_dict(), pt_modelPathToFile)
        torch.save(sc_model.state_dict(), sc_modelPathToFile)
        torch.save(mi_model.state_dict(), mi_modelPathToFile)

        return fv_modelPathToFile, pt_modelPathToFile, sc_modelPathToFile, mi_modelPathToFile

    else:
        raise ValueError("No valid trainType.")  

def save_best_models_in_wandb(trainType, fv_modelPathToFile, pt_modelPathToFile, sc_modelPathToFile, mi_modelPathToFile="" ):
    if "MTL_MI" in trainType:
        fv_artifact = wandb.Artifact(trainType+'_fv_model', type='model')
        pt_artifact = wandb.Artifact(trainType+'_pt_model', type='model')
        sc_artifact = wandb.Artifact(trainType+'_sc_model', type='model')
        mi_artifact = wandb.Artifact(trainType+'_mi_model', type='model')

        fv_artifact.add_file(fv_modelPathToFile)
        pt_artifact.add_file(pt_modelPathToFile)
        sc_artifact.add_file(sc_modelPathToFile)
        mi_artifact.add_file(mi_modelPathToFile)

        wandb.log_artifact(fv_artifact)
        wandb.log_artifact(pt_artifact)
        wandb.log_artifact(sc_artifact)
        wandb.log_artifact(mi_artifact)

        wandb.join()
    else:
        raise ValueError("No valid trainType.")



def saveWandbRun(run, saveFile, syncFile, params_dict, k=""):
    project = set_project_for_wandb(params_dict)
    group = set_group_for_wandb(params_dict)
    namerun = set_name_run_for_wandb(params_dict)
    with open(saveFile, 'a+') as f1, open(syncFile, 'a+') as f2:
        f1.write("\n"+str(datetime.now()) + "     " +
                 project+" " + group + " " + namerun + " k: "+k + "\n")
        f1.write(run.dir)
        f2.write("wandb sync " + run.dir[:-6] + "\n")
    wandb.finish()


def set_name_run_for_wandb(params_dict, confounding_ratio="", mi_batches=0, mi=0, testType=""):
    """Create the name of the run. Name used for wandb.
    """

    trainType, confoundingRatio, nonConfoundingRatio = extract_params(
        params_dict, confounding_ratio)
    if "MI" in trainType:
        mi = params_dict["mi_lambda"]["value"]
        mi_batches = params_dict["nr_batches_MI_train"]["value"]
        
        mi_str = get_static_strings()
        namerun = testType+str(params_dict["feature_vector_length"]["value"])+"_"+params_dict["training_dataset"]["value"]+"_" +trainType+"_" + \
            confoundingRatio+"_"+nonConfoundingRatio + \
            "_"+mi_str+"_"+str(mi_batches)+"_"+str(mi)
    else:
        namerun = testType+str(params_dict["feature_vector_length"]["value"])+"_"+params_dict["training_dataset"]["value"]+"_" +trainType + \
            "_"+confoundingRatio+"_"+str((100-int(confoundingRatio)))
    return namerun


def set_group_for_wandb(params_dict, testType = ""):
    if testType == "":
        return params_dict["training_dataset"]["value"]
    else:
        return "Eval_" + params_dict["training_dataset"]["value"]


def set_project_for_wandb(params_dict):
    return params_dict["wandbProjectName"]["value"]


def get_static_strings():
    """Creates constant strings. To generate name of run.
    """
    mi_str = "MI"
    return mi_str


def extract_params(params_dict, confounding_ratio=""):
    """ Extract parameters to generate the name of a run.

    Args:
        params_dict (_type_): dict with all variabels from yaml.

    Returns:
        extracted variables
    """
    trainType = params_dict["trainType"]["value"]
    if confounding_ratio == "":
        confoundingRatio = int(params_dict["confoundingRatio"]["value"]*100)
        nonConfoundingRatio = int(100-confoundingRatio)
    else:
        confoundingRatio = int(confounding_ratio*100)
        nonConfoundingRatio = int((1-confounding_ratio)*100)

    return trainType, str(confoundingRatio), str(nonConfoundingRatio)
