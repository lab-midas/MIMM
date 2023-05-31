import os
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.manifold import TSNE 

import config.Load_Parameter

from Models.FeatureEncoder import FeatureEncoderNetwork, FeatureSelectionNetworkExtended
from Models.ClassificationHeads import PTModel, SCModel

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])

def numpy(t):
    return t.cpu().numpy()

def tsne(trainLoader, testLoader,fv_modelPathToFile, pt_modelPathToFile,sc_modelPathToFile, params):
    
    # Get the hyperparameters saved in yaml.
    torch.manual_seed(params.randomSeed)
    training_dataset = params.training_dataset.casefold()
    # Set up model
    if "MorphoMNIST".casefold() in training_dataset:
        fv_model = FeatureEncoderNetwork().cuda()
    elif "FashionMNIST".casefold() in training_dataset:
        fv_model = FeatureSelectionNetworkExtended().cuda()
    else:
        raise ValueError('No valid trainType selected. No model found')
    
    fv_model.load_state_dict(torch.load(fv_modelPathToFile))
    fv_model.cuda()
    fv_model.eval()

    pt_model = PTModel()
    if pt_modelPathToFile != "":
        pt_model.load_state_dict(torch.load(pt_modelPathToFile))
        pt_model.cuda()
        pt_model.eval()
    if sc_modelPathToFile != "":
        sc_model = SCModel()
        sc_model.load_state_dict(torch.load(sc_modelPathToFile))
        sc_model.cuda()
        sc_model.eval()


    # Save images in list
    samples = []
    for imgInit, l1, l2  in testLoader.dataset.dataset:
        img = imgInit.clone()
        img.requires_grad_()
        samples.append((img, l1, l2))

    fv_list_pt, fv_list_sc, l1_list, l2_list = [], [], [], []
    for img, l1, l2 in samples:
        fv = fv_model(img.reshape((1,1,params.input_img_size,params.input_img_size)))
        # small number
        fv_list_pt.append(fv[:,:2].detach().cpu().numpy())
        fv_list_sc.append(fv[:,2:].detach().cpu().numpy())

        l1_list.append(l1)
        l2_list.append(l2)

    fv_arr_pt = np.stack(fv_list_pt, axis = 1)[0,:,:]
    fv_arr_sc = np.stack(fv_list_sc, axis = 1)[0,:,:]

    # Create TSNE Vectors
    tsne_mtl =  TSNE(n_components=2, random_state=0).fit_transform(fv_arr_pt)
    tsne_mtl_sc =  TSNE(n_components=2, random_state=0).fit_transform(fv_arr_sc)

    # Create plots
    if params.training_dataset != "FashionMNIST":
        Path(os.path.expanduser('~')+"/MIMM/src/Plots/"+params.training_dataset).mkdir(parents=True, exist_ok=True)
        if params.training_dataset == "MorphoMNIST":
            labels_PT = ["Small Number (0-4)", "High Number (5-9)"]
            labels_SC = ["Thin", "Thick"]            

        # Vector CL, Label CF
        create_tsne_plot(l2_list, tsne_mtl, plottitle=params.trainType, plotname=params.training_dataset+'/'+params.trainType+"_V_PT_L_SC_"+str(params.k), label=labels_SC)
        # Vector CL, Label CL
        create_tsne_plot(l1_list, tsne_mtl, plottitle=params.trainType, plotname=params.training_dataset+'/'+params.trainType+"_V_PT_L_PT_"+str(params.k), label=labels_PT)
        # Vector CF, Label CL
        create_tsne_plot(l1_list, tsne_mtl_sc, plottitle=params.trainType, plotname=params.training_dataset+'/'+params.trainType+"_V_SC_L_PT_"+str(params.k), label=labels_PT)
        create_tsne_plot(l2_list, tsne_mtl_sc, plottitle=params.trainType, plotname=params.training_dataset+'/'+params.trainType+"_V_SC_L_SC_"+str(params.k), label=labels_SC)
    
    elif params.training_dataset == "FashionMNIST":
        Path(os.path.expanduser('~')+"/MIMM/src/Plots/"+params.training_dataset).mkdir(parents=True, exist_ok=True)

        # Vector CL, Label CF
        create_tsne_plot_multiple_classes(l2_list, tsne_mtl, plottitle=params.trainType, plotname=params.training_dataset+'/'+params.trainType+"_V_PT_L_SC_"+str(params.k))
        # Vector CL, Label CL
        create_tsne_plot_multiple_classes(l1_list, tsne_mtl, plottitle=params.trainType, plotname=params.training_dataset+'/'+params.trainType+"_V_PT_L_PT_"+str(params.k))
        # Vector CF, Label CL
        create_tsne_plot_multiple_classes(l1_list, tsne_mtl_sc, plottitle=params.trainType, plotname=params.training_dataset+'/'+params.trainType+"_V_SC_L_PT_"+str(params.k))
        # Vector CF, Label CF
        create_tsne_plot_multiple_classes(l2_list, tsne_mtl_sc, plottitle=params.trainType, plotname=params.training_dataset+'/'+params.trainType+"_V_SC_L_SC_"+str(params.k))
  
def create_tsne_plot(label_list, tsne_vector, plottitle, plotname, label):
    params = config.Load_Parameter.params
    l0_mtl_x, l1_mtl_x, l0_mtl_y, l1_mtl_y= [], [], [], []
    for l, elem in zip(label_list, tsne_vector):
        if l == 0:
            l0_mtl_x.append(elem[0])
            l0_mtl_y.append(elem[1])
        else:
            l1_mtl_x.append(elem[0])
            l1_mtl_y.append(elem[1])
    plt.figure(figsize=(8,6))
    plt.scatter(l0_mtl_x, l0_mtl_y, color = "r", s= [3], label=label[0])
    plt.scatter(l1_mtl_x, l1_mtl_y, color = "b", s= [3], label=label[1])
    plt.title(plottitle, fontsize=16)
    plt.legend(fontsize=16)
    plt.xlabel("Dimension 1", fontsize=16)
    plt.ylabel("Dimension 2", fontsize=16)
    plt.savefig(os.path.expanduser('~')+"/MIMM/src/Plots/"+plotname+".svg")
    plt.close()

def create_tsne_plot_multiple_classes(label_list, tsne_vector, plottitle, plotname):
    params = config.Load_Parameter.params

    l_mtl_x = [[] for i in range(10)]
    l_mtl_y = [[] for i in range(10)]

    for l, elem in zip(label_list, tsne_vector):
        l_mtl_x[l].append(elem[0])
        l_mtl_y[l].append(elem[1])

    plt.figure(figsize=(8,6))
    col = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'lime', 'purple', 'orange']
    for i in range(10):
        plt.scatter(l_mtl_x[i], l_mtl_y[i], color = col[i], alpha=0.3, s= [3])

    plt.title(plottitle, fontsize=16)
    plt.xlabel("Dimension 1", fontsize=16)
    plt.ylabel("Dimension 2", fontsize=16)
    plt.savefig(os.path.expanduser('~')+"/MIMM/src/Plots/"+plotname+".svg")
    plt.close()
