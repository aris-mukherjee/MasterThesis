from unittest import case
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import nibabel as nib
import os
import glob
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import morphology
import scipy.ndimage.interpolation
from skimage import transform
import logging
import utils
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from skimage import io
from skimage import color
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optimizer
import copy
import pickle





def test_single_volume_FETS_train_eval(image, label, net, i2n_module_t1, i2n_module_t1ce, i2n_module_t2, i2n_module_flair, use_tta, tta_epochs, writer, layer_names_for_stats, classes, dataset, optim, model_type, seed, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.cpu().detach().numpy(), label.cpu().detach().numpy()   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    i2n_module_t1.to(device)
    i2n_module_t1ce.to(device)
    i2n_module_t2.to(device)
    i2n_module_flair.to(device)
    net.to(device)

    
    prediction = np.zeros_like(label[:, :, :, 0])
    

    

    #entropy_loss = HLoss()  #Entropy Loss
    
    
    batch_size = 5
    activations = {}
    gauss_param = {}
    mu = {}
    var = {}

    batch_mean_list = []
    batch_var_list = []
    

    iid_samples_seen = {}
           
    for name, m in net.named_modules():
        if name in layer_names_for_stats:
            hook_fn = _hook_store_activations(name, activations)
            m.register_forward_hook(hook_fn)

    shuffled_slices = np.random.permutation(image.shape[0])
    for slices in chunks(shuffled_slices, batch_size):
        slice = image[slices, :, :]
        input = torch.from_numpy(slice).float().cuda()
        input = input.permute(0, 3, 1, 2)
        i2n_module_t1.eval()
        i2n_module_t1ce.eval()
        i2n_module_t2.eval()
        i2n_module_flair.eval()
        net.eval()
        for param in i2n_module_t1.parameters():
            param.requires_grad = False

        for param in i2n_module_t1ce.parameters():
            param.requires_grad = False

        for param in i2n_module_t2.parameters():
            param.requires_grad = False

        for param in i2n_module_flair.parameters():
            param.requires_grad = False    

        for param in net.parameters():
            param.requires_grad = False

        
        norm_output_t1 = i2n_module_t1(input[:, 0, :, :].unsqueeze(1))
        norm_output_t1ce = i2n_module_t1ce(input[:, 1, :, :].unsqueeze(1))
        norm_output_t2 = i2n_module_t2(input[:, 2, :, :].unsqueeze(1))
        norm_output_flair = i2n_module_flair(input[:, 3, :, :].unsqueeze(1))

        norm_output = torch.cat((norm_output_t1, norm_output_t1ce, norm_output_t2, norm_output_flair), 1)

        


        if model_type == 'UNET':

            outputs = net(norm_output)

            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            i=0
            for s in slices:
                prediction[s] = out[i]
                i =i+1

            

            for name in layer_names_for_stats:
                C = activations[name].size(1)

                mu[name] = torch.zeros(C).to(device)
                var[name] = torch.zeros(C).to(device)

                iid_samples_seen[name] = 0

            #   Feature KDE computation
            # ------------------------------
            for name in layer_names_for_stats:

                # Act: (N, C, H, W) - consider as iid along N, H & W
                N, C, H, W = activations[name].size()
                iids = activations[name].permute(1, 0, 2, 3).flatten(1)

                N1 = iid_samples_seen[name]
                N2 = iids.size(1)

                batch_mean = torch.mean(iids, dim=1)
                batch_sqr_mean = torch.mean(iids**2, dim=1)

                mu[name] = (
                    N1 / (N1 + N2) * mu[name] +
                    N2 / (N1 + N2) * batch_mean
                )
                var[name] = (
                    N1 / (N1 + N2) * var[name] +
                    N2 / (N1 + N2) * batch_sqr_mean
                )

                iid_samples_seen[name] += N2


                batch_mean_list.append(mu[name])
                batch_var_list.append(var[name])

                #gauss_param[name] = torch.stack([mu[name], var[name]], dim=1)

         #num_experts = count_num_experts(layer_names_for_stats, activations)
    
        elif model_type == 'UNWT':   


            outputs, batch_mean, batch_sqr_mean, batch_var, batch_sqr_var = net(norm_output)

            batch_mean_list.append(batch_mean)
            batch_var_list.append(batch_var)

            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            i=0
            for s in slices:
                prediction[s] = out[i]
                i =i+1


              
    case_mean = sum(batch_mean_list)/len(batch_mean_list)
    case_var = sum(batch_var_list)/len(batch_var_list)

    gauss_param[case] = torch.stack([case_mean, case_var], dim=1)
            
            
    save_pdfs(gauss_param, case, model_type) 
       
    

    # ============================
    # Calculate Dice & Hausdorff
    # ============================         
    metric_list_whole_tumor = []
    metric_list_enhancing_tumor = []
    metric_list_tumor_core = []


    #whole tumor 
    print("WHOLE TUMOR (ALL LABELS)")
    #for i in range(0, classes):
    metric_list_whole_tumor.append(calculate_metric_percase(prediction > 0, label[:, :, :, 0] > 0))


    print("TUMOR CORE (LABELS 1 and 4")
    prediction = np.array(prediction)
    label = np.array(label)
    prediction[np.where(prediction == 2)] = 0
    label[np.where(label == 2)] = 0

    metric_list_tumor_core.append(calculate_metric_percase(prediction > 0, label[:, :, :, 0] > 0))

    print("ENHANCING TUMOR (ONLY LABEL 4")
    prediction = np.array(prediction)
    label = np.array(label)
    prediction[np.where(prediction < 3)] = 0
    label[np.where(label < 3)] = 0



    metric_list_enhancing_tumor.append(calculate_metric_percase(prediction > 0, label[:, :, :, 0] > 0))

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    img_itk.SetSpacing((1, 1, z_spacing))
    prd_itk.SetSpacing((1, 1, z_spacing))
    lab_itk.SetSpacing((1, 1, z_spacing))
    sitk.WriteImage(prd_itk, test_save_path + '/'+"{}".format(case) + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + '/'+"{}".format(case) + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + '/'+"{}".format(case) + "_gt.nii.gz")


    return metric_list_whole_tumor, metric_list_enhancing_tumor, metric_list_tumor_core



def load_pdfs(cache_name):
    with open(cache_name, 'rb') as f:
        cache = pickle.load(f)
        pdfs = cache['pdfs']
        num_experts = cache['num_experts']

def save_pdfs(gauss_param, case, model_type):
    cache_name = f'/scratch_net/biwidl217_second/arismu/Data_MT/data_FoE/{model_type}/SD_data_{case}_{model_type}.pkl'
    with open(cache_name, 'wb') as f:
        pickle.dump(gauss_param, f)


def count_num_experts(layer_names_for_stats, activations):
    num_1d_experts = 0
    for name in layer_names_for_stats:
        num_1d_experts += activations[name].size(1)
    
    return num_1d_experts


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    print(f"Predsum: {pred.sum()}")
    print(f"GtSum: {gt.sum()}")
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def _hook_store_activations(module_name, activations):
    """ """
    def hook_fn(m, i, o):   
        activations[module_name] = o
    return hook_fn


def chunks(lst, n):
    """ Yield successive n-sized chunks from lst """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]