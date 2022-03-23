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

writer = SummaryWriter('/scratch_net/biwidl217_second/arismu/Tensorboard/' + 'Test_Images_Output') 
i = 0

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


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



def  test_single_volume(image, label, net, classes, dataset, optim, model_type, seed, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.cpu().detach().numpy(), label.cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)

        foreground_list = []
        label_list = []

        # ============================
        # Perform the prediction slice by slice
        # ============================ 

        for ind in range(image.shape[0]):  
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            
            #utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + '4_test.nii.gz', data = slice, affine = np.eye(4))
            #utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + '4_label.nii.gz', data = label[:, :, ind], affine = np.eye(4))

            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():

                outputs = net(input)
                out_soft = torch.softmax(outputs, dim=1)
                out_hard = (out_soft>0.5).float()
                out_hard_argmax = torch.argmax(out_hard, dim=1).squeeze(0) 
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                #color_map = torch.tensor([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
                rgb = np.zeros((256, 256, 3))
                #io.imshow(color.label2rgb(out, slice))
                """ if (dataset == 'NCI' and case == 5 and ind == 10) or (dataset == 'UCL' and case == 3 and ind == 14) or (dataset == 'HK' and case == 2 and ind == 13) or (dataset == 'BIDMC' and case == 5 and ind == 26):
                    
                    for i in range(3):
                        out_soft_squeezed = out_soft.squeeze(0)
                        out_soft_squeezed = out_soft_squeezed[i, :, :]
                        out_soft_squeezed = out_soft_squeezed.cpu().detach().numpy()
                        plt.imshow(out_soft_squeezed, cmap = 'gray', vmin = 0, vmax = 1)
                        plt.savefig('/scratch_net/biwidl217_second/arismu/Data_MT/2022/%s_%s_%s_hard_pred_case%s_slice%s_channel%s_seed%s.png' % (dataset, model_type, optim, case, ind, i, seed)) """

                out_soft_sq = out_soft.squeeze(0)
                out_soft_foreground = out_soft_sq[1, :, : ] + out_soft_sq[2, :, : ]
                out_soft_foreground = out_soft_foreground.flatten()
                out_soft_foreground = out_soft_foreground.cpu().detach().numpy()
                foreground_list.append(out_soft_foreground)
                

                label_temp = label[ind]
                label_temp = label_temp.flatten()
                label_temp[np.where(label_temp > 0)] = 1
                label_list.append(label_temp)
                                                                                    

                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred

    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    
    label_list_arr = np.array(label_list)
    label_list_arr = label_list_arr.flatten()

    foreground_list_arr = np.array(foreground_list)
    foreground_list_arr = foreground_list_arr.flatten()

    #test_label = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    #test_pred = [0.3, 0.2, 0.9, 0.4, 0.7, 0.3, 0.5, 0.6, 0.3, 0.9]
    #disp = CalibrationDisplay.from_predictions(label_list_arr, foreground_list_arr)
    #disp = CalibrationDisplay.from_predictions(test_label, test_pred)
    #plt.show()
    #plt.savefig(f'/scratch_net/biwidl217_second/arismu/Data_MT/plots/{dataset}_case{case}.png')

    


    # ============================
    # Calculate Dice & Hausdorff
    # ============================         
    metric_list = []
    


    for i in range(0, classes):
        metric_list.append(calculate_metric_percase(prediction > 0, label > 0))
        
    
         

    # ============================
    # Save images, predictions and ground truths
    # ============================
    
    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+"{}".format(case) + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+"{}".format(case) + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+"{}".format(case) + "_gt.nii.gz")
    return metric_list, foreground_list_arr, label_list_arr




# ===================================================
# ===================================================
def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False

# ===================================================
# ===================================================
def get_latest_model_checkpoint_path(folder, name):
    '''
    Returns the checkpoint with the highest iteration number with a given name
    :param folder: Folder where the checkpoints are saved
    :param name: Name under which you saved the model
    :return: The path to the checkpoint with the latest iteration
    '''

    iteration_nums = []
    for file in glob.glob(os.path.join(folder, '%s*.meta' % name)):

        file = file.split('/')[-1]
        file_base, postfix_and_number, rest = file.split('.')[0:3]
        it_num = int(postfix_and_number.split('-')[-1])

        iteration_nums.append(it_num)

    latest_iteration = np.max(iteration_nums)

    return os.path.join(folder, name + '-' + str(latest_iteration))

# ===================================================
# ===================================================
def load_nii(img_path):

    '''
    Shortcut to load a nifti file
    '''

    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header

# ===================================================
# ===================================================
def save_nii(img_path, data, affine, header=None):
    '''
    Shortcut to save a nifty file
    '''
    if header == None:
        nimg = nib.Nifti1Image(data, affine=affine)
    else:
        nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)

# ===================================================
# ===================================================
def normalise_image(image, norm_type = 'div_by_max'):
    '''
    make image zero mean and unit standard deviation
    '''
    if norm_type == 'zero_mean':
        img_o = np.float32(image.copy())
        m = np.mean(img_o)
        s = np.std(img_o)
        normalized_img = np.divide((img_o - m), s)
        
    elif norm_type == 'div_by_max':
        perc1 = np.percentile(image,1)
        perc99 = np.percentile(image,99)
        normalized_img = np.divide((image - perc1), (perc99 - perc1))
        normalized_img[normalized_img < 0] = 0.0
        normalized_img[normalized_img > 1] = 1.0
    
    return normalized_img
    
# ===============================================================
# ===============================================================
def crop_or_pad_slice_to_size(slice, nx, ny):
    x, y = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped

# ===============================================================
# ===============================================================
def crop_or_pad_slice_to_size_1hot(slice, nx, ny):
    
    x, y, c = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny, :]
    else:
        slice_cropped = np.zeros((nx, ny, c))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :, :] = slice[:, y_s:y_s + ny, :]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y, :] = slice[x_s:x_s + nx, :, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y, :] = slice[:, :, :]

    return slice_cropped

# ===============================================================
# ===============================================================
def crop_or_pad_volume_to_size_along_x(vol, nx):
    
    x = vol.shape[0]
    x_s = (x - nx) // 2
    x_c = (nx - x) // 2

    if x > nx: # original volume has more slices that the required number of slices
        vol_cropped = vol[x_s:x_s + nx, :, :]
    else: # original volume has equal of fewer slices that the required number of slices
        vol_cropped = np.zeros((nx, vol.shape[1], vol.shape[2]))
        vol_cropped[x_c:x_c + x, :, :] = vol

    return vol_cropped

# ===============================================================
# ===============================================================
def crop_or_pad_volume_to_size_along_x_1hot(vol, nx):
    
    x = vol.shape[0]
    x_s = (x - nx) // 2
    x_c = (nx - x) // 2

    if x > nx: # original volume has more slices that the required number of slices
        vol_cropped = vol[x_s:x_s + nx, :, :, :]
    else: # original volume has equal of fewer slices that the required number of slices
        vol_cropped = np.zeros((nx, vol.shape[1], vol.shape[2], vol.shape[3]))
        vol_cropped[x_c:x_c + x, :, :, :] = vol
        vol_cropped[:x_c, :, :, 0] = 1
        vol_cropped[x_c+x:, :, :, 0] = 1

    return vol_cropped

# ===============================================================
# ===============================================================
def crop_or_pad_volume_to_size_along_z(vol, nz):
    
    z = vol.shape[2]
    z_s = (z - nz) // 2
    z_c = (nz - z) // 2

    if z > nz: # original volume has more slices that the required number of slices
        vol_cropped = vol[:, :, z_s:z_s + nz]
    else: # original volume has equal of fewer slices that the required number of slices
        vol_cropped = np.zeros((vol.shape[0], vol.shape[1], nz))
        vol_cropped[:, :, z_c:z_c + z] = vol
    
    return vol_cropped

# ===============================================================
# Group the segmentation classes into the required categories 
# ===============================================================
def group_segmentation_classes(seg_mask):
    
    seg_mask_modified = group_segmentation_classes_15(seg_mask)
    return seg_mask_modified

# ===============================================================
# Group the segmentation classes into the required categories 
# ===============================================================
def group_segmentation_classes_15(a):
    """
    Args:
    label_data : Freesurfer generated Labels Data of a 3D MRI scan.
    Returns:
    relabelled_data
    """
    
    background_ids = [0] # [background]
    csf_ids = [24] # [csf]
    brainstem_ids = [16] # [brain stem]    
    cerebellum_wm_ids = [7, 46]
    cerebellum_gm_ids = [8, 47]
    cerebral_wm_ids = [2, 41, 251, 252, 253, 254, 255]
    cerebral_gm_ids = np.arange(1000, 3000)
    cerebral_cortex_ids = [3,42]
    thalamus_ids = [10, 49]
    hippocampus_ids = [17, 53]
    amygdala_ids = [18, 54]
    ventricle_ids = [4, 43, 14, 15, 72] # lat, 3rd, 4th, 5th
    choroid_plexus_ids = [31, 63]
    caudate_ids = [11, 50]
    putamen_ids = [12, 51]
    pallidum_ids = [13, 52]
    accumbens_ids = [26, 58]
    ventral_DC_ids = [28, 60]
    misc_ids = [5, 44, 30, 62, 77, 80, 85] # inf lat ventricle, right, left vessel, hypointensities, optic-chiasm
    
    a = np.array(a, dtype = 'uint16')
    b = np.zeros((a.shape[0], a.shape[1], a.shape[2]), dtype = 'uint16')

    unique_ids = np.unique(a)    
    # print("Unique labels in the original segmentation mask:", unique_ids)
    
    for i in unique_ids:
        if (i in cerebral_gm_ids): b[a == i] = 3
        elif (i in cerebral_cortex_ids): b[a == i] = 3
        elif (i in accumbens_ids): b[a == i] = 3
        elif (i in background_ids): b[a == i] = 0
        elif (i in cerebellum_gm_ids): b[a == i] = 1
        elif (i in cerebellum_wm_ids): b[a == i] = 2
        elif (i in cerebral_wm_ids): b[a == i] = 4
        elif (i in misc_ids): b[a == i] = 4
        elif (i in thalamus_ids): b[a == i] = 5
        elif (i in hippocampus_ids): b[a == i] = 6
        elif (i in amygdala_ids): b[a == i] = 7
        elif (i in ventricle_ids): b[a == i] = 8    
        elif (i in choroid_plexus_ids): b[a == i] = 8    
        elif (i in caudate_ids): b[a == i] = 9
        elif (i in putamen_ids): b[a == i] = 10
        elif (i in pallidum_ids): b[a == i] = 11
        elif (i in ventral_DC_ids): b[a == i] = 12
        elif (i in csf_ids): b[a == i] = 13
        elif (i in brainstem_ids): b[a == i] = 14
        else:
            print('unknown id:', i)
            print('num_voxels:', np.shape(np.where(a==i))[1])
        
    print("Unique labels in the modified segmentation mask: ", np.unique(b))
    
    return b
    
# ==================================================================
# taken from: https://gist.github.com/erniejunior/601cdf56d2b424757de5
# ==================================================================   
def elastic_transform_image_and_label(image, # 2d
                                      label,
                                      sigma,
                                      alpha,
                                      random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    
    # random_state.rand(*shape) generate an array of image size with random uniform noise between 0 and 1
    # random_state.rand(*shape)*2 - 1 becomes an array of image size with random uniform noise between -1 and 1
    # applying the gaussian filter with a relatively large std deviation (~20) makes this a relatively smooth deformation field, but with very small deformation values (~1e-3)
    # multiplying it with alpha (500) scales this up to a reasonable deformation (max-min:+-10 pixels)
    # multiplying it with alpha (1000) scales this up to a reasonable deformation (max-min:+-25 pixels)
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    distored_label = map_coordinates(label, indices, order=0, mode='reflect').reshape(shape)
    
    return distored_image, distored_label

# ==================================================================
# taken from: https://gist.github.com/erniejunior/601cdf56d2b424757de5
# ==================================================================   
def elastic_transform_label(label, # 2d
                            sigma,
                            alpha,
                            random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = label.shape
    
    # random_state.rand(*shape) generate an array of image size with random uniform noise between 0 and 1
    # random_state.rand(*shape)*2 - 1 becomes an array of image size with random uniform noise between -1 and 1
    # applying the gaussian filter with a relatively large std deviation (~20) makes this a relatively smooth deformation field, but with very small deformation values (~1e-3)
    # multiplying it with alpha (500) scales this up to a reasonable deformation (max-min:+-10 pixels)
    # multiplying it with alpha (1000) scales this up to a reasonable deformation (max-min:+-25 pixels)
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    distored_label = map_coordinates(label, indices, order=0, mode='reflect').reshape(shape)
    
    return distored_label

# ==================================================================
# taken from: https://gist.github.com/erniejunior/601cdf56d2b424757de5
# ==================================================================   
def elastic_transform_label_3d(label, # 3d
                               sigma,
                               alpha,
                               random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = (label.shape[1], label.shape[2])    
    
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    distored_label = np.copy(label)
    # save deformation field for all slices of the image
    for zz in range(label.shape[0]):
        distored_label[zz,:,:] = map_coordinates(label[zz,:,:], indices, order=0, mode='reflect').reshape(shape)
    
    return distored_label

# ==================================================================
# taken from: https://gist.github.com/erniejunior/601cdf56d2b424757de5
# ==================================================================   
def elastic_transform_label_pair_3d(label1,
                                    label2, # 3d
                                    sigma,
                                    alpha,
                                    random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = (label1.shape[1], label1.shape[2])    
    
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    distored_label1 = np.copy(label1)
    distored_label2 = np.copy(label2)
    
    # save deformation field for all slices of the 3d image
    for zz in range(label1.shape[0]):
        distored_label1[zz,:,:] = map_coordinates(label1[zz,:,:], indices, order=0, mode='reflect').reshape(shape)
        distored_label2[zz,:,:] = map_coordinates(label2[zz,:,:], indices, order=0, mode='reflect').reshape(shape)
    
    return distored_label1, distored_label2

# ===========================      
# data augmentation: random elastic deformations, translations, rotations, scaling
# data augmentation: gamma contrast, brightness (one number added to the entire slice), additive noise (random gaussian noise image added to the slice)
# ===========================        
def do_data_augmentation(images,
                         labels,
                         data_aug_ratio,
                         sigma,
                         alpha,
                         trans_min,
                         trans_max,
                         rot_min,
                         rot_max,
                         scale_min,
                         scale_max,
                         gamma_min,
                         gamma_max,
                         brightness_min,
                         brightness_max,
                         noise_min,
                         noise_max,
                         rot90 = False):
        
    images_ = np.copy(images)
    labels_ = np.copy(labels)

    
    #import utils
    #utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/NCI/NIFTI_NOTaugmented' + '_img_n4.nii.gz', data = images, affine = np.eye(4))
    #utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/NCI/NIFTI_NOTaugmented' + '_lbl.nii.gz', data = labels, affine = np.eye(4))
    
    for i in range(images.shape[2]):

        # ========
        # elastic deformation
        # ========
        if np.random.rand() < data_aug_ratio:
            
            images_[:,:,i], labels_[:,:,i] = elastic_transform_image_and_label(images_[:,:,i],
                                                                               labels_[:,:,i],
                                                                               sigma = sigma,
                                                                               alpha = alpha) 

        # ========
        # translation
        # ========
        if np.random.rand() < data_aug_ratio:
            
            random_shift_x = np.random.uniform(trans_min, trans_max)
            random_shift_y = np.random.uniform(trans_min, trans_max)
            
            images_[:,:,i] = scipy.ndimage.interpolation.shift(images_[:,:,i],
                                                               shift = (random_shift_x, random_shift_y),
                                                               order = 1)
            
            labels_[:,:,i] = scipy.ndimage.interpolation.shift(labels_[:,:,i],
                                                               shift = (random_shift_x, random_shift_y),
                                                               order = 0)
            
        # ========
        # rotation
        # ========
        if np.random.rand() < data_aug_ratio:
            
            random_angle = np.random.uniform(rot_min, rot_max)
            
            images_[:,:,i] = scipy.ndimage.interpolation.rotate(images_[:,:,i],
                                                                reshape = False,
                                                                angle = random_angle,
                                                                axes = (1, 0),
                                                                order = 1)
            
            labels_[:,:,i]= scipy.ndimage.interpolation.rotate(labels_[:,:,i],
                                                                reshape = False,
                                                                angle = random_angle,
                                                                axes = (1, 0),
                                                                order = 0)
            
        # ========
        # scaling
        # ========
        if np.random.rand() < data_aug_ratio:
            
            n_x, n_y = images_.shape[0], images_.shape[1]
            
            scale_val = np.round(np.random.uniform(scale_min, scale_max), 2)
            
            images_i_tmp = transform.rescale(images_[:,:,i], 
                                             scale_val,
                                             order = 1,
                                             preserve_range = True,
                                             mode = 'constant')
            
            # should we set anti_aliasing = False here?
            # otherwise, gaussian smoothing is applied before downscaling -> this makes the labels float instead of ints
            # anti_aliasing was set to false by default in the earlier version of skimage that we were using in the TTA DAE code...
            # now using a higher version of skimage (0.17.2), as reverting to 0.14.0 causes incompability with some other module on Euler...
            # not doing anti_aliasing=False while downsampling in evaluation led to substantial errors...
            labels_i_tmp = transform.rescale(labels_[:,:,i],
                                             scale_val,
                                             order = 0,
                                             preserve_range = True,
                                             anti_aliasing = False,
                                             mode = 'constant')
            
            images_[:,:,i] = crop_or_pad_slice_to_size(images_i_tmp, n_x, n_y)
            labels_[:,:,i] = crop_or_pad_slice_to_size(labels_i_tmp, n_x, n_y)

        # ========
        # rotate 90 / 180 / 270
        # Doing this for cardiac images (the data has this type of variability)
        # ========
        if rot90 == True:
            if np.random.rand() < data_aug_ratio:
                num_rotations = np.random.randint(1,4) # 1/2/3
                images_[:,:,i] = np.rot90(images_[:,:,i], k=num_rotations)
                labels_[:,:,i] = np.rot90(labels_[:,:,i], k=num_rotations)
            
        # ========
        # contrast
        # ========
        if np.random.rand() < data_aug_ratio:
            
            # gamma contrast augmentation
            c = np.round(np.random.uniform(gamma_min, gamma_max), 2)
            images_[:,:,i] = images_[:,:,i]**c
            # not normalizing after the augmentation transformation,
            # as it leads to quite strong reduction of the intensity range when done after high values of gamma augmentation

        # ========
        # brightness
        # ========
        if np.random.rand() < data_aug_ratio:
            
            # brightness augmentation
            c = np.round(np.random.uniform(brightness_min, brightness_max), 2)
            images_[:,:,i] = images_[:,:,i] + c
            
        # ========
        # noise
        # ========
        if np.random.rand() < data_aug_ratio:
            
            # noise augmentation
            n = np.random.normal(noise_min, noise_max, size = images_[:,:,i].shape)
            images_[:,:,i] = images_[:,:,i] + n

    
    #import utils
    #utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/NCI/NIFTI_augmented' + '_img_n4.nii.gz', data = images_, affine = np.eye(4))
    #utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/NCI/NIFTI_augmented' + '_lbl.nii.gz', data = labels_, affine = np.eye(4))

    return images_, labels_

# ==================================================================
#
# ==================================================================        
def do_data_augmentation_on_3d_labels(labels,
                                      data_aug_ratio,
                                      sigma,
                                      alpha,
                                      trans_min,
                                      trans_max,
                                      rot_min,
                                      rot_max,
                                      scale_min,
                                      scale_max,
                                      do_rot90 = False):
    
    labels_ = np.copy(labels[0,...])
        
    # ========
    # elastic deformation
    # ========
    if np.random.rand() < data_aug_ratio:
        
        labels_ = elastic_transform_label_3d(labels_,
                                             sigma = sigma,
                                             alpha = alpha)
        
    # ========
    # translation
    # ========
    if np.random.rand() < data_aug_ratio:
        
        random_shift_x = np.random.uniform(trans_min, trans_max)
        random_shift_y = np.random.uniform(trans_min, trans_max)
        
        for zz in range(labels_.shape[0]):
            labels_[zz,:,:] = scipy.ndimage.interpolation.shift(labels_[zz,:,:],
                                                                shift = (random_shift_x, random_shift_y),
                                                                order = 0)
        
    # ========
    # rotation
    # ========
    if np.random.rand() < data_aug_ratio:
        
        random_angle = np.random.uniform(rot_min, rot_max)
        
        for zz in range(labels_.shape[0]):
            labels_[zz,:,:] = scipy.ndimage.interpolation.rotate(labels_[zz,:,:],
                                                                 reshape = False,
                                                                 angle = random_angle,
                                                                 axes = (1, 0),
                                                                 order = 0)
            
    # ========
    # scaling
    # ========
    if np.random.rand() < data_aug_ratio:
        
        n_x, n_y = labels_.shape[1], labels_.shape[2]
        
        scale_val = np.round(np.random.uniform(scale_min, scale_max), 2)
        
        for zz in range(labels_.shape[0]):
            labels_i_tmp = transform.rescale(labels_[zz,:,:],
                                             scale_val,
                                             order = 0,
                                             preserve_range = True,
                                             mode = 'constant')
    
            labels_[zz,:,:] = crop_or_pad_slice_to_size(labels_i_tmp, n_x, n_y)

    # ========
    # 90 degree rotation
    # ========
    if do_rot90 == True:
        if np.random.rand() < data_aug_ratio:
            num_rotations = np.random.randint(1, 4) # 1 / 2 / 3
            for zz in range(labels_.shape[0]): # assumes same dimensionality in x and y directions
                labels_[zz,:,:] = np.rot90(labels_[zz,:,:], k=num_rotations)

    return np.expand_dims(labels_, axis=0)

# ===============================================================
# ===============================================================
def make_onehot(arr, nlabels):

    # taken from https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy/36960495
    ncols = nlabels
    out = np.zeros((arr.size, ncols), dtype=np.uint8)
    out[np.arange(arr.size), arr.ravel()] = 1
    out.shape = arr.shape + (ncols,)
    return out

# ================================================================== 
# Computes hausdorff distance between binary labels (compute separately for each label)
# ==================================================================    
def compute_surface_distance_per_label(y_1,
                                       y_2,
                                       sampling = 1,
                                       connectivity = 1):

    y1 = np.atleast_1d(y_1.astype(np.bool))
    y2 = np.atleast_1d(y_2.astype(np.bool))
    
    conn = morphology.generate_binary_structure(y1.ndim, connectivity)

    S1 = y1.astype(np.float32) - morphology.binary_erosion(y1, conn).astype(np.float32)
    S2 = y2.astype(np.float32) - morphology.binary_erosion(y2, conn).astype(np.float32)
    
    S1 = S1.astype(np.bool)
    S2 = S2.astype(np.bool)
    
    dta = morphology.distance_transform_edt(~S1, sampling)
    dtb = morphology.distance_transform_edt(~S2, sampling)
    
    sds = np.concatenate([np.ravel(dta[S2 != 0]), np.ravel(dtb[S1 != 0])])
    
    return sds

# ==================================================================   
# ==================================================================   
def compute_surface_distance(y1,
                             y2,
                             nlabels):
    
    mean_surface_distance_list = []
    hausdorff_distance_list = []
    
    for l in range(1, nlabels):

        surface_distance = compute_surface_distance_per_label(y_1 = (y1 == l),
                                                              y_2 = (y2 == l))
    
        mean_surface_distance = surface_distance.mean()
        # hausdorff_distance = surface_distance.max()
        hausdorff_distance = np.percentile(surface_distance, 95)

        mean_surface_distance_list.append(mean_surface_distance)
        hausdorff_distance_list.append(hausdorff_distance)
        
    return np.array(hausdorff_distance_list)

# ================================================================
# ================================================================
def print_results(fname, dataset):

    with open(fname, "r") as f:
        lines = f.readlines()

    pat_id = []
    dice = []

    for count in range(2, 22):
        line = lines[count]

        if dataset == 'PROMISE':
            pat_id.append(int(line[4:6]))
            dice.append(float(line[46:46+line[46:].find(',')]))
        elif dataset == 'USZ':
            pat_id.append(int(line[6:line.find(':')]))
            line = line[line.find(':') + 39 : ]
            dice.append(float(line[:line.find(',')]))

    pat_id = np.array(pat_id)
    dice = np.array(dice)
    results = np.stack((pat_id, dice))
    sorted_results = np.stack((np.sort(results[0,:]), results[1, np.argsort(results[0,:])]))

    # ==================================================================
    # sort
    # ==================================================================
    print('========== sorted results ==========')
    if dataset == 'PROMISE':
        for c in range(1, sorted_results.shape[1]):
            print(str(sorted_results[0,c]) + ',' + str(sorted_results[1,c]))
            if c == 9:
                print(str(sorted_results[0,0]) + ',' + str(sorted_results[1,0]))

    elif dataset == 'USZ':
        for c in range(0, sorted_results.shape[1]):
            print(str(sorted_results[0,c]) + ',' + str(sorted_results[1,c]))

    print('====================================')
    print(lines[31])
    print('====================================')

# ==================================================================
# ==================================================================
def make_noise_masks_3d(shape,
                        mask_type,
                        mask_params,
                        nlabels,
                        labels_1hot = None,
                        is_num_masks_fixed = False,
                        is_size_masks_fixed = False):
    
    blank_masks = np.ones(shape = shape)
    wrong_labels = np.zeros(shape = shape)
                   
    # ====================
    # make a random number of noise boxes in this (3d) image
    # ====================
    if is_num_masks_fixed is True:
        num_noise_squares = mask_params[1]
    else:
        num_noise_squares = np.random.randint(1, mask_params[1]+1)
        
    for _ in range(num_noise_squares):
            
        # ====================
        # choose the size of the noise box randomly 
        # ====================
        if is_size_masks_fixed is True:
            r = mask_params[0]
        else:
            r = np.random.randint(1, mask_params[0]+1)
            
        # ====================
        # Ensure that the box can fit in the volume is all dimensions
        # ====================
        r1 = np.minimum(r, shape[1]//2 - 2)
        r2 = np.minimum(r, shape[2]//2 - 2)
        r3 = np.minimum(r, shape[3]//2 - 2)

        # ====================
        # choose the center of the noise box randomly 
        # ====================
        mcx = np.random.randint(r1+1, shape[1]-(r1+1))
        mcy = np.random.randint(r2+1, shape[2]-(r2+1))
        mcz = np.random.randint(r3+1, shape[3]-(r3+1))
            
        # ====================
        # set the labels in this box to 0
        # ====================
        blank_masks[:, mcx-r1:mcx+r1, mcy-r2:mcy+r2, mcz-r3:mcz+r3, :] = 0
        
        # ====================
        # Replace the labels in the box, either with zeros or with the labels in a box of the same dimensions, somewhere else in the volume
        # ====================
        if mask_type is 'squares_jigsaw':               
            # choose another box in the image from which copy labels to the previous box
            mcx_src = np.random.randint(r1+1, shape[1]-(r1+1))
            mcy_src = np.random.randint(r2+1, shape[2]-(r2+1))
            mcz_src = np.random.randint(r3+1, shape[3]-(r3+1))
            wrong_labels[:, mcx-r1:mcx+r1, mcy-r2:mcy+r2, mcz-r3:mcz+r3, :] = labels_1hot[:, mcx_src-r1:mcx_src+r1, mcy_src-r2:mcy_src+r2, mcz_src-r3:mcz_src+r3, :]
            
        elif mask_type is 'squares_zeros':                
            # set the labels in this box to zero
            wrong_labels[:, mcx-r1:mcx+r1, mcy-r2:mcy+r2, mcz-r3:mcz+r3, 0] = 1
    
    return blank_masks, wrong_labels

# ===========================================================================
# ===========================================================================
def rescale_image_and_label(image,
                            label,
                            num_classes,
                            slice_thickness_this_subject,
                            new_resolution,
                            new_depth):
    
    image_rescaled = []
    label_rescaled = []
            
    # ======================
    # rescale in 3d
    # ======================
    scale_vector = [slice_thickness_this_subject / new_resolution, # for this axes, the resolution was kept unchanged during the initial 2D data preprocessing. but for the atlas (made from hcp labels), all of them have 0.7mm slice thickness
                    1.0, # the resolution along these 2 axes was made as required in the initial 2d data processing already
                    1.0]
    
    image_rescaled = transform.rescale(image,
                                       scale_vector,
                                       order=1,
                                       preserve_range=True,
                                       multichannel=False,
                                       mode = 'constant')

    # RESCALING TYPE 1
    # label_onehot = make_onehot(label, num_classes)
    # label_onehot_rescaled = transform.rescale(label_onehot,
    #                                           scale_vector,
    #                                           order=1,
    #                                           preserve_range=True,
    #                                           multichannel=True,
    #                                           mode='constant')
    # label_rescaled = np.argmax(label_onehot_rescaled, axis=-1)

    # RESCALING TYPE 2
    label_rescaled = transform.rescale(label,
                                       scale_vector,
                                       order=0,
                                       preserve_range=True,
                                       multichannel=False,
                                       mode='constant',
                                       anti_aliasing = False)
        
    # =================
    # crop / pad
    # =================
    image_rescaled_cropped = crop_or_pad_volume_to_size_along_x(image_rescaled, new_depth).astype(np.float32)
    label_rescaled_cropped = crop_or_pad_volume_to_size_along_x(label_rescaled, new_depth).astype(np.uint8)
            
    return image_rescaled_cropped, label_rescaled_cropped


def do_data_augmentation_FETS(images,
                         labels,
                         data_aug_ratio,
                         sigma,
                         alpha,
                         trans_min,
                         trans_max,
                         rot_min,
                         rot_max,
                         scale_min,
                         scale_max,
                         gamma_min,
                         gamma_max,
                         brightness_min,
                         brightness_max,
                         noise_min,
                         noise_max,
                         rot90 = False):
        
    images_ = np.copy(images)
    labels_ = np.copy(labels)

    
    #import utils
    #utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/NCI/NIFTI_NOTaugmented' + '_img_n4.nii.gz', data = images, affine = np.eye(4))
    #utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/NCI/NIFTI_NOTaugmented' + '_lbl.nii.gz', data = labels, affine = np.eye(4))
    
    for i in range(images.shape[3]):

        # ========
        # elastic deformation
        # ========
        if np.random.rand() < data_aug_ratio:

            augm_list = []
            lab_list = []

            for j in range(images.shape[2]):
                images_[:,:,j,i], labels_[:,:,j,i] = elastic_transform_image_and_label(images_[:,:,j,i],
                                                                                labels_[:,:,j,i],
                                                                                sigma = sigma,
                                                                                alpha = alpha)

                augm_list.append(images[:, :, j, i])
                lab_list.append(labels[:, :, j, i])
            
            images_[..., i] = torch.stack(augm_list, dim = -1)
            labels_[..., i] = torch.stack(lab_list, dim = -1)

        # ========
        # translation
        # ========
        if np.random.rand() < data_aug_ratio:
            
            random_shift_x = np.random.uniform(trans_min, trans_max)
            random_shift_y = np.random.uniform(trans_min, trans_max)

            augm_list = []
            lab_list = []

            for j in range(images.shape[2]): 
                images_[:,:,j,i] = scipy.ndimage.interpolation.shift(images_[:,:,j,i],
                                                                shift = (random_shift_x, random_shift_y),
                                                                order = 1)

                labels_[:,:,j,i] = scipy.ndimage.interpolation.shift(labels_[:,:,j,i],
                                                               shift = (random_shift_x, random_shift_y),
                                                               order = 0)                                               
            
                augm_list.append(images[:, :, j, i])
                lab_list.append(labels[:, :, j, i])

            
            images_[..., i] = torch.stack(augm_list, dim = -1)
            labels_[..., i] = torch.stack(lab_list, dim = -1)

            
            
        # ========
        # rotation
        # ========
        if np.random.rand() < data_aug_ratio:
            
            random_angle = np.random.uniform(rot_min, rot_max)
            augm_list = []
            lab_list = []
            
            for j in range(images.shape[2]):
                images_[:,:,j, i] = scipy.ndimage.interpolation.rotate(images_[:,:,j,i],
                                                                    reshape = False,
                                                                    angle = random_angle,
                                                                    axes = (1, 0),
                                                                    order = 1)

                
                labels_[:,:,j, i]= scipy.ndimage.interpolation.rotate(labels_[:,:,j, i],
                                                                reshape = False,
                                                                angle = random_angle,
                                                                axes = (1, 0),
                                                                order = 0)


                augm_list.append(images[:, :, j, i])
                lab_list.append(labels[:, :, j, i])

            images_[..., i] = torch.stack(augm_list, dim = -1)
            labels_[..., i] = torch.stack(lab_list, dim = -1)
            
            
            
        # ========
        # scaling
        # ========
        if np.random.rand() < data_aug_ratio:
            
            n_x, n_y = images_.shape[0], images_.shape[1]
            augm_list = []
            lab_list = []
            images_i_tmp = []
            images_i_tmp = torch.Tensor(images_i_tmp)
            
            scale_val = np.round(np.random.uniform(scale_min, scale_max), 2)

            for j in range(images.shape[2]):
                images_i_tmp = transform.rescale(images_[:,:,j,i], 
                                                scale_val,
                                                order = 1,
                                                preserve_range = True,
                                                mode = 'constant')
                
                images_[:,:,j, i] = crop_or_pad_slice_to_size(images_i_tmp, n_x, n_y)

                labels_i_tmp = transform.rescale(labels_[:,:,j,i],
                                             scale_val,
                                             order = 0,
                                             preserve_range = True,
                                             anti_aliasing = False,
                                             mode = 'constant')
            

            
                labels_[:,:,j,i] = crop_or_pad_slice_to_size(labels_i_tmp, n_x, n_y)

                augm_list.append(images[:, :, j, i])
                lab_list.append(labels[:, :, j, i])

                #augm_list.append(images_i_tmp)
            
            #augm_list = torch.Tensor(augm_list)
            #images_i_tmp = torch.stack(augm_list, dim = -1)
            
            images_[..., i] = torch.stack(augm_list, dim = -1)
            labels_[..., i] = torch.stack(lab_list, dim = -1)

            # should we set anti_aliasing = False here?
            # otherwise, gaussian smoothing is applied before downscaling -> this makes the labels float instead of ints
            # anti_aliasing was set to false by default in the earlier version of skimage that we were using in the TTA DAE code...
            # now using a higher version of skimage (0.17.2), as reverting to 0.14.0 causes incompability with some other module on Euler...
            # not doing anti_aliasing=False while downsampling in evaluation led to substantial errors...
            

        # ========
        # rotate 90 / 180 / 270
        # Doing this for cardiac images (the data has this type of variability)
        # ========
        if rot90 == True:
            if np.random.rand() < data_aug_ratio:
                num_rotations = np.random.randint(1,4) # 1/2/3
                augm_list = []
                lab_list = []
                for j in range(images.shape[2]):
                    images_[:,:,j,i] = np.rot90(images_[:,:,j,i], k=num_rotations)
                    labels_[:,:,i] = np.rot90(labels_[:,:,i], k=num_rotations)
                    augm_list.append(images[:, :, j, i])
                    lab_list.append(labels[:, :, j, i])
                    

                images_[..., i] = torch.stack(augm_list, dim = -1)
                labels_[..., i] = torch.stack(lab_list, dim = -1)

                
            
        # ========
        # contrast
        # ========
        if np.random.rand() < data_aug_ratio:
            
            # gamma contrast augmentation
            c = np.round(np.random.uniform(gamma_min, gamma_max), 2)
            augm_list = []
            for j in range(images.shape[2]):
                images_[:,:,j,i] = images_[:,:,j,i]**c
                augm_list.append(images[:, :, j, i])

            images_[..., i] = torch.stack(augm_list, dim = -1)
            # not normalizing after the augmentation transformation,
            # as it leads to quite strong reduction of the intensity range when done after high values of gamma augmentation

        # ========
        # brightness
        # ========
        if np.random.rand() < data_aug_ratio:
            
            # brightness augmentation
            c = np.round(np.random.uniform(brightness_min, brightness_max), 2)
            augm_list = []
            for j in range(images.shape[2]):
                images_[:,:,j,i] = images_[:,:,j,i] + c
                augm_list.append(images[:, :, j, i])

            images_[..., i] = torch.stack(augm_list, dim = -1)
            
        # ========
        # noise
        # ========
        if np.random.rand() < data_aug_ratio:
            
            # noise augmentation

            
            n = np.random.normal(noise_min, noise_max, size = images_[:,:,0,i].shape)
            augm_list = []
            for j in range(images.shape[2]):
            
                images_[:,:,j,i] = images_[:,:,j,i] + n
                augm_list.append(images[:, :, j, i])

            images_[..., i] = torch.stack(augm_list, dim = -1)

    
    #import utils
    #utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/NCI/NIFTI_augmented' + '_img_n4.nii.gz', data = images_, affine = np.eye(4))
    #utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/NCI/NIFTI_augmented' + '_lbl.nii.gz', data = labels_, affine = np.eye(4))

    return images_, labels_


def chunks(lst, n):
    """ Yield successive n-sized chunks from lst """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]



def test_single_volume_FETS(image, label, net, i2n_module_t1, i2n_module_t1ce, i2n_module_t2, i2n_module_flair, use_tta, tta_epochs, writer, classes, dataset, optim, model_type, seed, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.cpu().detach().numpy(), label.cpu().detach().numpy()   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    i2n_module_t1.to(device)
    i2n_module_t1ce.to(device)
    i2n_module_t2.to(device)
    i2n_module_flair.to(device)
    net.to(device)

    base_lr = 1e-3
    prediction = np.zeros_like(label[:, :, :, 0])
    norm_out_t1 = torch.zeros(155, 240, 240)
    norm_out_t1ce = torch.zeros(155, 240, 240)
    norm_out_t2 = torch.zeros(155, 240, 240)
    norm_out_flair = torch.zeros(155, 240, 240)

    
    label_copy = copy.deepcopy(label)
    optim_unet = optimizer.Adam(net.parameters(), lr=base_lr)
    optim_i2n_t1 = optimizer.Adam(i2n_module_t1.parameters(), lr=base_lr)
    optim_i2n_t1ce = optimizer.Adam(i2n_module_t1ce.parameters(), lr=base_lr)
    optim_i2n_t2 = optimizer.Adam(i2n_module_t2.parameters(), lr=base_lr)
    optim_i2n_flair = optimizer.Adam(i2n_module_flair.parameters(), lr=base_lr)

    foreground_list = []
    label_list = []
    entropy_loss = HLoss()  #Entropy Loss
    
    iter_num = 0
    batch_size = 5
    max_iterations = image.shape[0]/batch_size * tta_epochs #31 iterations per volume x 10 epochs
    

    # ============================
    # Perform the prediction slice by slice
    # ============================ 

    if use_tta == True:
        for epoch in range(tta_epochs):
            norm_out_t1 = torch.zeros(155, 240, 240)
            norm_out_t1ce = torch.zeros(155, 240, 240)
            norm_out_t2 = torch.zeros(155, 240, 240)
            norm_out_flair = torch.zeros(155, 240, 240)
            loss_ii = 0.0
            num_batches = 0
            shuffled_slices = np.random.permutation(image.shape[0])
            for slices in chunks(shuffled_slices, batch_size):
                slice = image[slices, :, :]
                input = torch.from_numpy(slice).float().cuda()
                input = input.permute(0, 3, 1, 2)
                i2n_module_t1.train()
                i2n_module_t1ce.train()
                i2n_module_t2.train()
                i2n_module_flair.train()
                net.eval()
                for param in i2n_module_t1.parameters():
                    param.requires_grad = True

                for param in i2n_module_t1ce.parameters():
                    param.requires_grad = True

                for param in i2n_module_t2.parameters():
                    param.requires_grad = True

                for param in i2n_module_flair.parameters():
                    param.requires_grad = True    

                for param in net.parameters():
                    param.requires_grad = False


                norm_output_t1 = i2n_module_t1(input[:, 0, :, :].unsqueeze(1))
                norm_output_t1ce = i2n_module_t1ce(input[:, 1, :, :].unsqueeze(1))
                norm_output_t2 = i2n_module_t2(input[:, 2, :, :].unsqueeze(1))
                norm_output_flair = i2n_module_flair(input[:, 3, :, :].unsqueeze(1))

                norm_output = torch.cat((norm_output_t1, norm_output_t1ce, norm_output_t2, norm_output_flair), 1)

                outputs = net(norm_output)
                
                with torch.no_grad():
                    i=0
                    for s_norm in slices:
                        norm_out_t1[s_norm, :, :] = norm_output[i, 0, :, :]
                        norm_out_t1ce[s_norm, :, :] = norm_output[i, 1, :, :]
                        norm_out_t2[s_norm, :, :] = norm_output[i, 2, :, :]
                        norm_out_flair[s_norm, :, :] = norm_output[i, 3, :, :]
                        i =i+1

                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                i=0
                for s in slices:
                    prediction[s] = out[i]
                    i =i+1
                #out_soft = torch.softmax(outputs, dim=1)
                loss = entropy_loss(outputs)
                loss.backward()
                loss_ii += loss.item()
                
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optim_unet.param_groups:
                    param_group['lr'] = lr_
                for param_group in optim_i2n_t1.param_groups:
                    param_group['lr'] = lr_
                for param_group in optim_i2n_t1ce.param_groups:
                    param_group['lr'] = lr_   
                for param_group in optim_i2n_t2.param_groups:
                    param_group['lr'] = lr_
                for param_group in optim_i2n_flair.param_groups:
                    param_group['lr'] = lr_

                iter_num += 1
                num_batches += 1
            
            epoch_loss = loss_ii/(batch_size*240*240)
            #log loss_ii 
            writer.add_scalar('info/tta_entropy_loss', epoch_loss, iter_num)
            #log dice score
            metric_whole_tumor = []
            metric_enhancing_tumor = []
            metric_tumor_core = []
            metric_whole_tumor.append(calculate_metric_percase(prediction > 0, label_copy[:, :, :, 0] > 0))

            prediction_whole = copy.deepcopy(prediction)
            prediction_whole = prediction_whole.swapaxes(1, 0)
            prediction_whole = prediction_whole.swapaxes(2, 1)
            prediction_whole = np.rot90(prediction_whole, 3)

            label_tensorboard = copy.deepcopy(label_copy)
            label_tensorboard = label_tensorboard.swapaxes(0, 1)
            label_tensorboard = label_tensorboard.swapaxes(1, 2)
            label_tensorboard = np.rot90(label_tensorboard, 3)
            

            utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + f'TTA_prediction_whole_iter{iter_num}.nii.gz', data = prediction_whole[:, :, :], affine = np.eye(4))
            utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + f'TTA_label_iter{iter_num}.nii.gz', data = label_tensorboard[:, :, :, 0], affine = np.eye(4))
            
            writer.add_image(f'prediction_whole_tumor{dataset}/Image', prediction_whole[:, :, 99], iter_num)

            writer.add_image(f'label_whole_tumor{dataset}/Image', label_tensorboard[:, :, 99, 0], iter_num)
            

            prediction = np.array(prediction)
            label_copy = np.array(label_copy)
            prediction[np.where(prediction == 2)] = 0
            label_copy[np.where(label_copy == 2)] = 0
            metric_tumor_core.append(calculate_metric_percase(prediction > 0, label_copy[:, :, :, 0] > 0))

            prediction_core = copy.deepcopy(prediction)
            prediction_core = prediction_core.swapaxes(1, 0)
            prediction_core = prediction_core.swapaxes(2, 1)
            prediction_core = np.rot90(prediction_core, 3)

            writer.add_image(f'prediction_tumor_core{dataset}/Image', prediction_core[:, :, 99], iter_num)
            utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + f'TTA_prediction_core_iter{iter_num}.nii.gz', data = prediction_core[:, :, :], affine = np.eye(4))

            writer.add_image(f'label_tumor_core{dataset}/Image', label_tensorboard[:, :, 99, 0], iter_num)
    

            prediction = np.array(prediction)
            label_copy = np.array(label_copy)
            prediction[np.where(prediction < 3)] = 0
            label_copy[np.where(label_copy < 3)] = 0
            metric_enhancing_tumor.append(calculate_metric_percase(prediction > 0, label_copy[:, :, :, 0] > 0))

            prediction_enhancing = copy.deepcopy(prediction)
            prediction_enhancing = prediction_enhancing.swapaxes(1, 0)
            prediction_enhancing = prediction_enhancing.swapaxes(2, 1)
            prediction_enhancing = np.rot90(prediction_enhancing, 3)

            writer.add_image(f'prediction_enhancing_tumor_t1_{dataset}/Image', prediction_enhancing[:, :, 99], iter_num)
            utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + f'TTA_prediction_enhancing_iter{iter_num}.nii.gz', data = prediction_enhancing[:, :, :], affine = np.eye(4))

            writer.add_image(f'label_enhancing_tumor_t1_{dataset}/Image', label_tensorboard[:, :, 99, 0], iter_num)

            metric_whole_tumor = np.array(metric_whole_tumor)
            metric_enhancing_tumor = np.array(metric_enhancing_tumor)
            metric_tumor_core = np.array(metric_tumor_core)

            writer.add_scalar(f'info/dice_whole_tumor_{dataset}', np.mean(metric_whole_tumor, axis = 0)[0] , iter_num)
            writer.add_scalar(f'info/dice_enhancing_tumor_{dataset}', np.mean(metric_enhancing_tumor, axis = 0)[0] , iter_num)
            writer.add_scalar(f'info/dice_tumor_core_{dataset}', np.mean(metric_tumor_core, axis = 0)[0] , iter_num)


            norm_out_t1 = norm_out_t1.cpu().detach().numpy() 
            norm_out_t1ce = norm_out_t1ce.cpu().detach().numpy() 
            norm_out_t2 = norm_out_t2.cpu().detach().numpy() 
            norm_out_flair = norm_out_flair.cpu().detach().numpy() 

            norm_out_t1 = norm_out_t1.swapaxes(1, 0)
            norm_out_t1 = norm_out_t1.swapaxes(2, 1)
            norm_out_t1 = np.rot90(norm_out_t1, 3)
            
            norm_out_t1ce = norm_out_t1ce.swapaxes(1, 0)
            norm_out_t1ce = norm_out_t1ce.swapaxes(2, 1)
            norm_out_t1ce = np.rot90(norm_out_t1ce, 3)

            norm_out_t2 = norm_out_t2.swapaxes(1, 0)
            norm_out_t2 = norm_out_t2.swapaxes(2, 1)
            norm_out_t2 = np.rot90(norm_out_t2, 3)

            norm_out_flair = norm_out_flair.swapaxes(1, 0)
            norm_out_flair = norm_out_flair.swapaxes(2, 1)
            norm_out_flair = np.rot90(norm_out_flair, 3)


            utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + f'TTA_norm_out_t1_iter{iter_num}.nii.gz', data = norm_out_t1[:, :, :], affine = np.eye(4))
            utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + f'TTA_norm_out_t1ce_iter{iter_num}.nii.gz', data = norm_out_t1ce[:, :, :], affine = np.eye(4))
            utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + f'TTA_norm_out_t2_iter{iter_num}.nii.gz', data = norm_out_t2[:, :, :], affine = np.eye(4))
            utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + f'TTA_norm_out_flair_iter{iter_num}.nii.gz', data = norm_out_flair[:, :, :], affine = np.eye(4))

            

            writer.add_image(f'norm_output_t1_{dataset}/Image', norm_out_t1[:, :, 99], iter_num)
            writer.add_image(f'norm_output_t1ce_{dataset}/Image', norm_out_t1ce[:, :, 99], iter_num)
            writer.add_image(f'norm_output_t2_{dataset}/Image', norm_out_t2[:, :, 99], iter_num)
            writer.add_image(f'norm_output_flair_{dataset}/Image', norm_out_flair[:, :, 99], iter_num)

            #breakpoint()

            optim_i2n_t1.step()
            optim_i2n_t1ce.step()
            optim_i2n_t2.step()
            optim_i2n_flair.step()

            optim_i2n_t1.zero_grad()
            optim_i2n_t1ce.zero_grad()
            optim_i2n_t2.zero_grad()
            optim_i2n_flair.zero_grad()


            logging.info(f"TTA Entropy loss: {epoch_loss}")

        logging.info(f"TTA loop done for subject {case}")


    # ============================
    # Inference with updated parameters
    # ============================  
    prediction = np.zeros_like(label[:, :, :, 0])

    for ind in range(image.shape[0]):  
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        if x != patch_size[0] or y != patch_size[1]:
            slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
        
        #utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + '4_test.nii.gz', data = slice, affine = np.eye(4))
        #utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + '4_label.nii.gz', data = label[:, :, ind], affine = np.eye(4))

        input = torch.from_numpy(slice).unsqueeze(0).float().cuda()
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
        
        with torch.no_grad():

            norm_output_t1 = i2n_module_t1(input[:, 0, :, :].unsqueeze(1))
            norm_output_t1ce = i2n_module_t1ce(input[:, 1, :, :].unsqueeze(1))
            norm_output_t2 = i2n_module_t2(input[:, 2, :, :].unsqueeze(1))
            norm_output_flair = i2n_module_flair(input[:, 3, :, :].unsqueeze(1))

            norm_output = torch.cat((norm_output_t1, norm_output_t1ce, norm_output_t2, norm_output_flair), 1)

            outputs = net(norm_output)
            out_soft = torch.softmax(outputs, dim=1)
            out_hard = (out_soft>0.5).float()
            out_hard_argmax = torch.argmax(out_hard, dim=1).squeeze(0) 
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            #color_map = torch.tensor([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
            rgb = np.zeros((256, 256, 3))
            #io.imshow(color.label2rgb(out, slice))

            out_soft_sq = out_soft.squeeze(0)
            out_soft_foreground = out_soft_sq[1, :, : ] + out_soft_sq[2, :, : ]
            out_soft_foreground = out_soft_foreground.flatten()
            out_soft_foreground = out_soft_foreground.cpu().detach().numpy()
            foreground_list.append(out_soft_foreground)
            

            label_temp = label[ind, :, :, 0]
            label_temp = label_temp.flatten()
            label_temp[np.where(label_temp > 0)] = 1
            label_list.append(label_temp)
                                                                                

            out = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            else:
                pred = out
            prediction[ind] = pred
    
    label_list_arr = np.array(label_list)
    label_list_arr = label_list_arr.flatten()

    foreground_list_arr = np.array(foreground_list)
    foreground_list_arr = foreground_list_arr.flatten()

        

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


    
        
        
         

    # ============================
    # Save images, predictions and ground truths
    # ============================
    
    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+"{}".format(case) + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+"{}".format(case) + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+"{}".format(case) + "_gt.nii.gz")
    return metric_list_whole_tumor, metric_list_enhancing_tumor, metric_list_tumor_core, foreground_list_arr, label_list_arr


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b



def update_params(param, grad, loss, learning_rate):
  return param - learning_rate * grad