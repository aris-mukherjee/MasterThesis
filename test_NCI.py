import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import config.system_paths as sys_config
import utils_data
from networks.unet_class import UNET
import utils
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from calibration_functions import find_bin_values
from calibration_functions import find_area


seed = 1234
model_type = 'UNET'
measure_calibration = False


parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='/itet-stor/arismu/bmicdatasets-originals/Originals/Challenge_Datasets/NCI_Prostate/', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--test_dataset', type=str,
                    default='RUNMC', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--max_iterations', type=int,default=6800, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=400, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
parser.add_argument('--is_savenii', type=bool, default=True, help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=1e-3, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--target_resolution', type=float, default=0.625, help='target resolution')   
parser.add_argument('--image_depth_tr', type=int, default=32, help='target resolution') 
parser.add_argument('--image_depth_ts', type=int, default=32, help='target resolution')    
parser.add_argument('--test_cv_fold_num', type = int, default = 1) # 1 / 2 / 3 / 4
parser.add_argument('--NORMALIZE', type = int, default = 1) # 1 / 0
args = parser.parse_args()



def inference(args, model, test_save_path=None):

    # ============================
    # Load test data
    # ============================ 
    
    loaded_test_data = utils_data.load_testing_data(args.test_dataset,  
                                                    args.test_cv_fold_num,
                                                    args.img_size,
                                                    args.target_resolution,
                                                    args.image_depth_ts)


    imts = loaded_test_data[0]  
    gtts = loaded_test_data[1]
    orig_data_res_x = loaded_test_data[2]
    orig_data_res_y = loaded_test_data[3]
    orig_data_res_z = loaded_test_data[4]
    orig_data_siz_x = loaded_test_data[5]
    orig_data_siz_y = loaded_test_data[6]
    orig_data_siz_z = loaded_test_data[7]
    name_test_subjects = loaded_test_data[8]
    num_test_subjects = loaded_test_data[9]
    ids = loaded_test_data[10]

    

    model.eval()
    metric_list = 0.0
    pred_list = []
    label_list = []  

    
    for sub_num in range(num_test_subjects):


        # ============================
        # Group slices belonging to the same patients
        # ============================ 

        subject_id_start_slice = np.sum(orig_data_siz_z[:sub_num])   
        subject_id_end_slice = np.sum(orig_data_siz_z[:sub_num+1])   
        image = imts[:,:, subject_id_start_slice:subject_id_end_slice] 
        label = gtts[:,:, subject_id_start_slice:subject_id_end_slice] 


        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        image, label = image.cuda(), label.cuda()      
        image = image.permute(2, 0, 1)
        label = label.permute(2, 0, 1)

       
        # ==================================================================
        # setup logging
        # ==================================================================
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
        subject_name = str(name_test_subjects[sub_num])[2:-1]
        logging.info('============================================================')
        logging.info('Subject ' + str(sub_num+1) + ' out of ' + str(num_test_subjects) + ': ' + subject_name)

        # ============================
        # Perform the prediction for each test patient individually & calculate Dice score and Hausdorff distance
        # ============================ 

        metric_i, pred_l, label_l = test_single_volume(image, label, model, classes=args.num_classes, dataset = 'NCI', optim = 'ADAM', model_type = f'{model_type}', seed = f'{seed}', patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=sub_num, z_spacing=args.z_spacing)

        metric_list += np.array(metric_i)
        pred_list.extend(pred_l)
        label_list.extend(label_l)

        logging.info('case %s mean_dice %f mean_hd95 %f' % (sub_num, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / num_test_subjects   #get mean metrics for every class

        
    
    if measure_calibration:
        first_bin_frac_pos, second_bin_frac_pos, third_bin_frac_pos, fourth_bin_frac_pos, fifth_bin_frac_pos = find_bin_values(pred_list, label_list)
        find_area(first_bin_frac_pos, second_bin_frac_pos, third_bin_frac_pos, fourth_bin_frac_pos, fifth_bin_frac_pos)
        disp = CalibrationDisplay.from_predictions(label_list, pred_list)
        plt.show()
        plt.savefig(f'/scratch_net/biwidl217_second/arismu/Data_MT/plots/{model_type}_NCI.png')

   
    # ============================
    # Log the mean performance achieved for each class
    # ============================ 

    logging.info('AVERAGE OVER ALL TEST SUBJECTS: mean_dice %f mean_hd95 %f' % (metric_list[0][0], metric_list[0][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    return "Testing Finished!"

    


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'RUNMC': {
            'volume_path': '/itet-stor/arismu/bmicdatasets-originals/Originals/Challenge_Datasets/NCI_Prostate/',
            'num_classes': 3,
            'z_spacing': 1,
        },
    }
    dataset_name = args.test_dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_name
    args.z_spacing = dataset_config[dataset_name]['z_spacing']

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))

    if model_type == 'UNWT':
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    elif model_type == 'UNET':   
        net = UNET(in_channels = 3, out_channels = 3, features = [32, 64, 128, 256]).cuda()
    else:
        print("Model type error")

   
    # ============================
    # Load the trained parameters into the model
    # ============================  
    snapshot = os.path.join(f'/scratch_net/biwidl217_second/arismu/Master_Thesis_Codes/project_TransUNet/model/2022/{model_type}/', f'{model_type}_best_val_loss_seed{seed}.pth')
    net.load_state_dict(torch.load(snapshot))

    # ============================
    # Save the predictions as nii files
    # ============================ 

    if args.is_savenii:
        args.test_save_dir = f'../predictions_2022/{model_type}/'
        test_save_path = os.path.join(args.test_save_dir, f'NCI_{model_type}_test_seed{seed}')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)

print("test.py successfully executed")

