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
from gauss_params_train_eval import test_single_volume_FETS_train_eval
from utils import test_single_volume, test_single_volume_FETS
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import config.system_paths as sys_config
import utils_data
from networks.unet_class import UNET
import utils
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt
from calibration_functions import find_bin_values, find_bin_values_FETS
from calibration_functions import find_area
from normalisation_module import Normalisation_Module_flair, Normalisation_Module_t1, Normalisation_Module_t1ce, Normalisation_Module_t2
from tensorboardX import SummaryWriter
import SimpleITK as sitk


seed = 1234
model_type = 'UNET'
data_aug = '0.25'
use_tta = True
tta_epochs = 10

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='/itet-stor/arismu/bmicdatasets-originals/Originals/Challenge_Datasets/FeTS/', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--test_dataset', type=str,
                    default='FETS_train', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--max_iterations', type=int,default=6800, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=400, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=240, help='input patch size of network input')
#parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
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
    
    writer = SummaryWriter(f'/scratch_net/biwidl217_second/arismu/Tensorboard/2022/FETS/{model_type}/TTA/TEST/' + f'TRAIN_EVAL_FETS_{model_type}_log_seed{seed}_da{data_aug}')


    # ============================
    # Load training data
    # ============================   
    logging.info('============================================================')
    logging.info('Loading data...')
    loaded_tr_data_part1 = utils_data.load_training_data('FETS_train_part1',
                                                   args.img_size,
                                                   args.target_resolution,
                                                   args.test_cv_fold_num)
    imtr_part1 = loaded_tr_data_part1[0]
    gttr_part1 = loaded_tr_data_part1[1]
    num_test_subjects_part1 = loaded_tr_data_part1[2]

    loaded_tr_data_part2 = utils_data.load_training_data('FETS_train_part2',
                                                   args.img_size,
                                                   args.target_resolution,
                                                   args.test_cv_fold_num)
    imtr_part2 = loaded_tr_data_part2[0]
    gttr_part2 = loaded_tr_data_part2[1]
    num_test_subjects_part2 = loaded_tr_data_part2[2] 

    loaded_tr_data_part3 = utils_data.load_training_data('FETS_train_part3',
                                                   args.img_size,
                                                   args.target_resolution,
                                                   args.test_cv_fold_num)
    imtr_part3 = loaded_tr_data_part3[0]
    gttr_part3 = loaded_tr_data_part3[1]
    num_test_subjects_part3 = loaded_tr_data_part3[2] 


    num_test_subjects = num_test_subjects_part1 + num_test_subjects_part2 + num_test_subjects_part3

    
    imtr_part2 = np.array(imtr_part2)

    utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + 'PART3_SUBJECT_30.nii.gz', data = imtr_part3[:, :, 17360:17980], affine = np.eye(4))
    utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + 'PART3_LABEL_30.nii.gz', data = gttr_part3[:, :, 17360:17980], affine = np.eye(4))

    utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + 'PART2_SUBJECT_30.nii.gz', data = imtr_part2[:, :, 17360:17980], affine = np.eye(4))
    utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + 'PART2_LABEL_30.nii.gz', data = gttr_part2[:, :, 17360:17980], affine = np.eye(4))



    
    imtr = np.concatenate((imtr_part1, imtr_part2), axis = 2) 
    imtr = np.concatenate((imtr, imtr_part3), axis = 2)

   
    gttr = np.concatenate((gttr_part1, gttr_part2), axis = 2) 
    gttr = np.concatenate((gttr, gttr_part3), axis = 2)

    
    
    gttr[np.where(gttr == 4)] = 3  #turn labels [0 1 2 4] into [0 1 2 3]

    #utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + 'SUBJECT_29.nii.gz', data = imtr[:, :, 17360:17980], affine = np.eye(4))
    #utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + 'LABEL_29.nii.gz', data = gttr[:, :, 17360:17980], affine = np.eye(4))

    #utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + 'SUBJECT_30.nii.gz', data = imtr[:, :, 17980:18600], affine = np.eye(4))
    #utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + 'LABEL_30.nii.gz', data = gttr[:, :, 17980:18600], affine = np.eye(4))



    imtr = torch.from_numpy(imtr)
    gttr = torch.from_numpy(gttr)



    
    img_list = []
    label_list = []


    lim1 = 0
    lim2 = 155 
    lim3 = 310
    lim4 = 465
    x = []


    for i in range(imtr.shape[2]):
        if lim1 != 0 and (lim1 % 155) == 0:
            lim1 = lim4
            lim2 = lim1 + 155
            lim3 = lim1 + 310
            lim4 = lim1 + 465
        x.append(imtr[:, :, lim1])
        x.append(imtr[:, :, lim2])
        x.append(imtr[:, :, lim3])
        x.append(imtr[:, :, lim4])
        y = torch.stack(x, dim = -1)
        img_list.append(y)


        lim1 += 1
        lim2 += 1
        lim3 += 1
        lim4 += 1
        x = []
        
        if lim4 == (imtr.shape[2]):
            break
    

    lim1 = 0
    lim2 = 155 
    lim3 = 310
    lim4 = 465
    x = []


    for i in range(gttr.shape[2]):
        if lim1 != 0 and (lim1 % 155) == 0:
            lim1 = lim4
            lim2 = lim1 + 155
            lim3 = lim1 + 310
            lim4 = lim1 + 465
        x.append(gttr[:, :, lim1])
        x.append(gttr[:, :, lim2])
        x.append(gttr[:, :, lim3])
        x.append(gttr[:, :, lim4])
        y = torch.stack(x, dim = -1)
        label_list.append(y)


        lim1 += 1
        lim2 += 1
        lim3 += 1
        lim4 += 1
        x = []
        
        if lim4 == (gttr.shape[2]):
            break



    lim1 = 0
    lim2 = 155 
    lim3 = 310
    lim4 = 465
    x = []

    sub_subject_id_start_slice = 0
    subject_id_end_slice = 155

    metric_list_whole = 0.0
    metric_list_enhancing = 0.0
    metric_list_core = 0.0
    

    logging.info('Training Images: %s' %str(imtr.shape)) # expected: [num_slices, img_size_x, img_size_y]
    logging.info('Training Labels: %s' %str(gttr.shape)) # expected: [num_slices, img_size_x, img_size_y]
   

    logging.info('============================================================')


    for sub_num in range(num_test_subjects):


        i2n_module_t1 = Normalisation_Module_t1(in_channels = 1)
        i2n_module_t1ce = Normalisation_Module_t1ce(in_channels = 1)
        i2n_module_t2 = Normalisation_Module_t2(in_channels = 1)
        i2n_module_flair = Normalisation_Module_flair(in_channels = 1)

        save_t1_path = os.path.join(f'/scratch_net/biwidl217_second/arismu/Master_Thesis_Codes/project_TransUNet/model/2022/FETS/{model_type}/TTA/', f'FETS_{model_type}_{seed}_da{data_aug}_TTA_NORM_T1' + '.pth')
        save_t1ce_path = os.path.join(f'/scratch_net/biwidl217_second/arismu/Master_Thesis_Codes/project_TransUNet/model/2022/FETS/{model_type}/TTA/', f'FETS_{model_type}_{seed}_da{data_aug}_TTA_NORM_T1CE' + '.pth')
        save_t2_path = os.path.join(f'/scratch_net/biwidl217_second/arismu/Master_Thesis_Codes/project_TransUNet/model/2022/FETS/{model_type}/TTA/', f'FETS_{model_type}_{seed}_da{data_aug}_TTA_NORM_T2' + '.pth')
        save_flair_path = os.path.join(f'/scratch_net/biwidl217_second/arismu/Master_Thesis_Codes/project_TransUNet/model/2022/FETS/{model_type}/TTA/', f'FETS_{model_type}_{seed}_da{data_aug}_TTA_NORM_FLAIR' + '.pth')

        i2n_module_t1.load_state_dict(torch.load(save_t1_path))
        i2n_module_t1ce.load_state_dict(torch.load(save_t1ce_path))
        i2n_module_t2.load_state_dict(torch.load(save_t2_path)) 
        i2n_module_flair.load_state_dict(torch.load(save_flair_path))


        # ============================
        # Group slices belonging to the same patients
        # ============================ 

        image = img_list[sub_subject_id_start_slice:subject_id_end_slice]
        label = label_list[sub_subject_id_start_slice:subject_id_end_slice]
        sub_subject_id_start_slice = subject_id_end_slice
        subject_id_end_slice = sub_subject_id_start_slice + 155

    
        image = torch.stack(image)
        label = torch.stack(label)


        #utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + 'FETS_TEST_SD_test.nii.gz', data = image, affine = np.eye(4))

        image, label = image.cuda(), label.cuda()      
     
       
        # ==================================================================
        # setup logging
        # ==================================================================
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
        logging.info('============================================================')
        logging.info('Subject ' + str(sub_num+1))

        # ============================
        # Perform the prediction for each test patient individually & calculate dice score and Hausdorff distance
        # ============================  
        
        metric_whole, metric_enhancing, metric_core = test_single_volume_FETS_train_eval(image, label, model, i2n_module_t1, i2n_module_t1ce, i2n_module_t2, i2n_module_flair, use_tta, tta_epochs, writer, layer_names_for_stats, classes=args.num_classes, dataset = 'FETS_SD', optim = 'ADAM', model_type = f'{model_type}', seed = '{seed}', patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=sub_num, z_spacing=args.z_spacing)
        


        logging.info("WHOLE TUMOR:")
        metric_list_whole += np.mean(metric_whole, axis = 0)[0]
        logging.info('case %s mean_dice %f mean_hd95 %f' % (sub_num, np.mean(metric_whole, axis=0)[0], np.mean(metric_whole, axis=0)[1]))

        logging.info("ENHANCING TUMOR:")
        metric_list_enhancing += np.mean(metric_enhancing, axis = 0)[0]
        logging.info('case %s mean_dice %f mean_hd95 %f' % (sub_num, np.mean(metric_enhancing, axis=0)[0], np.mean(metric_enhancing, axis=0)[1]))

        logging.info("TUMOR CORE:")
        metric_list_core += np.mean(metric_core, axis=0)[0]
        logging.info('case %s mean_dice %f mean_hd95 %f' % (sub_num, np.mean(metric_core, axis=0)[0], np.mean(metric_core, axis=0)[1]))

        

    logging.info('---------------------------------------------------------------------------------')
    logging.info(f'Mean dice score on WHOLE TUMOR: {metric_list_whole/num_test_subjects}')
    logging.info(f'Mean dice score on ENHANCING TUMOR: {metric_list_enhancing/num_test_subjects}')
    logging.info(f'Mean dice score on TUMOR CORE: {metric_list_core/num_test_subjects}')
    logging.info('---------------------------------------------------------------------------------')
    

        # ============================
        # Log the mean performance achieved for each class
        # ============================ 


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
        'FETS_train': {
            'volume_path': '/itet-stor/arismu/bmicdatasets-originals/Originals/Challenge_Datasets/FeTS/',
            'num_classes': 4,
            'z_spacing': 1,
        },
    }
    dataset_name = args.test_dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_name
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # ============================
    # Same snapshot path as defined in the train script to access the trained model
    # ============================  

    args.exp = 'TU_' + dataset_name + str(args.img_size) 
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 6800 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 400 else snapshot_path
    if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 1e-3 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    #net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    net = UNET(in_channels = 4, out_channels = 4, features = [32, 64, 128, 256]).cuda()

    snapshot = os.path.join(f'/scratch_net/biwidl217_second/arismu/Master_Thesis_Codes/project_TransUNet/model/2022/FETS/{model_type}/TTA/', f'FETS_{model_type}_best_val_loss_seed{seed}_da{data_aug}_TTA.pth')
    #if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'no_data_aug_' + 'epoch_' + str(args.max_epochs-1))

    # ============================
    # Load the trained parameters into the model
    # ============================  

    net.load_state_dict(torch.load(snapshot))


    #size = sum(p.numel() for p in net.parameters())
    #print(f'Number of parameters: {size}')

    # ============================
    # Logging
    # ============================ 

    snapshot_name = snapshot_path.split('/')[-1]
    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)


    layer_names_for_stats = []
    for name, m in net.named_modules():
        if isinstance(m, torch.nn.modules.conv.Conv2d):
            layer_names_for_stats.append(name)



    # ============================
    # Save the predictions as nii files
    # ============================ 
    #import pdb; pdb.set_trace()
   
    test_save_dir = f'../predictions_2022/FETS/{model_type}/TTA/'
    test_save_path = os.path.join(test_save_dir, f'SD_FETS_{model_type}_test_seed{seed}_da{data_aug}_TTA')
    os.makedirs(test_save_path, exist_ok=True)

    inference(args, net, test_save_path)

print("test.py successfully executed")


