import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.unet_class import UNET
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer_prostate import trainer_prostate

model_type = 'UNET'
load_pretrain = False

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/itet-stor/arismu/bmicdatasets-originals/Originals/Challenge_Datasets/NCI_Prostate/', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='RUNMC', help='experiment_name')
#parser.add_argument('--list_dir', type=str,
#                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=6800, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=400, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=1e-3,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--target_resolution', type=float, default=0.625, help='target resolution')                    


parser.add_argument('--tr_run_number', type = int, default = 3) # 1 / 
parser.add_argument('--tr_cv_fold_num', type = int, default = 1) # 1 / 2
parser.add_argument('--da_ratio', type = float, default = 0.25) # 0.0 / 0.25

args = parser.parse_args()


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
    dataset_name = args.dataset
    dataset_config = {
        'RUNMC': {
            'root_path': '/itet-stor/arismu/bmicdatasets-originals/Originals/Challenge_Datasets/NCI_Prostate/',
            'num_classes': 3,
            'target_resolution': 0.625
        },
    }


    if args.batch_size != 16 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 16
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.target_resolution = dataset_config[dataset_name]['target_resolution']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

    # ===========================    
    # create an instance of the model 
    # ===========================      
    
    if model_type == 'UNWT':
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
        if load_pretrain:
            net.load_from(weights=np.load(config_vit.pretrained_path))

    elif model_type == 'UNET':
        net = UNET(in_channels = 3, out_channels = 3, features = [32, 64, 128, 256]).cuda()
        if load_pretrain:
            net.load_state_dict(torch.load(f'/scratch_net/biwidl217_second/arismu/Master_Thesis_Codes/project_TransUNet/model/2021/{model_type}/{model_type}_seed{args.seed}.pth'))
    else:
        print('Error: model instantiation')



    # ===========================    
    # start training 
    # ===========================  

    trainer = {'RUNMC': trainer_prostate,}
    trainer[dataset_name](args, net)