import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
import utils_data 
import config.system_paths as sys_config
import config.params as exp_config
import utils

model_type = 'UNET'
seed = 1234

def trainer_prostate(args, model):


    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # ============================
    # Load training data
    # ============================   
    logging.info('============================================================')
    logging.info('Loading data...')
    loaded_tr_data = utils_data.load_training_data(args.dataset,
                                                   args.img_size,
                                                   args.target_resolution,
                                                   args.tr_cv_fold_num)
    imtr = loaded_tr_data[0]
    gttr = loaded_tr_data[1]
    imvl = loaded_tr_data[9]
    gtvl = loaded_tr_data[10]

    logging.info('Training Images: %s' %str(imtr.shape)) # expected: [num_slices, img_size_x, img_size_y]
    logging.info('Training Labels: %s' %str(gttr.shape)) # expected: [num_slices, img_size_x, img_size_y]
    logging.info('Validation Images: %s' %str(imvl.shape))
    logging.info('Validation Labels: %s' %str(gtvl.shape))
    logging.info('============================================================')


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # ============================
    # Loss, optimizer, etc.
    # ============================  

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    writer = SummaryWriter('/scratch_net/biwidl217_second/arismu/Tensorboard/2022/' + f'{model_type}_log_seed{seed}') 
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * (args.batch_size+1) 
    logging.info("{} iterations per epoch. {} max iterations ".format(args.batch_size+1 , max_iterations))
    best_val_loss = 1.0

    # ============================
    # Training loop: loop over batches and perform data augmentation on the fly with a certain probability
    # ============================  

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for sampled_batch in iterate_minibatches(args, imtr, gttr, batch_size = exp_config.batch_size, train_or_eval = 'train'):
            model.train()
            image_batch, label_batch = sampled_batch[0], sampled_batch[1]
            image_batch = torch.from_numpy(image_batch)
            label_batch = torch.from_numpy(label_batch)
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()      
            image_batch = image_batch.permute(2, 3, 0, 1)
            label_batch = label_batch.permute(2, 0, 1)
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

    # ============================
    # Write to Tensorboard
    # ============================  

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 17 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)


            # ===========================
            # Compute the loss on the entire training set
            # ===========================
            if (iter_num+1) % 85 == 0:   #every 5 epochs (17 iterations per epoch)
                logging.info('Training Data Eval:')
                train_loss = do_train_eval(imtr, gttr, batch_size, model, ce_loss, dice_loss)                   
                
                logging.info('  Average segmentation loss on training set: %.4f' % (train_loss))

                writer.add_scalar('info/total_loss_training_set', train_loss, iter_num)
                

            # ===========================
            # Evaluate the model periodically on a validation set 
            # ===========================
            if (iter_num+1) % 85 == 0:
                logging.info('Validation Data Eval:')
                val_loss = do_validation_eval(imvl, gtvl, batch_size, model, ce_loss, dice_loss)                    
                
                logging.info('  Average segmentation loss on validation set: %.4f' % (val_loss))

                writer.add_scalar('info/total_loss_validation_set', val_loss, iter_num)

                if val_loss < best_val_loss:
                    save_mode_path = os.path.join(f'/scratch_net/biwidl217_second/arismu/Master_Thesis_Codes/project_TransUNet/model/2022/{model_type}/', f'{model_type}_best_val_loss_seed{seed}' + '.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info(f"Found new lowest validation loss at iteration {iter_num}! Save model to {save_mode_path}")
                    best_val_loss = val_loss
            

    writer.close()
    return "Training Finished!"


def iterate_minibatches(args, 
                        images,
                        labels,
                        batch_size,
                        train_or_eval = 'train'):

    # ===========================
    # generate indices to randomly select subjects in each minibatch
    # ===========================
    n_images = images.shape[2]
    random_indices = np.random.permutation(n_images)

    # ===========================
    for b_i in range(n_images // batch_size):

        if b_i + batch_size > n_images:
            continue

        
        batch_indices = np.sort(random_indices[b_i*batch_size:(b_i+1)*batch_size])
        
        x = images[..., batch_indices]
        y = labels[..., batch_indices]

        # ===========================    
        # data augmentation (contrast changes + random elastic deformations)
        # ===========================      
        if args.da_ratio > 0.0:

            # ===========================    
            # doing data aug both during training as well as during evaluation on the validation set (used for model selection)
            # ===========================             
            # 90 degree rotation for cardiac images as the orientation is fixed for all other anatomies.
            do_rot90 = args.dataset in ['HVHD', 'CSF', 'UHE']
            x, y = utils.do_data_augmentation(images = x,
                                              labels = y,
                                              data_aug_ratio = args.da_ratio,
                                              sigma = exp_config.sigma,
                                              alpha = exp_config.alpha,
                                              trans_min = exp_config.trans_min,
                                              trans_max = exp_config.trans_max,
                                              rot_min = exp_config.rot_min,
                                              rot_max = exp_config.rot_max,
                                              scale_min = exp_config.scale_min,
                                              scale_max = exp_config.scale_max,
                                              gamma_min = exp_config.gamma_min,
                                              gamma_max = exp_config.gamma_max,
                                              brightness_min = exp_config.brightness_min,
                                              brightness_max = exp_config.brightness_max,
                                              noise_min = exp_config.noise_min,
                                              noise_max = exp_config.noise_max,
                                              rot90 = do_rot90)

        x = np.expand_dims(x, axis=-1)
        
        yield x, y


def do_train_eval(images, labels, batch_size, model, ce_loss, dice_loss):


    n_images = images.shape[2]
    random_indices = np.random.permutation(n_images)

    loss_ii = 0
    num_batches = 0

    model.eval()

    with torch.no_grad():
        # ===========================
        for b_i in range(n_images // batch_size):

            if b_i + batch_size > n_images:
                continue
            
            batch_indices = np.sort(random_indices[b_i*batch_size:(b_i+1)*batch_size])
            
            x = images[..., batch_indices]
            y = labels[..., batch_indices]

            x = np.expand_dims(x, axis=-1)

            x = torch.from_numpy(x)
            y = torch.from_numpy(y)
            
            x, y = x.cuda(), y.cuda()   

            x = x.permute(2, 3, 0, 1)
            y = y.permute(2, 0, 1)
            
            
            outputs = model(x)
            train_loss_ce = ce_loss(outputs, y[:].long())
            train_loss_dice = dice_loss(outputs, y, softmax=True)
            train_loss = 0.5 * train_loss_ce + 0.5 * train_loss_dice

            loss_ii += train_loss
            num_batches += 1

        
        avg_loss = loss_ii / num_batches

        return avg_loss      


def do_validation_eval(images, labels, batch_size, model, ce_loss, dice_loss):


    n_images = images.shape[2]
    random_indices = np.random.permutation(n_images)

    loss_ii = 0
    num_batches = 0

    model.eval()

    with torch.no_grad():
        # ===========================
        for b_i in range(n_images // batch_size):

            if b_i + batch_size > n_images:
                continue
            
            batch_indices = np.sort(random_indices[b_i*batch_size:(b_i+1)*batch_size])
            
            x = images[..., batch_indices]
            y = labels[..., batch_indices]

            x = np.expand_dims(x, axis=-1)

            x = torch.from_numpy(x)
            y = torch.from_numpy(y)
            
            x, y = x.cuda(), y.cuda()   

            x = x.permute(2, 3, 0, 1)
            y = y.permute(2, 0, 1)

            outputs = model(x)
            val_loss_ce = ce_loss(outputs, y[:].long())
            val_loss_dice = dice_loss(outputs, y, softmax=True)
            val_loss = 0.5 * val_loss_ce + 0.5 * val_loss_dice


            loss_ii += val_loss
            num_batches += 1

        
        avg_loss = loss_ii / num_batches

        return avg_loss 
