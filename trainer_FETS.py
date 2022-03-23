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
from normalisation_module import Normalisation_Module_flair, Normalisation_Module_t1, Normalisation_Module_t1ce, Normalisation_Module_t2
import torch.nn.functional as F


seed = 1234
model_type = 'UNET'
data_aug = '0.25_TTA_NO_DICT_OPTIM'

def trainer_fets(args, model):


    i2n_module_t1 = Normalisation_Module_t1(in_channels = 1)
    i2n_module_t1ce = Normalisation_Module_t1ce(in_channels = 1)
    i2n_module_t2 = Normalisation_Module_t2(in_channels = 1)
    i2n_module_flair = Normalisation_Module_flair(in_channels = 1)

    

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
    imtr_part1 = loaded_tr_data[0]
    gttr_part1 = loaded_tr_data[1]
    imtr_part2 = loaded_tr_data[3]
    gttr_part2 = loaded_tr_data[4]
    imtr_part3 = loaded_tr_data[6]
    gttr_part3 = loaded_tr_data[7]

    loaded_val_data = utils_data.load_validation_data('FETS_val',
                                                   args.img_size,
                                                   args.target_resolution,
                                                   args.tr_cv_fold_num)


    imvl = loaded_val_data[0]
    gtvl = loaded_val_data[1]
    imvl = np.array(imvl)
    gtvl = np.array(gtvl)

    
    gtvl[np.where(gtvl == 4)] = 3 
    
    

    
    imtr = np.concatenate((imtr_part1, imtr_part2), axis = 2) 
    imtr = np.concatenate((imtr, imtr_part3), axis = 2)

    gttr = np.concatenate((gttr_part1, gttr_part2), axis = 2) 
    gttr = np.concatenate((gttr, gttr_part3), axis = 2)

    
    
    gttr[np.where(gttr == 4)] = 3  #turn labels [0 1 2 4] into [0 1 2 3]



    imtr = torch.from_numpy(imtr)
    gttr = torch.from_numpy(gttr)

    imvl = torch.from_numpy(imvl)
    gtvl = torch.from_numpy(gtvl)


    
    img_list = []
    label_list = []

    val_img_list = []
    val_label_list = []



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

    for i in range(imvl.shape[2]):
        if lim1 != 0 and (lim1 % 155) == 0:
            lim1 = lim4
            lim2 = lim1 + 155
            lim3 = lim1 + 310
            lim4 = lim1 + 465
        x.append(imvl[:, :, lim1])
        x.append(imvl[:, :, lim2])
        x.append(imvl[:, :, lim3])
        x.append(imvl[:, :, lim4])
        y = torch.stack(x, dim = -1)
        val_img_list.append(y)


        lim1 += 1
        lim2 += 1
        lim3 += 1
        lim4 += 1
        x = []
        
        if lim4 == (imvl.shape[2]):
            break

    lim1 = 0
    lim2 = 155 
    lim3 = 310
    lim4 = 465
    x = []



    for i in range(gtvl.shape[2]):
        if lim1 != 0 and (lim1 % 155) == 0:
            lim1 = lim4
            lim2 = lim1 + 155
            lim3 = lim1 + 310
            lim4 = lim1 + 465
        x.append(gtvl[:, :, lim1])
        x.append(gtvl[:, :, lim2])
        x.append(gtvl[:, :, lim3])
        x.append(gtvl[:, :, lim4])
        y = torch.stack(x, dim = -1)
        val_label_list.append(y)


        lim1 += 1
        lim2 += 1
        lim3 += 1
        lim4 += 1
        x = []
        
        if lim4 == (gtvl.shape[2]):
            break
    

    

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
        i2n_module_t1 = nn.DataParallel(i2n_module_t1)
        i2n_module_t1ce = nn.DataParallel(i2n_module_t1ce)
        i2n_module_t2 = nn.DataParallel(i2n_module_t2)
        i2n_module_flair = nn.DataParallel(i2n_module_flair)
    model.train()
    i2n_module_t1.train()
    i2n_module_t1ce.train()
    i2n_module_t2.train()
    i2n_module_flair.train()
    
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    optim_i2n_t1 = optim.Adam(i2n_module_t1.parameters(), lr=base_lr)
    optim_i2n_t1ce = optim.Adam(i2n_module_t1ce.parameters(), lr=base_lr)
    optim_i2n_t2 = optim.Adam(i2n_module_t2.parameters(), lr=base_lr)
    optim_i2n_flair = optim.Adam(i2n_module_flair.parameters(), lr=base_lr)
    writer = SummaryWriter(f'/scratch_net/biwidl217_second/arismu/Tensorboard/2022/FETS/{model_type}/TTA/' + f'FETS_{model_type}_log_seed{seed}_da{data_aug}') 
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = 119351
    logging.info("{} iterations per epoch. {} max iterations ".format(args.batch_size+1 , max_iterations))
    best_val_loss = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============================
    # Training loop: loop over batches and perform data augmentation on the fly with a certain probability
    # ============================  

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        print(f"EPOCH: {epoch_num}")
        for sampled_batch in iterate_minibatches(args, img_list, label_list, batch_size = exp_config.batch_size, train_or_eval = 'train'):
            model.train()
            i2n_module_t1.train()
            i2n_module_t1ce.train()
            i2n_module_t2.train()
            i2n_module_flair.train()

            i2n_module_t1.to(device)
            i2n_module_t1ce.to(device)
            i2n_module_t2.to(device)
            i2n_module_flair.to(device)
    
            
            image_batch, label_batch = sampled_batch[0], sampled_batch[1]
            image_batch = torch.from_numpy(image_batch)
            label_batch = torch.from_numpy(label_batch)
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()      
            image_batch = image_batch.permute(3, 2, 0, 1)
            label_batch = label_batch.permute(3, 2, 0, 1)
            norm_output_t1 = i2n_module_t1(image_batch[:, 0, :, :].unsqueeze(1))
            norm_output_t1ce = i2n_module_t1ce(image_batch[:, 1, :, :].unsqueeze(1))
            norm_output_t2 = i2n_module_t2(image_batch[:, 2, :, :].unsqueeze(1))
            norm_output_flair = i2n_module_flair(image_batch[:, 3, :, :].unsqueeze(1))

            norm_output = torch.cat((norm_output_t1, norm_output_t1ce, norm_output_t2, norm_output_flair), 1)

            norm_output = norm_output.cuda()

            
            outputs = model(norm_output)            
            loss_ce = ce_loss(outputs, label_batch[:, 0, :, :].long())
            loss_dice = dice_loss(outputs, label_batch[:, 0, :, :], softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            optim_i2n_t1.zero_grad()
            optim_i2n_t1ce.zero_grad()
            optim_i2n_t2.zero_grad()
            optim_i2n_flair.zero_grad()
            loss.backward()
            optimizer.step()
            optim_i2n_t1.step()
            optim_i2n_t1ce.step()
            optim_i2n_t2.step()
            optim_i2n_flair.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            for param_group in optim_i2n_t1.param_groups:
                param_group['lr'] = lr_
            for param_group in optim_i2n_t1ce.param_groups:
                param_group['lr'] = lr_   
            for param_group in optim_i2n_t2.param_groups:
                param_group['lr'] = lr_
            for param_group in optim_i2n_flair.param_groups:
                param_group['lr'] = lr_
            
            #add lr reduction for all optimizers

    # ============================
    # Write to Tensorboard
    # ============================  

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))


            #if iter_num % 500 == 0:
            #    image = image_batch[1, 0:4, :, :]
            #    image = (image - image.min()) / (image.max() - image.min())
            #    writer.add_image('train/Image', image, iter_num)
            #    outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
            #    writer.add_image('train/Prediction', outputs[1, 0:4, :, :] * 50, iter_num)
                #labs = label_batch[1, 0, :, :].unsqueeze(0) * 50
                #writer.add_image('train/GroundTruth', labs, iter_num)

            

            # ===========================
            # Compute the loss on the entire training set
            # ===========================
            if (iter_num+1) % 853 == 0:   #every epoch (3410 iterations per epoch)
                logging.info('Training Data Eval:')
                train_loss = do_train_eval(img_list, label_list, batch_size, model, ce_loss, dice_loss)                   
                
                logging.info('  Average segmentation loss on training set: %.4f' % (train_loss))

                writer.add_scalar('info/total_loss_training_set', train_loss, iter_num)
                

            # ===========================
            # Evaluate the model periodically on a validation set 
            # ===========================
            if (iter_num+1) % 853 == 0:
                logging.info('Validation Data Eval:')
                val_loss = do_validation_eval(val_img_list, val_label_list, batch_size, model, ce_loss, dice_loss)                    
                
                logging.info('  Average segmentation loss on validation set: %.4f' % (val_loss))

                writer.add_scalar('info/total_loss_validation_set', val_loss, iter_num)

                if val_loss < best_val_loss:
                    save_mode_path = os.path.join(f'/scratch_net/biwidl217_second/arismu/Master_Thesis_Codes/project_TransUNet/model/2022/FETS/{model_type}/TTA/', f'FETS_{model_type}_best_val_loss_seed{seed}_da{data_aug}' + '.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    save_t1_path = os.path.join(f'/scratch_net/biwidl217_second/arismu/Master_Thesis_Codes/project_TransUNet/model/2022/FETS/{model_type}/TTA/', f'FETS_{model_type}_{seed}_da{data_aug}_NORM_T1' + '.pth')
                    save_t1ce_path = os.path.join(f'/scratch_net/biwidl217_second/arismu/Master_Thesis_Codes/project_TransUNet/model/2022/FETS/{model_type}/TTA/', f'FETS_{model_type}_{seed}_da{data_aug}_NORM_T1CE' + '.pth')
                    save_t2_path = os.path.join(f'/scratch_net/biwidl217_second/arismu/Master_Thesis_Codes/project_TransUNet/model/2022/FETS/{model_type}/TTA/', f'FETS_{model_type}_{seed}_da{data_aug}_NORM_T2' + '.pth')
                    save_flair_path = os.path.join(f'/scratch_net/biwidl217_second/arismu/Master_Thesis_Codes/project_TransUNet/model/2022/FETS/{model_type}/TTA/', f'FETS_{model_type}_{seed}_da{data_aug}_NORM_FLAIR' + '.pth')

                    torch.save(i2n_module_t1.state_dict(), save_t1_path)
                    torch.save(i2n_module_t1ce.state_dict(), save_t1ce_path)
                    torch.save(i2n_module_t2.state_dict(), save_t2_path)
                    torch.save(i2n_module_flair.state_dict(), save_flair_path)

                    logging.info(f"Found new lowest validation loss at iteration {iter_num}! Save model to {save_mode_path}")
                    best_val_loss = val_loss
            
            #if (iter_num+1) % 10000 == 0:
             #   logging.info(f'Saving model at iteration {iter_num}')
            #    save_mode_path = os.path.join(f'/scratch_net/biwidl217_second/arismu/Master_Thesis_Codes/project_TransUNet/model/2022/FETS/{model_type}/', f'FETS_{model_type}_seed{seed}_iternum{iter_num}_da{data_aug}' + '.pth')
            #    torch.save(model.state_dict(), save_mode_path)

    # ============================
    # Save the trained model parameters
    # ============================  


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
    #n_images = images.shape[2]
    n_images = len(images)
    random_indices = np.random.permutation(n_images)

    # ===========================
    for b_i in range(n_images // batch_size):

        if b_i + batch_size > n_images:
            continue

        
        batch_indices = np.sort(random_indices[b_i*batch_size:(b_i+1)*batch_size])

        b_list = []
        lab_list = []

        for b in batch_indices:
            b_list.append(images[b])
            lab_list.append(labels[b])
        x = torch.stack(b_list)
        y = torch.stack(lab_list)
        x = x.permute(1, 2, 3, 0)
        y = y.permute(1, 2, 3, 0)
        


        # ===========================    
        # data augmentation (contrast changes + random elastic deformations)
        # ===========================      
        if args.da_ratio > 0.0:

            # ===========================    
            # doing data aug both during training as well as during evaluation on the validation set (used for model selection)
            # ===========================             
            # 90 degree rotation for cardiac images as the orientation is fixed for all other anatomies.
            do_rot90 = args.dataset in ['HVHD', 'CSF', 'UHE']

            
            x, y = utils.do_data_augmentation_FETS(images = x,
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

                

               

        #x = np.expand_dims(x, axis=-1)
        
        yield x, y


def do_train_eval(images, labels, batch_size, model, ce_loss, dice_loss):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    i2n_module_t1 = Normalisation_Module_t1(in_channels = 1)
    i2n_module_t1ce = Normalisation_Module_t1ce(in_channels = 1)
    i2n_module_t2 = Normalisation_Module_t2(in_channels = 1)
    i2n_module_flair = Normalisation_Module_flair(in_channels = 1)

    i2n_module_t1.to(device)
    i2n_module_t1ce.to(device)
    i2n_module_t2.to(device)
    i2n_module_flair.to(device)

    n_images = len(images)
    random_indices = np.random.permutation(n_images)

    loss_ii = 0
    num_batches = 0


    model.eval()
    i2n_module_t1.eval()
    i2n_module_t1ce.eval()
    i2n_module_t2.eval()
    i2n_module_flair.eval()

    with torch.no_grad():
        # ===========================
        for b_i in range(n_images // batch_size):

            if b_i + batch_size > n_images:
                continue
            
            batch_indices = np.sort(random_indices[b_i*batch_size:(b_i+1)*batch_size])

            b_list = []
            lab_list = []

            for b in batch_indices:
                b_list.append(images[b])
                lab_list.append(labels[b])
            x = torch.stack(b_list)
            y = torch.stack(lab_list)

            
            
            
            x, y = x.cuda(), y.cuda()   

            x = x.permute(0, 3, 1, 2)
            y = y.permute(0, 3, 1, 2)

            norm_output_t1 = i2n_module_t1(x[:, 0, :, :].unsqueeze(1))
            norm_output_t1ce = i2n_module_t1ce(x[:, 1, :, :].unsqueeze(1))
            norm_output_t2 = i2n_module_t2(x[:, 2, :, :].unsqueeze(1))
            norm_output_flair = i2n_module_flair(x[:, 3, :, :].unsqueeze(1))

            norm_output = torch.cat((norm_output_t1, norm_output_t1ce, norm_output_t2, norm_output_flair), 1)
            
            norm_output = norm_output.cuda()
            
            outputs = model(norm_output)
            train_loss_ce = ce_loss(outputs, y[:, 0, :, :].long())
            train_loss_dice = dice_loss(outputs, y[:, 0, :, :], softmax=True)
            train_loss = 0.5 * train_loss_ce + 0.5 * train_loss_dice

            loss_ii += train_loss
            num_batches += 1

        
        avg_loss = loss_ii / num_batches

        return avg_loss      


def do_validation_eval(images, labels, batch_size, model, ce_loss, dice_loss):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    i2n_module_t1 = Normalisation_Module_t1(in_channels = 1)
    i2n_module_t1ce = Normalisation_Module_t1ce(in_channels = 1)
    i2n_module_t2 = Normalisation_Module_t2(in_channels = 1)
    i2n_module_flair = Normalisation_Module_flair(in_channels = 1)

    i2n_module_t1.to(device)
    i2n_module_t1ce.to(device)
    i2n_module_t2.to(device)
    i2n_module_flair.to(device)

    writer = SummaryWriter(f'/scratch_net/biwidl217_second/arismu/Tensorboard/2022/FETS/{model_type}/TTA/' + f'FETS_{model_type}_log_seed{seed}_da{data_aug}') 

    n_images = len(images)
    random_indices = np.random.permutation(n_images)

    loss_ii = 0
    num_batches = 0

    model.eval()
    i2n_module_t1.eval()
    i2n_module_t1ce.eval()
    i2n_module_t2.eval()
    i2n_module_flair.eval()

    with torch.no_grad():
        # ===========================
        for b_i in range(n_images // batch_size):

            if b_i + batch_size > n_images:
                continue
            
            batch_indices = np.sort(random_indices[b_i*batch_size:(b_i+1)*batch_size])

            b_list = []
            lab_list = []

            for b in batch_indices:
                b_list.append(images[b])
                lab_list.append(labels[b])
            x = torch.stack(b_list)
            y = torch.stack(lab_list)
            #x = images[..., batch_indices]
            #y = labels[..., batch_indices]

            #x = np.expand_dims(x, axis=-1)

            #x = torch.from_numpy(x)
            #y = torch.from_numpy(y)
            
            x, y = x.cuda(), y.cuda()   

            x = x.permute(0, 3, 1, 2)
            y = y.permute(0, 3, 1, 2)


            norm_output_t1 = i2n_module_t1(x[:, 0, :, :].unsqueeze(1))
            norm_output_t1ce = i2n_module_t1ce(x[:, 1, :, :].unsqueeze(1))
            norm_output_t2 = i2n_module_t2(x[:, 2, :, :].unsqueeze(1))
            norm_output_flair = i2n_module_flair(x[:, 3, :, :].unsqueeze(1))

            norm_output = torch.cat((norm_output_t1, norm_output_t1ce, norm_output_t2, norm_output_flair), 1)

            norm_output = norm_output.cuda()

            outputs = model(norm_output)
            val_loss_ce = ce_loss(outputs, y[:, 0, :, :].long())
            val_loss_dice = dice_loss(outputs, y[:, 0, :, :], softmax=True)
            val_loss = 0.5 * val_loss_ce + 0.5 * val_loss_dice


            loss_ii += val_loss
            num_batches += 1

            #if b_i % 100 == 0:
            #    image = x[1, 0:4, :, :]
            #    image = (image - image.min()) / (image.max() - image.min())
            #    writer.add_image('validation/Image', image, b_i)
            #    outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
            #    writer.add_image('validation/Prediction', outputs[1, 0:4, :, :] * 50, b_i)
            #    #labs = y[1, ...].unsqueeze(0) * 50
                #writer.add_image('train/GroundTruth', labs[0, :, :], b_i)

        
        avg_loss = loss_ii / num_batches

        return avg_loss 

