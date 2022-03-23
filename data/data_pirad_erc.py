import os
import numpy as np
import logging
import utils
import config.system_paths as sys_config
from skimage.transform import rescale
import gc
import h5py


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 5

# ===============================================================
# This function unzips and pre-processes the data if this has not already been done.
# If this already been done, it reads the processed data and returns it.                    
# ===============================================================                         
def load_data(input_folder,
              preproc_folder,
              idx_start,
              idx_end,
              size,
              target_resolution,
              labeller = 'ek',
              force_overwrite = False):

    # ===============================
    # create the pre-processing folder, if it does not exist
    # ===============================
    utils.makefolder(preproc_folder)    
    
    # ===============================
    # file to create or directly read if it already exists
    # ===============================
    #size_str = '_'.join([str(i) for i in size])
    #res_str = '_'.join([str(i) for i in target_resolution])
    data_file_name = 'data_2d_from_%d_to_%d_size_%s_res_%s_%s.hdf5' % (idx_start, idx_end, size, target_resolution, labeller)
    data_file_path = os.path.join(preproc_folder, data_file_name)
    
    # ===============================
    # if the images have not already been extracted, do so
    # ===============================
    if not os.path.exists(data_file_path) or force_overwrite:
        
        logging.info('This configuration of protocol and data indices has not yet been preprocessed')
        logging.info('Preprocessing now...')
        prepare_data(input_folder,
                     data_file_path,
                     idx_start,
                     idx_end,
                     size,
                     target_resolution,
                     labeller)
    else:        
        logging.info('Already preprocessed this configuration. Loading now...')
        
    return h5py.File(data_file_path, 'r')

# ===============================================================
# count the total number of 2d slices to be read: this number is required to specify the dataset size while opening the hdf5 file
# ===============================================================
def count_slices(folder_list,
                 idx_start,
                 idx_end):

    num_slices = 0
    for idx in range(idx_start, idx_end):
        image, _, _ = utils.load_nii(folder_list[idx] + '/t2_tse_tra.nii.gz')
        num_slices = num_slices + image.shape[2]

    return num_slices

# ===============================================================
# Main function that prepares a dataset from the raw challenge data to an hdf5 dataset.
# Extract the required files from their zipped directories
# ===============================================================
def prepare_data(input_folder,
                 output_filepath,
                 idx_start,
                 idx_end,
                 size,
                 target_resolution,
                 labeller):

    # ===============================
    # create a hdf5 file
    # ===============================
    hdf5_file = h5py.File(output_filepath, "w")
    
    # ===============================
    # read all the patient folders from the base input folder
    # ===============================
    folder_list = []
    for folder in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder)        
        if os.path.isdir(folder_path) and 't2_tse_tra.nii.gz' in os.listdir(folder_path):
            if 'segmentation_' + labeller + '.nii.gz' in os.listdir(folder_path) or 'segmentation_tra_' +  labeller + '.nii.gz' in os.listdir(folder_path):
                folder_list.append(folder_path)
    
    # ===============================
    # Create datasets for images and labels
    # ===============================
    data = {}
    num_slices = count_slices(folder_list, idx_start, idx_end)
    data['images'] = hdf5_file.create_dataset("images", list((size,size)) + [num_slices], dtype=np.float32)
    data['labels'] = hdf5_file.create_dataset("labels", list((size,size)) + [num_slices], dtype=np.uint8)
    
    # ===============================
    # initialize lists
    # ===============================        
    label_list = []
    image_list = []
    nx_list = []
    ny_list = []
    nz_list = []
    px_list = []
    py_list = []
    pz_list = []
    pat_names_list = []
    
    # ===============================        
    # ===============================        
    write_buffer = 0
    counter_from = 0
    patient_counter = 0
    
    # ===============================
    # iterate through the requested indices
    # ===============================
    for idx in range(idx_start, idx_end):

        patname = folder_list[idx][folder_list[idx].rfind('/')+1:]

        print(f"NEW PATIENT: {patname}")
        
        patient_counter = patient_counter + 1
        
        # ==================
        # read the image file
        # ==================
        image, _, image_hdr = utils.load_nii(folder_list[idx] + '/t2_tse_tra_n4.nii.gz')

        utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + '00_test.nii.gz', data = image, affine = np.eye(4))

        # ============
        # normalize the image to be between 0 and 1
        # ============
        image_normalized = utils.normalise_image(image, norm_type='div_by_max')

        utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + '01_test.nii.gz', data = image_normalized, affine = np.eye(4))
        
        # ==================
        # collect some header info.
        # ==================
        px_list.append(float(image_hdr.get_zooms()[0]))
        py_list.append(float(image_hdr.get_zooms()[1]))
        pz_list.append(float(image_hdr.get_zooms()[2]))
        nx_list.append(image.shape[0])
        ny_list.append(image.shape[1])
        nz_list.append(image.shape[2])
        pat_names_list.append(folder_list[idx][folder_list[idx].rfind('/')+1:])
        
        # ==================
        # read the label file
        # ==================        
        if 'segmentation_' + labeller + '.nii.gz' in os.listdir(folder_list[idx]):
            label, _, _ = utils.load_nii(folder_list[idx] + '/segmentation_' + labeller + '.nii.gz')
        elif 'segmentation_tra_' + labeller + '.nii.gz' in os.listdir(folder_list[idx]):
            label, _, _ = utils.load_nii(folder_list[idx] + '/segmentation_tra_' + labeller + '.nii.gz')

        
        

        
        # ================== 
        # remove extra label from some images
        # ================== 
        label[label > 2] = 0
        print(np.unique(label))
        utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + '01_label.nii.gz', data = label, affine = np.eye(4))
        # ======================================================  
        ### PROCESSING LOOP FOR SLICE-BY-SLICE 2D DATA ###################
        # ======================================================
        scale_vector = [image_hdr.get_zooms()[0] / target_resolution,
                        image_hdr.get_zooms()[1] / target_resolution]

        for zz in range(image.shape[2]):

            print(f'Max label value: {label.max()}')
            print(f'Min label value: {label.min()}')

            # ============
            # rescale the images and labels so that their orientation matches that of the nci dataset
            # ============        

            
            
            if patname in ['88800036', '88800035', '88800046', '88800038', '88800021', '88800053']:
                    # For these images, rescaling directly to the target resolution (0.625) leads to faultily rescaled labels (all pixels get the value 0)
                    # Not sure what is causing this.
                    # Using this intermediate scaling as a workaround.
                    scale_vector_tmp = [image_hdr.get_zooms()[0] / 0.625, image_hdr.get_zooms()[1] / 0.625]
                    image2d_rescaled = rescale(image_normalized[:, :, zz], scale_vector_tmp, order=1, preserve_range=True, multichannel=False, mode = 'constant')
                    label2d_rescaled = rescale(label[:, :, zz], scale_vector_tmp, order=0, preserve_range=True, multichannel=False, mode='constant')
                    utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + '201_label.nii.gz', data = label2d_rescaled, affine = np.eye(4))
                    print(f'1Max label value label2d_rescaled: {label2d_rescaled.max()}')
                    print(f'1Min label value label2d_rescaled: {label2d_rescaled.min()}')
                    scale_vector_tmp = [0.625 / target_resolution, 0.625 / target_resolution]
                    image2d_rescaled = rescale(image2d_rescaled, scale_vector_tmp, order=1, preserve_range=True, multichannel=False, mode = 'constant')
                    label2d_rescaled = rescale(label2d_rescaled, scale_vector_tmp, order=0, preserve_range=True, multichannel=False, mode='constant')
                    utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + '301_label.nii.gz', data = label2d_rescaled, affine = np.eye(4))
            else:
            
                    image2d_rescaled = rescale(np.squeeze(image_normalized[:, :, zz]),
                                                  scale_vector,
                                                  order=1,
                                                  preserve_range=True,
                                                  multichannel=False,
                                                  mode = 'constant')

                    
                                      
 
                    label2d_rescaled = rescale(np.squeeze(label[:, :, zz]),
                                                  scale_vector,
                                                  order=0,
                                                  preserve_range=True,
                                                  multichannel=False,
                                                  mode='constant')
            
            utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + '02_test.nii.gz', data = image2d_rescaled, affine = np.eye(4))
            utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + '02_label.nii.gz', data = label2d_rescaled, affine = np.eye(4))

            print(f'2Max label value label2d_rescaled: {label2d_rescaled.max()}')
            print(f'2Min label value label2d_rescaled: {label2d_rescaled.min()}')

            # ============
            # rotate the images and labels so that their orientation matches that of the nci dataset
            # ============            
            image2d_rescaled_rotated = np.rot90(image2d_rescaled, k=3)
            label2d_rescaled_rotated = np.rot90(label2d_rescaled, k=3)

            utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + '03_test.nii.gz', data = image2d_rescaled_rotated, affine = np.eye(4))

            # ============            
            # crop or pad to make of the same size
            # ============            
            image2d_rescaled_rotated_cropped = crop_or_pad_slice_to_size(image2d_rescaled_rotated, size, size)
            label2d_rescaled_rotated_cropped = crop_or_pad_slice_to_size(label2d_rescaled_rotated, size, size)

            utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + '04_test.nii.gz', data = image2d_rescaled_rotated, affine = np.eye(4))

            image_list.append(image2d_rescaled_rotated_cropped)
            label_list.append(label2d_rescaled_rotated_cropped)

            write_buffer += 1

            # Writing needs to happen inside the loop over the slices
            if write_buffer >= MAX_WRITE_BUFFER:

                counter_to = counter_from + write_buffer

                _write_range_to_hdf5(data,
                                     image_list,
                                     label_list,
                                     counter_from,
                                     counter_to)
                
                _release_tmp_memory(image_list,
                                    label_list)

                # update counters 
                counter_from = counter_to
                write_buffer = 0

        print(f'Max label value image2d_rescaled_rotated_cropped: {image2d_rescaled_rotated_cropped.max()}')
        print(f'Min label value image2d_rescaled_rotated_cropped: {image2d_rescaled_rotated_cropped.min()}')
        utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + '05_test.nii.gz', data = image2d_rescaled, affine = np.eye(4))
        utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + '05_label.nii.gz', data = label2d_rescaled, affine = np.eye(4))
        
    logging.info('Writing remaining data')
    counter_to = counter_from + write_buffer
    #write_range_to_hdf5(data,
    #                     image_list,
    #                     label_list,
    #                     counter_from,
    #                     counter_to)
    #_release_tmp_memory(image_list,
    #                    label_list)

    # Write the small datasets
    hdf5_file.create_dataset('nx', data=np.asarray(nx_list, dtype=np.uint16))
    hdf5_file.create_dataset('ny', data=np.asarray(ny_list, dtype=np.uint16))
    hdf5_file.create_dataset('nz', data=np.asarray(nz_list, dtype=np.uint16))
    hdf5_file.create_dataset('px', data=np.asarray(px_list, dtype=np.float32))
    hdf5_file.create_dataset('py', data=np.asarray(py_list, dtype=np.float32))
    hdf5_file.create_dataset('pz', data=np.asarray(pz_list, dtype=np.float32))
    hdf5_file.create_dataset('patnames', data=np.asarray(pat_names_list, dtype="S10"))
    
    # After test train loop:
    hdf5_file.close()

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
# Helper function to write a range of data to the hdf5 datasets
# ===============================================================
def _write_range_to_hdf5(hdf5_data,
                         img_list,
                         mask_list,
                         counter_from,
                         counter_to):

    logging.info('Writing data from %d to %d' % (counter_from, counter_to))

    img_arr = np.asarray(img_list, dtype=np.float32)
    lab_arr = np.asarray(mask_list, dtype=np.uint8)

    img_arr = np.swapaxes(img_arr, 0, 2)
    lab_arr = np.swapaxes(lab_arr, 0, 2)

    hdf5_data['images'][..., counter_from : counter_to] = img_arr
    hdf5_data['labels'][..., counter_from : counter_to] = lab_arr

# ===============================================================
# Helper function to reset the tmp lists and free the memory
# ===============================================================
def _release_tmp_memory(img_list,
                        mask_list):

    img_list.clear()
    mask_list.clear()
    gc.collect()

# ===============================================================
# function to read a single subjects image and labels without any pre-processing
# ===============================================================
def load_without_size_preprocessing(input_folder,
                                    subject_name,
                                    labeller):

    # ==================
    # read the image file
    # ==================              
    folder_path = input_folder + subject_name
    image, _, _ = utils.load_nii(folder_path + '/t2_tse_tra_n4.nii.gz')
    # ============
    # normalize the image to be between 0 and 1
    # ============
    image = utils.normalise_image(image, norm_type='div_by_max')
    
    # ==================
    # read the label file
    # ==================        
    if 'segmentation_' + labeller + '.nii.gz' in os.listdir(folder_path):
        label, _, _ = utils.load_nii(folder_path + '/segmentation_' + labeller + '.nii.gz')
    elif 'segmentation_tra_' + labeller + '.nii.gz' in os.listdir(folder_path):
        label, _, _ = utils.load_nii(folder_path + '/segmentation_tra_' + labeller + '.nii.gz')    
    # ================== 
    # remove extra label from some images
    # ================== 
    label[label > 2] = 0
    
    return image, label
        
# ===============================================================
# Main function that runs if this file is run directly
# ===============================================================
if __name__ == '__main__':
    
    data_pirad_erc_test = load_data(input_folder = sys_config.orig_data_root_pirad_erc,
                                    preproc_folder = sys_config.preproc_folder_pirad_erc,
                                    idx_start = 0,
                                    idx_end = 20,
                                    size = (256, 256),
                                    target_resolution = (0.625, 0.625),
                                    labeller = 'ek')
    
    data_pirad_erc_val = load_data(input_folder = sys_config.orig_data_root_pirad_erc,
                                   preproc_folder = sys_config.preproc_folder_pirad_erc,
                                   idx_start = 20,
                                   idx_end = 40,
                                   size = (256, 256),
                                   target_resolution = (0.625, 0.625),
                                   labeller = 'ek')
    
    data_pirad_erc_train = load_data(input_folder = sys_config.orig_data_root_pirad_erc,
                                     preproc_folder = sys_config.preproc_folder_pirad_erc,
                                     idx_start = 40,
                                     idx_end = 68,
                                     size = (256, 256),
                                     target_resolution = (0.625, 0.625),
                                     labeller = 'ek')
    
# ===============================================================
# End of file
# ===============================================================