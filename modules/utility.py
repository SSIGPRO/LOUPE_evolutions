
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import os
from modules import models
import tensorflow.keras.backend as K

from tqdm import tqdm

def _rotate_corners_mask(mri, plot = False):
    """
        Splits the image in 4 windows and rotates them by 180°.
        
        This enambles to visualize the mask coherentely
        with our paper and with the original paper or LOUPE.
    """
    
    def rot180(inpu):
        output = np.rot90(np.rot90(inpu))
        return output
    
    row = int(np.shape(mri)[0]/2)
    col = int(np.shape(mri)[1]/2)

    ul = mri[:row,:col] # upper left
    ur = mri[:row,col:] # upper right
    bl = mri[row:,:col] # below left
    br = mri[row:,col:] # below right
    
    ul_rotated = rot180(ul)
    ur_rotated = rot180(ur)
    bl_rotated = rot180(bl)
    br_rotated = rot180(br)

    u = np.append(ul_rotated, ur_rotated, 1)
    b = np.append(bl_rotated, br_rotated, 1)
    mri_rotated = np.append(u,b,0)
    
    if plot == True:
        fig, ax = plt.subplots()
        ax.imshow(mri_rotated, vmin = 0, vmax = 1, cmap = 'gray')
        ax.title.set_text('mask')
        ax.axis('off')
    return mri_rotated

def handle_GPUs(GPUs = None, enable_GPU=1,):
    """
        For the specified GPUs, "memory growth" is activated.
        The un-specified GPUs are turned invisible to the kernel.
        
        GPUs is a string contraining the number of GPUs (e.g., GPUs = '0,1')
        If GPUs == None all GPUs are activated.
    """
    
    if GPUs != None:
        os.environ["CUDA_VISIBLE_DEVICES"]= GPUs
    
    physical_gpus = tf.config.list_physical_devices('GPU')
    if physical_gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for _gpu in physical_gpus:
                tf.config.experimental.set_memory_growth(_gpu, enable_GPU)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print('number of Logical GPUs:', len(logical_gpus))
    pass

def callbacks(ldir = None,   
              checkpoint = None,   
              monitor='val_loss',   
              patienceEarlyStop=100,  
              patienceLR=50,  
              min_lr = 0.00001,  
              min_delta = 0.00000001,  
              redureLR = 0.2,  
              verbose = 1,  
             ):
    
    callback_list=[
        tf.keras.callbacks.TerminateOnNaN(),       
        tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor,
                                             factor = redureLR,
                                             patience = patienceLR,
                                             min_lr = min_lr,
                                             min_delta=min_delta,
                                             verbose = verbose,
                                            ),
      
        tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                         min_delta=min_delta,
                                         patience=patienceEarlyStop,
                                         verbose=verbose,
                                         mode='auto',
                                         baseline=None,
                                         restore_best_weights=True,
                                        ),                 
    ]
    
    if ldir != None:
        callback_list += [        
            tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint,
                                               monitor = monitor,
                                               verbose = verbose,
                                               save_best_only = True,
                                               save_weights_only = True,
                                               save_freq = 'epoch',
                                              ),
        ]
    if checkpoint != None:
        callback_list += [
            tf.keras.callbacks.TensorBoard(log_dir=ldir,
                                           histogram_freq=0,
                                           write_graph=False,
                                           write_images=False,
                                           update_freq='epoch',
                                           embeddings_freq=0,
                                           embeddings_metadata=None,
                                          ),
        ]
    return callback_list

def copy_weights_by_matching_layer_name(model_dest, model, verbose = False):
    """
        It copies all the weights of the layers sharing the same name 
        from model to model_dest
    """
    
    for layer_dest in model_dest.layers:
        for layer in model.layers:
            if layer_dest.name == layer.name:
                layer_dest.set_weights(layer.get_weights())
                if verbose==True:
                    print(layer_dest.name)
                pass
            pass
        pass
    return model_dest


def loupe_model(input_shape, R, dec, L = 0, depth = 5, ):
    """
        It returns the selected LOUPE between:
        dec0|L0 (original LOUPE)
        dec1|L1
        dec1|L2
        dec2|L0
        
        input_shape is the shape of the generic image the network 
        takes as input (e.g., (320, 320, 1))
        
        R is the speed-up factor
        
        To create deci*|Lj (for a general i and j) run the function 
        "model = add_Dykstra_projection_to_model(model)" 
        from "models.py" (see the demo for an example)
    """
    
    assert dec in [0, 1, 2], '"dec" must be "0, 1, 2"'
    if dec == 0:
        model = models.dec0(input_shape = input_shape,
                            R = R,
                            depth = depth,
                           )
    elif dec == 1:
        model = models.dec1(input_shape = input_shape,
                            R = R,
                            L = L,
                            depth = depth,
                          )
    elif dec == 2:
        model = models.dec2(input_shape = input_shape,
                            R = R,
                            depth = depth,
                           )
    
    model = set_slope_trainability(model, trainable = False)
    
    return model

############### MODIFY LAYER TRAINABILITY ###############

def set_slope_trainability(model, trainable = False, verbose = False):
    """
        sets the slope trainability to "trainable"
        of all the layers "sampled_mask" of the model
    """
    
    for i, l in enumerate(model.layers):
        if l.name=='sampled_mask':
            if verbose == True:
                print('\n\n\t\t SET SLOPE TRAINABILITY')
                print('layer n°: ', i,' - ', l.name, ' - trainability:')
                print('- before : ', l.trainable)
            l.set_attributes(trainable = trainable)
            if verbose == True:
                print('- after: ', l.trainable)
    return model

def set_neurons_trainability(model, trainable, verbose=False):
    """
        sets the trainability of the decoder
    """
    for i, l in enumerate(model.layers):
        if l.name.find('mask')==-1:
            if verbose==True:
                print('\n\n\t\tSET NEURONS TRAINABILITY')
                print('layer n°: ', i,' - ', l.name, ' - trainability:')
                print('- before : ', l.trainable)
            l.set_attributes(trainable = trainable)
            if verbose==True:
                print('- after: ', l.trainable)
    return model


def set_probMask_trainability(model, trainable, verbose=False):
    """
        sets the the trainability of the encoder
    """
    for i, l in enumerate(model.layers):
        if l.name.find('prob_mask')!=-1 and l.name.find('prob_mask_')==-1:
            if verbose==True:
                print('\n\n\t\tSET PROB MASK TRAINABILITY')
                print('layer n°: ', i,' - ', l.name, ' - trainability:')
                print('- before : ', l.trainable)
            l.set_attributes(trainable = trainable)
            if verbose==True:
                print('- after: ', l.trainable)
    return model


def set_mask_slope(model, slope, verbose=False):
    """
        updates the slope value 
        of all the layers "sampled_mask" of the model
    """
    for i, l in enumerate(model.layers):
        if l.name.find('sampled_mask')!=-1:
            if verbose==True:
                print('\n\n\t\SET MASK SLOPE')
                print('layer n°: ', i,' - ', l.name, ' - slope:')
                print('- before : ', l.slope)
            l.set_attributes(slope = slope)
            if verbose==True:
                print('- after: ', l.slope)
    return model

def set_mask_hard_threshold(model, hard_threshold, verbose=False):
    """
        makes the model binarize (or not) the mask before using it
    """
    for i, l in enumerate(model.layers):
        if l.name.find('undersample')!=-1:
            if verbose==True:
                print('\n\n\t\tSET MASK THRESHOLD')
                print('layer n°: ', i,' - ', l.name, ' - hard_threshold:')
                print('- before : ', l.hard_threshold)
            l.set_attributes(hard_threshold = hard_threshold)
            if verbose==True:
                print('- after: ', l.hard_threshold)
    return model

def set_mask_randomicity(model, randomicity = True, verbose=False):
    """
       Fixes (or unfixes) the mask of the model 
    """
    if randomicity==True:
        maxval = 1.0
        minval = 0.0
    else:
        maxval = 0.50000001
        minval = 0.49999999
        
        
    for i, l in enumerate(model.layers):
        if l.name.find('random_mask')!=-1:
            if verbose==True:
                print('\n\n\t\tSET MASK RANDOMICITY')
                print('layer n°: ', i,' - ', l.name, ' - randomicity:')
                print('- before : ', l.maxval, l.minval)
            l.set_attributes(maxval = maxval, minval = minval)
            if verbose==True:
                print('- after: ', l.maxval, l.minval)
    return model

def set_mask_R(model, R, verbose=False):
    """
        updates the speed-up factore of all the masks of the model
    """
    for i, l in enumerate(model.layers):
        if l.name.find('prob_mask_scaled')!=-1:
            if verbose==True:
                print('\n\n\t\tSET MASK R')
                print('layer n°: ', i,' - ', l.name, ' - R:')
                print('- before : ', l.R)
            l.set_attributes(R = R)
            if verbose==True:
                print('- after: ', l.R)
    return model

def write_probMask(model, probMask, verbose = False):
    """
        updates the probability mask
    """
    j = 0
    for i, l in enumerate(model.layers):
        if l.name.find('prob_mask')!=-1 and l.name.find('prob_mask_scaled')==-1:
            if verbose==True:
                print('layer n°: ', i,' - ', l.name)
            l.write_probMask(probMask[j])
            j = j + 1
            if verbose==True:
                print('mask has been written')
    return model

def read_probMask(model, verbose = False):
    probMask = []
    for i, l in enumerate(model.layers):
        if l.name.find('prob_mask')!=-1 and l.name.find('prob_mask_scaled')==-1:
            if verbose==True:
                print('layer n°: ', i,' - ', l.name)
            probMask += [l.read_probMask()]
            if verbose==True:
                print('mask has been read')
    return probMask

def change_setting(model, setting = 'test', verbose = False):
    
    """
        Switchs the selected model from "train" to "test" mode 
        (or viceversa) by binarizing and fixing the mask.
        To fix the mask, the random number generator layer
        is tuned to only generate deterministic numbers (0.5)
        To binarize the mask the threshold operation (>0.5) is added.
    """
    
    assert setting == 'test' or setting == 'train', 'setting must be "test" or "setting"'
    
    if setting == 'test':
        randomicity = False
        hard_threshold = True
    elif setting == 'train':
        randomicity = True
        hard_threshold = False
        
    model = set_mask_randomicity(model, randomicity = randomicity, 
                                 verbose = verbose)
    model = set_mask_hard_threshold(model, hard_threshold = hard_threshold,
                                    verbose=verbose)
    
    return model