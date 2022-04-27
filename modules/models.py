#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:50:13 2020

@author: filippomartinini
"""

# third party
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Activation, LeakyReLU, Permute
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import AveragePooling2D, Conv2D
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, Multiply
from tensorflow.keras.layers import BatchNormalization, Concatenate, Add
from tensorflow.keras.layers import Subtract, Dense
from tensorflow.keras.initializers import Initializer
from tensorflow import keras
from modules import layers
import copy


def dec2(input_shape,
         R,
         depth = 5,
        ):
    """
    loupe_model

    Parameters:
        input_shape: input shape
        filt: number of base filters
        kern: kernel size
        R: desired acceleration rate
        pmask_slope: slope of logistic parameter in probability mask
        sample_slope: slope of logistic parameter in mask realization
        hard_threshold: whether to use binary masks (only for inference)
        
    #Returns:
        #keras model

    UNet leaky two channel
    """
    
    inputs = Input(shape=input_shape, name='input')

    last_tensor = inputs
    
    # if necessary, concatenate with zeros for FFT
    if input_shape[-1] == 1:
        last_tensor = layers.ConcatenateZero(name='concat_zero')(last_tensor)
        input_shape = input_shape[:-1]+(2,)

    # input -> kspace via FFT
    last_tensor_Fx = layers.FFT(name='fft')(last_tensor)

    last_tensor_mask = _mask_from_tensor(last_tensor, R,)

    y = layers.UnderSampleHolistic(name='undersample')([last_tensor_Fx,
                                                        last_tensor_mask])
    
    last_tensor = layers.IFFT(name='ifft')(y)
        
    unet_tensor = _unet_from_tensor(last_tensor, depth = depth)

    last_tensor = Add(name='unet_output')([last_tensor, unet_tensor])
    
    last_tensor = layers.FFT(name='fft_projection')(last_tensor)
         
    last_tensor = layers.UnderSampleHolistic(
            complement = 1,                                 
            name='undersample_projection_complement',)([last_tensor,
                                                        last_tensor_mask])
    
    last_tensor = Add(name='add_projection')([last_tensor, y])
    
    last_tensor = layers.IFFT(name='ifft_projection')(last_tensor)
    
    last_tensor = layers.ComplexAbs(name='abs_projection')(last_tensor)
        
    return Model(inputs = inputs, outputs = last_tensor, name = 'dec2')


def dec1(input_shape,
         R,
         L = 2,
         depth = 5,
        ):
    """
    loupe_model

    Parameters:
        input_shape: input shape
        filt: number of base filters
        kern: kernel size
        R: desired acceleration rate
        pmask_slope: slope of logistic parameter in probability mask
        sample_slope: slope of logistic parameter in mask realization
        hard_threshold: whether to use binary masks (only for inference)
        
    #Returns:
        #keras model

    UNet leaky two channel
    """
    assert L==1 or L==2, 'only "L==1" or "L==2" are valid entries.'
    inputs = Input(shape=input_shape, name='input')

    last_tensor = inputs
    
    # if necessary, concatenate with zeros for FFT
    if input_shape[-1] == 1:
        last_tensor = layers.ConcatenateZero(name='concat_zero')(last_tensor)
        input_shape = input_shape[:-1]+(2,)

    # input -> kspace via FFT
    last_tensor_Fx = layers.FFT(name='fft')(last_tensor)

    last_tensor_mask = _mask_from_tensor(last_tensor, R,)
    
    # Under-sample with the mask or the binary version of the mask
    y = layers.UnderSampleHolistic(name='undersample')([last_tensor_Fx, 
                                                        last_tensor_mask])
                                   
    if L == 2:
        y_bar = layers.UnderSampleHolistic(complement = 1,
                                           name='undersample_complement',
                                          )([last_tensor_Fx, 
                                             last_tensor_mask])

    # IFFT if trainTIFT==False, TIFT if trainTIFT==True
    last_tensor = layers.IFFT(name='ifft')(y)
        
    # hard-coded UNet
    unet_tensor = _unet_from_tensor(last_tensor, depth = depth, )

    # final output from model 
    add_tensor = Add(name='unet_output')([last_tensor, unet_tensor])
    
    # complex absolute layer
    abs_tensor = layers.ComplexAbs(name='abs')(add_tensor)
    
    last_tensor = layers.FFT(name='fft_projection')(add_tensor)
    
    
    
    
    y_preojected_for_regularization = layers.UnderSampleHolistic(
            complement = L-1,                                                      
            name='undersample_projection')([last_tensor, last_tensor_mask])
    
    if L == 1:
        y_for_regularization = y
    elif L == 2:
        y_for_regularization = y_bar
        
    regularization = Subtract(name='projection')([
            y_preojected_for_regularization, 
            y_for_regularization])
    
    
    outputs = [abs_tensor, regularization]
    
    return Model(inputs = inputs, outputs = outputs, name = 'dec1-L'+str(L))

def dec0(input_shape,
         R,
         depth = 5,
        ):
    """
    loupe_model

    Parameters:
        input_shape: input shape
        filt: number of base filters
        kern: kernel size
        R: desired acceleration rate
        pmask_slope: slope of logistic parameter in probability mask
        sample_slope: slope of logistic parameter in mask realization
        hard_threshold: whether to use binary masks (only for inference)
        
    #Returns:
        #keras model

    UNet leaky two channel
    """
    inputs = Input(shape=input_shape, name='input')

    last_tensor = inputs
    
    # if necessary, concatenate with zeros for FFT
    if input_shape[-1] == 1:
        last_tensor = layers.ConcatenateZero(name='concat_zero')(last_tensor)
        input_shape = input_shape[:-1]+(2,)

    # input -> kspace via FFT
    last_tensor_Fx = layers.FFT(name='fft')(last_tensor)

    last_tensor_mask = _mask_from_tensor(last_tensor, R,)
    
    # Under-sample with the mask or the binary version of the mask
    y = layers.UnderSampleHolistic(name='undersample')([last_tensor_Fx, 
                                                        last_tensor_mask])
                                   
    # IFFT if trainTIFT==False, TIFT if trainTIFT==True
    last_tensor = layers.IFFT(name='ifft')(y)
        
    # hard-coded UNet
    unet_tensor = _unet_from_tensor(last_tensor, depth = depth, 
                                    output_nb_feats = 1)
    
    # complex absolute layer
    last_tensor = layers.ComplexAbs(name='abs')(last_tensor)
    
    # final output from model 
    last_tensor = Add(name='unet_output')([last_tensor, unet_tensor])
        
    return Model(inputs = inputs, outputs = last_tensor, name = 'dec0')


def _mask_from_tensor(last_tensor, R, pmask_slope=5, sample_slope=200,):
    
    """
    Mask used in LOUPE
    """
    
    # build probability mask
    prob_mask_tensor = layers.ProbMask(name='prob_mask',
                                       slope=pmask_slope)(last_tensor) 

    # probability mask rescaled to have mean=sparsity
    prob_mask_tensor_rescaled = layers.RescaleProbMap(R, 
                                                      name='prob_mask_scaled',
                                                     )(prob_mask_tensor)

    # Realization of random uniform mask
    thresh_tensor = layers.RandomMask(name='random_mask')(prob_mask_tensor)

    # Realization of mask
    last_tensor_mask = layers.ThresholdRandomMask(slope=sample_slope, 
                                                  name='sampled_mask',
                                                 )([prob_mask_tensor_rescaled,
                                                    thresh_tensor]) 

    return last_tensor_mask
        
def _unet_from_tensor(tensor, 
                      filt = 64, 
                      kern = 3, 
                      depth = 5, 
                      trainable = True, 
                      acti = None, 
                      output_nb_feats = 2, 
                      batch_norm_before_acti = True,
                      pool_size = (2, 2),
                     ):

    output_tensor = tensor
    tensor_list = []
    for i in np.arange(1, depth):
        tensor = basic_UNET_block(
                output_tensor, filt*(2**(i-1)),
                kern, acti, i, trainable, 
                batch_norm_before_acti = batch_norm_before_acti)

        output_tensor = AveragePooling2D(pool_size=pool_size,
                                         name = 'pool_'+str(i), )(tensor)

        tensor_list += [tensor]

    output_tensor = basic_UNET_block(
            output_tensor, filt*(2**(depth-1)), 
            kern, acti, depth, trainable, 
            batch_norm_before_acti=batch_norm_before_acti)

    tensor_list = tensor_list[::-1]

    for j, i in enumerate(np.arange(depth+1, 2*depth)):

        output_tensor = UpSampling2D(size=pool_size, name = 'up_'+str(j),
                                    )(output_tensor)

        output_tensor = Concatenate(axis=-1, name = 'concat_'+str(j), 
                                   )([output_tensor, tensor_list[j]])

        output_tensor = basic_UNET_block(
                output_tensor, 
                filt*(2**(depth-2-j)), kern, acti, i, trainable,
                batch_norm_before_acti=batch_norm_before_acti)

    output_tensor = Conv2D(output_nb_feats, 1, padding = 'same',
                           name='output_UNET', 
                           trainable = trainable)(output_tensor)

    return output_tensor


def basic_UNET_block(inp, filt, kern, acti, identifier,
                     trainable = True, 
                     batch_norm_before_acti=False):
            
    idf = str(identifier)

    conv = Conv2D(filt, kern, activation = acti, padding = 'same',
                  name='conv_'+idf+'_1', trainable = trainable)(inp)
    conv = batch_norm_and_relu(conv, idf, '1', 
                               batch_norm_before_acti=batch_norm_before_acti,)
    conv = Conv2D(filt, kern, activation = acti, padding = 'same',
                  name='conv_'+idf+'_2', trainable = trainable)(conv)
    conv = batch_norm_and_relu(conv, idf, '2',
                               batch_norm_before_acti=batch_norm_before_acti,)

    return conv
        
def batch_norm_and_relu(conv, idf, idf_2, batch_norm_before_acti=False,):
    ReLu = LeakyReLU(name = 'leaky_re_lu_'+idf+'_'+idf_2)

    if batch_norm_before_acti == False:
        conv = ReLu(conv)

    conv = BatchNormalization(name='batch_norm_'+idf+'_'+idf_2)(conv)

    if batch_norm_before_acti == True:
        conv = ReLu(conv)

    return conv

def add_Dykstra_projection_to_model(model, iterations = 15):
    
    y = model.get_layer('undersample').output
    
    mask = model.get_layer('sampled_mask').output
    
    last_tensor = model.outputs[0]
    
    for i in range(iterations):
        
        last_tensor = layers.ConcatenateZero(name='concat_zero_Dykstra'+str(i),
                                            )(last_tensor)         
        
        last_tensor = layers.FFT(name='fft_Dykstra-'+str(i))(last_tensor)
        
        last_tensor = layers.UnderSampleHolistic(
                complement = True, 
                 hard_threshold = True, 
                 name='undersample_Dykstra'+str(i),
                )([last_tensor, mask])
        
        last_tensor = Add(name='add_Dykstra-'+str(i))([last_tensor, y])
        
        last_tensor = layers.IFFT(name='ifft_Dykstra-'+str(i))(last_tensor)
        
        last_tensor = layers.ComplexAbs(name='abs_Dykstra-'+str(i))(last_tensor)
    
        last_tensor = layers.Clip(name='clip_Dykstra-'+str(i))(last_tensor)
    
    
    inputs = model.inputs
    outputs = last_tensor
    
    
    model_Dykstra = Model(inputs = inputs, outputs = outputs, 
                          name = model.name+'-Dykstra')
    return model_Dykstra
    

def model_Fourier(input_shape, mode = 'direct'):
    
    assert mode == 'direct' or model == 'inverse', '"mode" is "direct" or "inverse"'
    
    input_x = Input(shape=input_shape, name='input_x')
    
    last_tensor = input_x
    if input_shape[-1] == 1:
        last_tensor = layers.ConcatenateZero(name='concat_zero',
                                            )(last_tensor)
        pass
    
    if mode == 'inverse':
        last_tensor = layers.FFT()(last_tensor)
    else:
        last_tensor = layers.IFFT()(last_tensor)
        
    inputs = [input_x]
    outputs = [last_tensor]
    modelFourier = Model(inputs=inputs,outputs=outputs, 
                         name = mode+'_fast_fourier_transform')
    return modelFourier

def encoder(model, ):
    inputs = model.inputs
    last_tensor = model.get_layer('ifft').output
    last_tensor = layers.ComplexAbs(name='abs')(last_tensor)
    encoder = Model(inputs = inputs, outputs = last_tensor)
    return encoder