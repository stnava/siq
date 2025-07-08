
from pathlib import Path
from pathlib import PurePath
import os
import pandas as pd
import math
import os.path
from os import path
from os.path import exists
import pickle
import sys
import numpy as np
import random
import functools
from operator import mul
from scipy.sparse.linalg import svds
from scipy.stats import pearsonr
import re

import ants
import antspynet
import antspyt1w
import tensorflow as tf
from tensorflow.python.eager.context import eager_mode, graph_mode

from multiprocessing import Pool

DATA_PATH = os.path.expanduser('~/.siq/')

def get_data( name=None, force_download=False, version=0, target_extension='.csv' ):
    """
    Get SIQ data filename

    The first time this is called, it will download data to ~/.siq.
    After, it will just read data from disk.  The ~/.siq may need to
    be periodically deleted in order to ensure data is current.

    Arguments
    ---------
    name : string
        name of data tag to retrieve
        Options:
            - 'all'

    force_download: boolean

    version: version of data to download (integer)

    Returns
    -------
    string
        filepath of selected data

    Example
    -------
    >>> import siq
    >>> siq.get_data()
    """
    os.makedirs(DATA_PATH, exist_ok=True)

    def download_data( version ):
        url = "https://figshare.com/ndownloader/articles/16912366/versions/" + str(version)
        target_file_name = "16912366.zip"
        target_file_name_path = tf.keras.utils.get_file(target_file_name, url,
            cache_subdir=DATA_PATH, extract = True )
        os.remove( DATA_PATH + target_file_name )

    if force_download:
        download_data( version = version )


    files = []
    for fname in os.listdir(DATA_PATH):
        if ( fname.endswith(target_extension) ) :
            fname = os.path.join(DATA_PATH, fname)
            files.append(fname)

    if len( files ) == 0 :
        download_data( version = version )
        for fname in os.listdir(DATA_PATH):
            if ( fname.endswith(target_extension) ) :
                fname = os.path.join(DATA_PATH, fname)
                files.append(fname)

    if name == 'all':
        return files

    datapath = None

    for fname in os.listdir(DATA_PATH):
        mystem = (Path(fname).resolve().stem)
        mystem = (Path(mystem).resolve().stem)
        mystem = (Path(mystem).resolve().stem)
        if ( name == mystem and fname.endswith(target_extension) ) :
            datapath = os.path.join(DATA_PATH, fname)

    return datapath



from keras.models import Model
from keras.layers import (Input, Add, Subtract,
                          PReLU, Concatenate,
                          UpSampling2D, UpSampling3D,
                          Conv2D, Conv2DTranspose,
                          Conv3D, Conv3DTranspose)


# define the DBPN network - this uses a model definition that is general to<br>
# both 2D and 3D. recommended parameters for different upsampling rates can<br>
# be found in the papers by Haris et al.  We make one significant change to<br>
# the original architecture by allowing standard interpolation for upsampling<br>
# instead of convolutional upsampling.  this is controlled by the interpolation<br>
# option.



def dbpn(input_image_size,
                                                 number_of_outputs=1,
                                                 number_of_base_filters=64,
                                                 number_of_feature_filters=256,
                                                 number_of_back_projection_stages=7,
                                                 convolution_kernel_size=(12, 12),
                                                 strides=(8, 8),
                                                 last_convolution=(3, 3),
                                                 number_of_loss_functions=1,
                                                 interpolation = 'nearest'
                                                ):
    idim = len( input_image_size ) - 1
    if idim == 2:
        myconv = Conv2D
        myconv_transpose = Conv2DTranspose
        myupsampling = UpSampling2D
        shax = ( 1, 2 )
        firstConv = (3,3)
        firstStrides=(1,1)
        smashConv=(1,1)
    if idim == 3:
        myconv = Conv3D
        myconv_transpose = Conv3DTranspose
        myupsampling = UpSampling3D
        shax = ( 1, 2, 3 )
        firstConv = (3,3,3)
        firstStrides=(1,1,1)
        smashConv=(1,1,1)
    def up_block_2d(L, number_of_filters=64, kernel_size=(12, 12), strides=(8, 8),
                    include_dense_convolution_layer=True):
        if include_dense_convolution_layer == True:
            L = myconv(filters = number_of_filters,
                       use_bias=True,
                       kernel_size=smashConv,
                       strides=firstStrides,
                       padding='same')(L)
            L = PReLU(alpha_initializer='zero',
                      shared_axes=shax)(L)
        # Scale up
        if idim == 2:
            H0 = myupsampling( size = strides, interpolation=interpolation )(L)
        if idim == 3:
            H0 = myupsampling( size = strides )(L)
        H0 = myconv(filters=number_of_filters,
                    kernel_size=firstConv,
                    strides=firstStrides,
                    use_bias=True,
                    padding='same')(H0)
        H0 = PReLU(alpha_initializer='zero',
                   shared_axes=shax)(H0)
        # Scale down
        L0 = myconv(filters=number_of_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    kernel_initializer='glorot_uniform',
                    padding='same')(H0)
        L0 = PReLU(alpha_initializer='zero',
                   shared_axes=shax)(L0)
        # Residual
        E = Subtract()([L0, L])
        # Scale residual up
        if idim == 2:
            H1 = myupsampling( size = strides, interpolation=interpolation  )(E)
        if idim == 3:
            H1 = myupsampling( size = strides )(E)
        H1 = myconv(filters=number_of_filters,
                    kernel_size=firstConv,
                    strides=firstStrides,
                    use_bias=True,
                    padding='same')(H1)
        H1 = PReLU(alpha_initializer='zero',
                   shared_axes=shax)(H1)
        # Output feature map
        up_block = Add()([H0, H1])
        return up_block
    def down_block_2d(H, number_of_filters=64, kernel_size=(12, 12), strides=(8, 8),
                    include_dense_convolution_layer=True):
        if include_dense_convolution_layer == True:
            H = myconv(filters = number_of_filters,
                       use_bias=True,
                       kernel_size=smashConv,
                       strides=firstStrides,
                       padding='same')(H)
            H = PReLU(alpha_initializer='zero',
                      shared_axes=shax)(H)
        # Scale down
        L0 = myconv(filters=number_of_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    kernel_initializer='glorot_uniform',
                    padding='same')(H)
        L0 = PReLU(alpha_initializer='zero',
                   shared_axes=shax)(L0)
        # Scale up
        if idim == 2:
            H0 = myupsampling( size = strides, interpolation=interpolation )(L0)
        if idim == 3:
            H0 = myupsampling( size = strides )(L0)
        H0 = myconv(filters=number_of_filters,
                    kernel_size=firstConv,
                    strides=firstStrides,
                    use_bias=True,
                    padding='same')(H0)
        H0 = PReLU(alpha_initializer='zero',
                   shared_axes=shax)(H0)
        # Residual
        E = Subtract()([H0, H])
        # Scale residual down
        L1 = myconv(filters=number_of_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    kernel_initializer='glorot_uniform',
                    padding='same')(E)
        L1 = PReLU(alpha_initializer='zero',
                   shared_axes=shax)(L1)
        # Output feature map
        down_block = Add()([L0, L1])
        return down_block
    inputs = Input(shape=input_image_size)
    # Initial feature extraction
    model = myconv(filters=number_of_feature_filters,
                   kernel_size=firstConv,
                   strides=firstStrides,
                   padding='same',
                   kernel_initializer='glorot_uniform')(inputs)
    model = PReLU(alpha_initializer='zero',
                  shared_axes=shax)(model)
    # Feature smashing
    model = myconv(filters=number_of_base_filters,
                   kernel_size=smashConv,
                   strides=firstStrides,
                   padding='same',
                   kernel_initializer='glorot_uniform')(model)
    model = PReLU(alpha_initializer='zero',
                  shared_axes=shax)(model)
    # Back projection
    up_projection_blocks = []
    down_projection_blocks = []
    model = up_block_2d(model, number_of_filters=number_of_base_filters,
      kernel_size=convolution_kernel_size, strides=strides)
    up_projection_blocks.append(model)
    for i in range(number_of_back_projection_stages):
        if i == 0:
            model = down_block_2d(model, number_of_filters=number_of_base_filters,
              kernel_size=convolution_kernel_size, strides=strides)
            down_projection_blocks.append(model)
            model = up_block_2d(model, number_of_filters=number_of_base_filters,
              kernel_size=convolution_kernel_size, strides=strides)
            up_projection_blocks.append(model)
            model = Concatenate()(up_projection_blocks)
        else:
            model = down_block_2d(model, number_of_filters=number_of_base_filters,
              kernel_size=convolution_kernel_size, strides=strides,
              include_dense_convolution_layer=True)
            down_projection_blocks.append(model)
            model = Concatenate()(down_projection_blocks)
            model = up_block_2d(model, number_of_filters=number_of_base_filters,
              kernel_size=convolution_kernel_size, strides=strides,
              include_dense_convolution_layer=True)
            up_projection_blocks.append(model)
            model = Concatenate()(up_projection_blocks)
    outputs = myconv(filters=number_of_outputs,
                     kernel_size=last_convolution,
                     strides=firstStrides,
                     padding = 'same',
                     kernel_initializer = "glorot_uniform")(model)
    if number_of_loss_functions == 1:
        deep_back_projection_network_model = Model(inputs=inputs, outputs=outputs)
    else:
        outputList=[]
        for k in range(number_of_loss_functions):
            outputList.append(outputs)
        deep_back_projection_network_model = Model(inputs=inputs, outputs=outputList)
    return deep_back_projection_network_model


# generate a random corner index for a patch

def get_random_base_ind( full_dims, patchWidth, off=8 ):
    baseInd = [None,None,None]
    for k in range(3):
        baseInd[k]=random.sample( range( off, full_dims[k]-1-patchWidth[k] ), 1 )[0]
    return baseInd


# extract a random patch
def get_random_patch( img, patchWidth ):
    mystd = 0
    while mystd == 0:
        inds = get_random_base_ind( full_dims = img.shape, patchWidth=patchWidth, off=8 )
        hinds = [None,None,None]
        for k in range(len(inds)):
            hinds[k] = inds[k] + patchWidth[k]
        myimg = ants.crop_indices( img, inds, hinds )
        mystd = myimg.std()
    return myimg

def get_random_patch_pair( img, img2, patchWidth ):
    mystd = mystd2 = 0
    ct = 0
    while mystd == 0 or mystd2 == 0:
        inds = get_random_base_ind( full_dims = img.shape, patchWidth=patchWidth, off=8  )
        hinds = [None,None,None]
        for k in range(len(inds)):
            hinds[k] = inds[k] + patchWidth[k]
        myimg = ants.crop_indices( img, inds, hinds )
        myimg2 = ants.crop_indices( img2, inds, hinds )
        mystd = myimg.std()
        mystd2 = myimg2.std()
        ct = ct + 1
        if ( ct > 20 ):
            return myimg, myimg2
    return myimg, myimg2

def pseudo_3d_vgg_features( inshape = [128,128,128], layer = 4, angle=0, pretrained=True, verbose=False ):
    def getLayerScaleFactorForTransferLearning( k, w3d, w2d ):
        myfact = np.round( np.prod( w3d[k].shape ) / np.prod(  w2d[k].shape) )
        return myfact
    vgg19 = tf.keras.applications.VGG19(
            include_top = False, weights = "imagenet",
            input_shape = [inshape[0],inshape[1],3],
            classes = 1000 )
    def findLayerIndex( layerName, mdl ):
          for k in range( len( mdl.layers ) ):
            if layerName == mdl.layers[k].name :
                return k - 1
          return None
    layer_index = layer-1 # findLayerIndex( 'block2_conv2', vgg19 )
    vggmodelRaw = antspynet.create_vgg_model_3d(
            [inshape[0],inshape[1],inshape[2],1],
            number_of_classification_labels = 1000,
            layers = [1, 2, 3, 4, 4],
            lowest_resolution = 64,
            convolution_kernel_size= (3, 3, 3), pool_size = (2, 2, 2),
            strides = (2, 2, 2), number_of_dense_units= 4096, dropout_rate = 0,
            style = 19, mode = "classification")
    if verbose:
        print( vggmodelRaw.layers[layer_index] )
        print( vggmodelRaw.layers[layer_index].name )
        print( vgg19.layers[layer_index] )
        print( vgg19.layers[layer_index].name )
    feature_extractor_2d = tf.keras.Model(
            inputs = vgg19.input,
            outputs = vgg19.layers[layer_index].output)
    feature_extractor = tf.keras.Model(
            inputs = vggmodelRaw.input,
            outputs = vggmodelRaw.layers[layer_index].output)
    wts_2d = feature_extractor_2d.weights
    wts = feature_extractor.weights
    def checkwtshape( a, b ):
        if len(a.shape) != len(b.shape):
                return False
        for j in range(len(a.shape)):
            if a.shape[j] != b.shape[j]:
                return False
        return True
    for ww in range(len(wts)):
        wts[ww]=wts[ww].numpy()
        wts_2d[ww]=wts_2d[ww].numpy()
        if checkwtshape( wts[ww], wts_2d[ww] ) and ww != 0:
            wts[ww]=wts_2d[ww]
        elif ww != 0:
            # FIXME - should allow doing this across different angles
            if angle == 0:
                wts[ww][:,:,0,:,:]=wts_2d[ww]/3.0
                wts[ww][:,:,1,:,:]=wts_2d[ww]/3.0
                wts[ww][:,:,2,:,:]=wts_2d[ww]/3.0
            if angle == 1:
                wts[ww][:,0,:,:,:]=wts_2d[ww]/3.0
                wts[ww][:,1,:,:,:]=wts_2d[ww]/3.0
                wts[ww][:,2,:,:,:]=wts_2d[ww]/3.0
            if angle == 2:
                wts[ww][0,:,:,:,:]=wts_2d[ww]/3.0
                wts[ww][1,:,:,:,:]=wts_2d[ww]/3.0
                wts[ww][2,:,:,:,:]=wts_2d[ww]/3.0
        else:
            wts[ww][:,:,:,0,:]=wts_2d[ww]
    if pretrained:
        feature_extractor.set_weights( wts )
        newinput = tf.keras.layers.Rescaling(  255.0, -127.5  )( feature_extractor.input )
        feature_extractor2 = feature_extractor( newinput )
        feature_extractor = tf.keras.Model( feature_extractor.input, feature_extractor2 )
    return feature_extractor

def pseudo_3d_vgg_features_unbiased( inshape = [128,128,128], layer = 4, verbose=False ):
    f = [
        pseudo_3d_vgg_features( inshape, layer, angle=0, pretrained=True, verbose=verbose ),
        pseudo_3d_vgg_features( inshape, layer, angle=1, pretrained=True ),
        pseudo_3d_vgg_features( inshape, layer, angle=2, pretrained=True ) ]
    f1=f[0].inputs
    f0o=f[0]( f1 )
    f1o=f[1]( f1 )
    f2o=f[2]( f1 )
    catter = tf.keras.layers.concatenate( [f0o, f1o, f2o ])
    feature_extractor = tf.keras.Model( f1, catter )
    return feature_extractor

def get_grader_feature_network( layer=6 ):
# perceptual features - one can explore different layers and features
# these layers - or combinations of these - are commonly used in the literature
# as feature spaces for loss functions.  weighting terms relative to MSE are
# also given in several papers which can help guide parameter setting.
    grader = antspynet.create_resnet_model_3d(
        [None,None,None,1],
        lowest_resolution = 32,
        number_of_classification_labels = 4,
        cardinality = 1,
        squeeze_and_excite = False )
    # the folder and data below as available from antspyt1w get_data
    graderfn = os.path.expanduser( "~/.antspyt1w/resnet_grader.h5" )
    if not exists( graderfn ):
        raise Exception("graderfn " + graderfn + " does not exist")
    grader.load_weights( graderfn)
    #    feature_extractor_23 = tf.keras.Model( inputs=grader.inputs, outputs=grader.layers[23].output )
    #   feature_extractor_44 = tf.keras.Model( inputs=grader.inputs, outputs=grader.layers[44].output )
    return tf.keras.Model( inputs=grader.inputs, outputs=grader.layers[layer].output )


def default_dbpn(
    strider, # length should equal dimensionality
    dimensionality = 3,
    nfilt=64,
    nff = 256,
    convn = 6,
    lastconv = 3,
    nbp=7,
    nChannelsIn=1,
    nChannelsOut=1,
    option = None,
    intensity_model=None,
    segmentation_model=None,
    sigmoid_second_channel=False,
    clone_intensity_to_segmentation=False,
    pro_seg = 0,
    freeze = False,
    verbose=False
 ):
    """
    Constructs a DBPN model based on input parameters, and can optionally
    use external models for intensity or segmentation processing.

    Args:
        strider (list): List of strides, length must match `dimensionality`.
        dimensionality (int): Number of dimensions (2 or 3). Default is 3.
        nfilt (int): Number of base filters. Default is 64.
        nff (int): Number of feature filters. Default is 256.
        convn (int): Convolution kernel size. Default is 6.
        lastconv (int): Size of the last convolution. Default is 3.
        nbp (int): Number of back projection stages. Default is 7.
        nChannelsIn (int): Number of input channels. Default is 1.
        nChannelsOut (int): Number of output channels. Default is 1.
        option (str): Model size option ('tiny', 'small', 'medium', 'large'). Default is None.
        intensity_model (tf.keras.Model): Optional external intensity model.
        segmentation_model (tf.keras.Model): Optional external segmentation model.
        sigmoid_second_channel (bool): If True, applies sigmoid to second channel in output.
        clone_intensity_to_segmentation (bool): If True, clones intensity model weights to segmentation.
        pro_seg (int): If greater than 0, adds a segmentation arm.
        freeze (bool): If True, freezes the layers of the intensity/segmentation models.
        verbose (bool): If True, prints detailed logs.

    Returns:
        Model: A Keras model based on the specified configuration.

    Raises:
        Exception: If `len(strider)` is not equal to `dimensionality`.
    """
    if option == 'tiny':
        nfilt=32
        nff = 64
        convn = 3
        lastconv = 1
        nbp=2
    elif option == 'small':
        nfilt=32
        nff = 64
        convn = 6
        lastconv = 3
        nbp=4
    elif option == 'medium':
        nfilt=64
        nff = 128
        convn = 6
        lastconv = 3
        nbp=4
    else:
        option='large'
    if verbose:
        print("Building mode of size: " + option)
        if intensity_model is not None:
            print("user-passed intensity model will be frozen - only segmentation will train")
        if segmentation_model is not None:
            print("user-passed segmentation model will be frozen - only intensity will train")

    if len(strider) != dimensionality:
        raise Exception("len(strider) != dimensionality")
    # **model instantiation**: these are close to defaults for the 2x network.<br>
    # empirical evidence suggests that making covolutions and strides evenly<br>
    # divisible by each other reduces artifacts.  2*3=6.
    # ofn='./models/dsr3d_'+str(strider)+'up_' + str(nfilt) + '_' + str( nff ) + '_' + str(convn)+ '_' + str(lastconv)+ '_' + str(os.environ['CUDA_VISIBLE_DEVICES'])+'_v0.0.h5'
    if dimensionality == 2 :
        mdl = dbpn( (None,None,nChannelsIn),
            number_of_outputs=nChannelsOut,
            number_of_base_filters=nfilt,
            number_of_feature_filters=nff,
            number_of_back_projection_stages=nbp,
            convolution_kernel_size=(convn, convn),
            strides=(strider[0], strider[1]),
            last_convolution=(lastconv, lastconv),
            number_of_loss_functions=1,
            interpolation='nearest')
    if dimensionality == 3 :
        mdl = dbpn( (None,None,None,nChannelsIn),
            number_of_outputs=nChannelsOut,
            number_of_base_filters=nfilt,
            number_of_feature_filters=nff,
            number_of_back_projection_stages=nbp,
            convolution_kernel_size=(convn, convn, convn),
            strides=(strider[0], strider[1], strider[2]),
            last_convolution=(lastconv, lastconv, lastconv), number_of_loss_functions=1, interpolation='nearest')
    if sigmoid_second_channel and pro_seg != 0 :
        if dimensionality == 2 :
            input_image_size = (None,None,2)
            if intensity_model is None:
                intensity_model = dbpn( (None,None,1),
                    number_of_outputs=1,
                    number_of_base_filters=nfilt,
                    number_of_feature_filters=nff,
                    number_of_back_projection_stages=nbp,
                    convolution_kernel_size=(convn, convn),
                    strides=(strider[0], strider[1]),
                    last_convolution=(lastconv, lastconv),
                    number_of_loss_functions=1,
                    interpolation='nearest')
            else:
                if freeze:
                    for layer in intensity_model.layers:
                        layer.trainable = False
            if segmentation_model is None:
                segmentation_model = dbpn( (None,None,1),
                        number_of_outputs=1,
                        number_of_base_filters=nfilt,
                        number_of_feature_filters=nff,
                        number_of_back_projection_stages=nbp,
                        convolution_kernel_size=(convn, convn),
                        strides=(strider[0], strider[1]),
                        last_convolution=(lastconv, lastconv),
                        number_of_loss_functions=1, interpolation='linear')
            else:
                if freeze:
                    for layer in segmentation_model.layers:
                        layer.trainable = False
        if dimensionality == 3 :
            input_image_size = (None,None,None,2)
            if intensity_model is None:
                intensity_model = dbpn( (None,None,None,1),
                    number_of_outputs=1,
                    number_of_base_filters=nfilt,
                    number_of_feature_filters=nff,
                    number_of_back_projection_stages=nbp,
                    convolution_kernel_size=(convn, convn, convn),
                    strides=(strider[0], strider[1], strider[2]),
                    last_convolution=(lastconv, lastconv, lastconv),
                    number_of_loss_functions=1, interpolation='nearest')
            else:
                if freeze:
                    for layer in intensity_model.layers:
                        layer.trainable = False
            if segmentation_model is None:
                segmentation_model = dbpn( (None,None,None,1),
                        number_of_outputs=1,
                        number_of_base_filters=nfilt,
                        number_of_feature_filters=nff,
                        number_of_back_projection_stages=nbp,
                        convolution_kernel_size=(convn, convn, convn),
                        strides=(strider[0], strider[1], strider[2]),
                        last_convolution=(lastconv, lastconv, lastconv),
                        number_of_loss_functions=1, interpolation='linear')
            else:
                if freeze:
                    for layer in segmentation_model.layers:
                        layer.trainable = False
        if verbose:
            print( "len intensity_model layers : " + str( len( intensity_model.layers )))
            print( "len intensity_model weights : " + str( len( intensity_model.weights )))
            print( "len segmentation_model layers : " + str( len( segmentation_model.layers )))
            print( "len segmentation_model weights : " + str( len( segmentation_model.weights )))
        if clone_intensity_to_segmentation:
            for k in range(len( segmentation_model.weights )):
                if k < len( intensity_model.weights ):
                    if intensity_model.weights[k].shape == segmentation_model.weights[k].shape:
                        segmentation_model.weights[k] = intensity_model.weights[k]
        inputs = tf.keras.Input(shape=input_image_size)
        insplit = tf.split( inputs, 2, dimensionality+1)
        outputs = [
            intensity_model( insplit[0] ),
            tf.nn.sigmoid( segmentation_model( insplit[1] ) ) ]
        mdlout = tf.concat( outputs, axis=dimensionality+1 )
        return Model(inputs=inputs, outputs=mdlout )
    if pro_seg > 0 and intensity_model is not None:
        if verbose and freeze:
            print("Add a segmentation arm to the end. freeze intensity. intensity_model(seg) => conv => sigmoid")
        if verbose and not freeze:
            print("Add a segmentation arm to the end. freeze intensity. intensity_model(seg) => conv => sigmoid")
        if freeze:
            for layer in intensity_model.layers:
                layer.trainable = False
        if dimensionality == 2 :
            input_image_size = (None,None,2)
        elif dimensionality == 3 :
            input_image_size = (None, None,None,2)
        if dimensionality == 2:
            myconv = Conv2D
            firstConv = (convn,convn)
            firstStrides=(1,1)
            smashConv=(pro_seg,pro_seg)
        if dimensionality == 3:
            myconv = Conv3D
            firstConv = (convn,convn,convn)
            firstStrides=(1,1,1)
            smashConv=(pro_seg,pro_seg,pro_seg)
        inputs = tf.keras.Input(shape=input_image_size)
        insplit = tf.split( inputs, 2, dimensionality+1)
        # define segmentation arm
        seggit = intensity_model( insplit[1] )
        L0 = myconv(filters=nff,
                    kernel_size=firstConv,
                    strides=firstStrides,
                    kernel_initializer='glorot_uniform',
                    padding='same')(seggit)
        L1 = myconv(filters=nff,
                    kernel_size=firstConv,
                    strides=firstStrides,
                    kernel_initializer='glorot_uniform',
                    padding='same')(L0)
        L2 = myconv(filters=1,
                    kernel_size=smashConv,
                    strides=firstStrides,
                    kernel_initializer='glorot_uniform',
                    padding='same')(L1)
        outputs = [
            intensity_model( insplit[0] ),
            tf.nn.sigmoid( L2 ) ]
        mdlout = tf.concat( outputs, axis=dimensionality+1 )
        return Model(inputs=inputs, outputs=mdlout )
    return mdl

def image_patch_training_data_from_filenames(
    filenames,
    target_patch_size,
    target_patch_size_low,
    nPatches = 128,
    istest   = False,
    patch_scaler=True,
    to_tensorflow = False,
    verbose = False
    ):
    # **data generation**<br>
    # recent versions of tensorflow/keras allow data generators to be passed<br>
    # directly to the fit function.  underneath, this does a fairly efficient split<br>
    # between GPU and CPU usage and data transfer.<br>
    # this generator randomly chooses between linear and nearest neighbor downsampling.<br>
    # the *patch_scale* option can also be seen here which impacts how the network<br>
    # sees/learns from image intensity.
    if verbose:
        print("begin image_patch_training_data_from_filenames")
    tardim = len( target_patch_size )
    strider = []
    for j in range( tardim ):
        strider.append( np.round( target_patch_size[j]/target_patch_size_low[j]) )
    if tardim == 3:
        shaperhi = (nPatches,target_patch_size[0],target_patch_size[1],target_patch_size[2],1)
        shaperlo = (nPatches,target_patch_size_low[0],target_patch_size_low[1],target_patch_size_low[2],1)
    if tardim == 2:
        shaperhi = (nPatches,target_patch_size[0],target_patch_size[1],1)
        shaperlo = (nPatches,target_patch_size_low[0],target_patch_size_low[1],1)
    patchesOrig = np.zeros(shape=shaperhi)
    patchesResam = np.zeros(shape=shaperlo)
    patchesUp = None
    if istest:
        patchesUp = np.zeros(shape=patchesOrig.shape)
    for myn in range(nPatches):
            if verbose:
                print(myn)
            imgfn = random.sample( filenames, 1 )[0]
            if verbose:
                print(imgfn)
            img = ants.image_read( imgfn ).iMath("Normalize")
            if img.components > 1:
                img = ants.split_channels(img)[0]
            img = ants.crop_image( img, ants.threshold_image( img, 0.05, 1 ) )
            ants.set_origin( img, ants.get_center_of_mass(img) )
            img = ants.iMath(img,"Normalize")
            spc = ants.get_spacing( img )
            newspc = []
            for jj in range(len(spc)):
                newspc.append(spc[jj]*strider[jj])
            interp_type = random.choice( [0,1] )
            if True:
                imgp = get_random_patch( img, target_patch_size )
                imgpmin = imgp.min()
                if patch_scaler:
                    imgp = imgp - imgpmin
                    imgpmax = imgp.max()
                    if imgpmax > 0 :
                        imgp = imgp / imgpmax
                rimgp = ants.resample_image( imgp, newspc, use_voxels = False, interp_type=interp_type  )
                if istest:
                    rimgbi = ants.resample_image( rimgp, spc, use_voxels = False, interp_type=0  )
                if tardim == 3:
                    patchesOrig[myn,:,:,:,0] = imgp.numpy()
                    patchesResam[myn,:,:,:,0] = rimgp.numpy()
                    if istest:
                        patchesUp[myn,:,:,:,0] = rimgbi.numpy()
                if tardim == 2:
                    patchesOrig[myn,:,:,0] = imgp.numpy()
                    patchesResam[myn,:,:,0] = rimgp.numpy()
                    if istest:
                        patchesUp[myn,:,:,0] = rimgbi.numpy()
    if to_tensorflow:
        patchesOrig = tf.cast( patchesOrig, "float32")
        patchesResam = tf.cast( patchesResam, "float32")
    if istest:
        if to_tensorflow:
            patchesUp = tf.cast( patchesUp, "float32")
    return patchesResam, patchesOrig, patchesUp


def seg_patch_training_data_from_filenames(
    filenames,
    target_patch_size,
    target_patch_size_low,
    nPatches = 128,
    istest   = False,
    patch_scaler=True,
    to_tensorflow = False,
    verbose = False
    ):
    if verbose:
        print("begin seg_patch_training_data_from_filenames")
    tardim = len( target_patch_size )
    strider = []
    nchan = 2
    for j in range( tardim ):
        strider.append( np.round( target_patch_size[j]/target_patch_size_low[j]) )
    if tardim == 3:
        shaperhi = (nPatches,target_patch_size[0],target_patch_size[1],target_patch_size[2],nchan)
        shaperlo = (nPatches,target_patch_size_low[0],target_patch_size_low[1],target_patch_size_low[2],nchan)
    if tardim == 2:
        shaperhi = (nPatches,target_patch_size[0],target_patch_size[1],nchan)
        shaperlo = (nPatches,target_patch_size_low[0],target_patch_size_low[1],nchan)
    patchesOrig = np.zeros(shape=shaperhi)
    patchesResam = np.zeros(shape=shaperlo)
    patchesUp = None
    if istest:
        patchesUp = np.zeros(shape=patchesOrig.shape)
    for myn in range(nPatches):
            if verbose:
                print(myn)
            imgfn = random.sample( filenames, 1 )[0]
            if verbose:
                print(imgfn)
            img = ants.image_read( imgfn ).iMath("Normalize")
            if img.components > 1:
                img = ants.split_channels(img)[0]
            img = ants.crop_image( img, ants.threshold_image( img, 0.05, 1 ) )
            ants.set_origin( img, ants.get_center_of_mass(img) )
            img = ants.iMath(img,"Normalize")
            spc = ants.get_spacing( img )
            newspc = []
            for jj in range(len(spc)):
                newspc.append(spc[jj]*strider[jj])
            interp_type = random.choice( [0,1] )
            seg_class = random.choice( [1,2] )
            if True:
                imgp = get_random_patch( img, target_patch_size )
                imgpmin = imgp.min()
                if patch_scaler:
                    imgp = imgp - imgpmin
                    imgpmax = imgp.max()
                    if imgpmax > 0 :
                        imgp = imgp / imgpmax
                segp = ants.threshold_image( imgp, "Otsu", 2 ).threshold_image( seg_class, seg_class )
                rimgp = ants.resample_image( imgp, newspc, use_voxels = False, interp_type=interp_type  )
                rsegp = ants.resample_image( segp, newspc, use_voxels = False, interp_type=interp_type  )
                if istest:
                    rimgbi = ants.resample_image( rimgp, spc, use_voxels = False, interp_type=0  )
                if tardim == 3:
                    patchesOrig[myn,:,:,:,0] = imgp.numpy()
                    patchesResam[myn,:,:,:,0] = rimgp.numpy()
                    patchesOrig[myn,:,:,:,1] = segp.numpy()
                    patchesResam[myn,:,:,:,1] = rsegp.numpy()
                    if istest:
                        patchesUp[myn,:,:,:,0] = rimgbi.numpy()
                if tardim == 2:
                    patchesOrig[myn,:,:,0] = imgp.numpy()
                    patchesResam[myn,:,:,0] = rimgp.numpy()
                    patchesOrig[myn,:,:,1] = segp.numpy()
                    patchesResam[myn,:,:,1] = rsegp.numpy()
                    if istest:
                        patchesUp[myn,:,:,0] = rimgbi.numpy()
    if to_tensorflow:
        patchesOrig = tf.cast( patchesOrig, "float32")
        patchesResam = tf.cast( patchesResam, "float32")
    if istest:
        if to_tensorflow:
            patchesUp = tf.cast( patchesUp, "float32")
    return patchesResam, patchesOrig, patchesUp

def read( filename ):
    import re
    isnpy = len( re.sub( ".npy", "", filename ) ) != len( filename )
    if not isnpy:
        myoutput = ants.image_read( filename )
    else:
        myoutput = np.load( filename )
    return myoutput

def auto_weight_loss( mdl, feature_extractor, x, y, feature=2.0, tv=0.1, verbose=True ):
    y_pred = mdl( x )
    squared_difference = tf.square( y - y_pred)
    if len( y.shape ) == 5:
            tdim = 3
            myax = [1,2,3,4]
    if len( y.shape ) == 4:
            tdim = 2
            myax = [1,2,3]
    msqTerm = tf.reduce_mean(squared_difference, axis=myax)
    temp1 = feature_extractor(y)
    temp2 = feature_extractor(y_pred)
    feature_difference = tf.square(temp1-temp2)
    featureTerm = tf.reduce_mean(feature_difference, axis=myax)
    msqw = 10.0
    featw = feature * msqw * msqTerm / featureTerm
    mytv = tf.cast( 0.0, 'float32')
    if tdim == 3:
        for k in range( y_pred.shape[0] ): # BUG not sure why myr fails .... might be old TF version
            sqzd = y_pred[k,:,:,:,:]
            mytv = mytv + tf.reduce_mean( tf.image.total_variation( sqzd ) )
    if tdim == 2:
        mytv = tf.reduce_mean( tf.image.total_variation( y_pred ) )
    tvw = tv * msqw * msqTerm / mytv
    if verbose :
        print( "MSQ: " + str( msqw * msqTerm ) )
        print( "Feat: " + str( featw * featureTerm ) )
        print( "Tv: " + str(  mytv * tvw ) )
    wts = [msqw,featw.numpy().mean(),tvw.numpy().mean()]
    return wts

def auto_weight_loss_seg( mdl, feature_extractor, x, y, feature=2.0, tv=0.1, dice=0.5, verbose=True ):
    y_pred = mdl( x )
    if len( y.shape ) == 5:
            tdim = 3
            myax = [1,2,3,4]
    if len( y.shape ) == 4:
            tdim = 2
            myax = [1,2,3]
    y_intensity = tf.split( y, 2, axis=tdim+1 )[0]
    y_seg = tf.split( y, 2, axis=tdim+1 )[1]
    y_intensity_p = tf.split( y_pred, 2, axis=tdim+1 )[0]
    y_seg_p = tf.split( y_pred, 2, axis=tdim+1 )[1]
    squared_difference = tf.square( y_intensity - y_intensity_p )
    msqTerm = tf.reduce_mean(squared_difference, axis=myax)
    temp1 = feature_extractor(y_intensity)
    temp2 = feature_extractor(y_intensity_p)
    feature_difference = tf.square(temp1-temp2)
    featureTerm = tf.reduce_mean(feature_difference, axis=myax)
    msqw = 10.0
    featw = feature * msqw * msqTerm / featureTerm
    mytv = tf.cast( 0.0, 'float32')
    if tdim == 3:
        for k in range( y_pred.shape[0] ): # BUG not sure why myr fails .... might be old TF version
            sqzd = y_pred[k,:,:,:,0]
            mytv = mytv + tf.reduce_mean( tf.image.total_variation( sqzd ) )
    if tdim == 2:
        mytv = tf.reduce_mean( tf.image.total_variation( y_pred[:,:,:,0] ) )
    tvw = tv * msqw * msqTerm / mytv
    mydice = binary_dice_loss( y_seg, y_seg_p )
    mydice = tf.reduce_mean( mydice )
    dicew = dice * msqw * msqTerm / mydice
    dicewt = np.abs( dicew.numpy().mean() )
    if verbose :
        print( "MSQ: " + str( msqw * msqTerm ) )
        print( "Feat: " + str( featw * featureTerm ) )
        print( "Tv: " + str(  mytv * tvw ) )
        print( "Dice: " + str( mydice * dicewt ) )
    wts = [msqw,featw.numpy().mean(),tvw.numpy().mean(), dicewt ]
    return wts

def numpy_generator( filenames ):
    patchesResam=patchesOrig=patchesUp=None
    yield (patchesResam, patchesOrig,patchesUp)

def image_generator(
    filenames,
    nPatches,
    target_patch_size,
    target_patch_size_low,
    patch_scaler=True,
    istest=False,
    verbose = False ):
    while True:
        patchesResam, patchesOrig, patchesUp = image_patch_training_data_from_filenames(
            filenames,
            target_patch_size = target_patch_size,
            target_patch_size_low = target_patch_size_low,
            nPatches = nPatches,
            istest   = istest,
            patch_scaler=patch_scaler,
            to_tensorflow = True,
            verbose = verbose )
        if istest:
            yield (patchesResam, patchesOrig,patchesUp)
        yield (patchesResam, patchesOrig)


def seg_generator(
    filenames,
    nPatches,
    target_patch_size,
    target_patch_size_low,
    patch_scaler=True,
    istest=False,
    verbose = False ):
    while True:
        patchesResam, patchesOrig, patchesUp = seg_patch_training_data_from_filenames(
            filenames,
            target_patch_size = target_patch_size,
            target_patch_size_low = target_patch_size_low,
            nPatches = nPatches,
            istest   = istest,
            patch_scaler=patch_scaler,
            to_tensorflow = True,
            verbose = verbose )
        if istest:
            yield (patchesResam, patchesOrig,patchesUp)
        yield (patchesResam, patchesOrig)


def train(
    mdl,
    filenames_train,
    filenames_test,
    target_patch_size,
    target_patch_size_low,
    output_prefix,
    n_test = 8,
    learning_rate=5e-5,
    feature_layer = 6,
    feature = 2,
    tv = 0.1,
    max_iterations = 1000,
    batch_size = 1,
    save_all_best = False,
    feature_type = 'grader',
    check_eval_data_iteration = 20,
    verbose = False  ):
    colnames = ['train_loss','test_loss','best','eval_psnr','eval_psnr_lin']
    training_path = np.zeros( [ max_iterations, len(colnames) ] )
    training_weights = np.zeros( [1,3] )
    if verbose:
        print("begin get feature extractor " + feature_type)
    if feature_type == 'grader':
        feature_extractor = get_grader_feature_network( feature_layer )
    elif feature_type == 'vggrandom':
        with eager_mode():
            feature_extractor = pseudo_3d_vgg_features( target_patch_size, feature_layer, pretrained=False )
    elif feature_type == 'vgg':
        with eager_mode():
            feature_extractor = pseudo_3d_vgg_features_unbiased( target_patch_size, feature_layer )
    else:
        raise Exception("feature type does not exist")
    if verbose:
        print("begin train generator")
    mydatgen = image_generator(
        filenames_train,
        nPatches=1,
        target_patch_size=target_patch_size,
        target_patch_size_low=target_patch_size_low,
        istest=False , verbose=False)
    if verbose:
        print("begin test generator")
    mydatgenTest = image_generator( filenames_test, nPatches=1,
        target_patch_size=target_patch_size,
        target_patch_size_low=target_patch_size_low,
        istest=True, verbose=True)
    patchesResamTeTf, patchesOrigTeTf, patchesUpTeTf = next( mydatgenTest )
    if len( patchesOrigTeTf.shape ) == 5:
            tdim = 3
            myax = [1,2,3,4]
    if len( patchesOrigTeTf.shape ) == 4:
            tdim = 2
            myax = [1,2,3]
    if verbose:
        print("begin train generator #2 at dim: " + str( tdim))
    mydatgenTest = image_generator( filenames_test, nPatches=1,
        target_patch_size=target_patch_size,
        target_patch_size_low=target_patch_size_low,
        istest=True, verbose=True)
    patchesResamTeTfB, patchesOrigTeTfB, patchesUpTeTfB = next( mydatgenTest )
    for k in range( n_test - 1 ):
        mydatgenTest = image_generator( filenames_test, nPatches=1,
            target_patch_size=target_patch_size,
            target_patch_size_low=target_patch_size_low,
            istest=True, verbose=True)
        temp0, temp1, temp2 = next( mydatgenTest )
        patchesResamTeTfB = tf.concat( [patchesResamTeTfB,temp0],axis=0)
        patchesOrigTeTfB = tf.concat( [patchesOrigTeTfB,temp1],axis=0)
        patchesUpTeTfB = tf.concat( [patchesUpTeTfB,temp2],axis=0)
    if verbose:
        print("begin auto_weight_loss")
    wts_csv = output_prefix + "_training_weights.csv"
    if exists( wts_csv ):
        wtsdf = pd.read_csv( wts_csv )
        wts = [wtsdf['msq'][0], wtsdf['feat'][0], wtsdf['tv'][0]]
        if verbose:
            print( "preset weights:" )
    else:
        with eager_mode():
            wts = auto_weight_loss( mdl, feature_extractor, patchesResamTeTf, patchesOrigTeTf,
                feature=feature, tv=tv )
        for k in range(len(wts)):
            training_weights[0,k]=wts[k]
        pd.DataFrame(training_weights, columns = ["msq","feat","tv"] ).to_csv( wts_csv )
        if verbose:
            print( "automatic weights:" )
    if verbose:
        print( wts )
    def my_loss_6( y_true, y_pred, msqwt = wts[0], fw = wts[1], tvwt = wts[2], mybs = batch_size ):
        squared_difference = tf.square(y_true - y_pred)
        if len( y_true.shape ) == 5:
            tdim = 3
            myax = [1,2,3,4]
        if len( y_true.shape ) == 4:
            tdim = 2
            myax = [1,2,3]
        msqTerm = tf.reduce_mean(squared_difference, axis=myax)
        temp1 = feature_extractor(y_true)
        temp2 = feature_extractor(y_pred)
        feature_difference = tf.square(temp1-temp2)
        featureTerm = tf.reduce_mean(feature_difference, axis=myax)
        loss = msqTerm * msqwt + featureTerm * fw
        mytv = tf.cast( 0.0, 'float32')
        # mybs =  int( y_pred.shape[0] ) --- should work but ... ?
        if tdim == 3:
            for k in range( mybs ): # BUG not sure why myr fails .... might be old TF version
                sqzd = y_pred[k,:,:,:,:]
                mytv = mytv + tf.reduce_mean( tf.image.total_variation( sqzd ) ) * tvwt
        if tdim == 2:
            mytv = tf.reduce_mean( tf.image.total_variation( y_pred ) ) * tvwt
        return( loss + mytv )
    if verbose:
        print("begin model compilation")
    opt = tf.keras.optimizers.Adam( learning_rate=learning_rate )
    mdl.compile(optimizer=opt, loss=my_loss_6)
    # set up some parameters for tracking performance
    bestValLoss=1e12
    bestSSIM=0.0
    bestQC0 = -1000
    bestQC1 = -1000
    if verbose:
        print( "begin training", flush=True  )
    for myrs in range( max_iterations ):
        tracker = mdl.fit( mydatgen,  epochs=2, steps_per_epoch=4, verbose=1,
            validation_data=(patchesResamTeTf,patchesOrigTeTf),
            workers = 1, use_multiprocessing=False )
        training_path[myrs,0]=tracker.history['loss'][0]
        training_path[myrs,1]=tracker.history['val_loss'][0]
        training_path[myrs,2]=0
        print( "ntrain: " + str(myrs) + " loss " + str( tracker.history['loss'][0] ) + ' val-loss ' + str(tracker.history['val_loss'][0]), flush=True  )
        if myrs % check_eval_data_iteration == 0:
            with tf.device("/cpu:0"):
                myofn = output_prefix + "_best_mdl.h5"
                if save_all_best:
                    myofn = output_prefix + "_" + str(myrs)+ "_mdl.h5"
                tester = mdl.evaluate( patchesResamTeTfB, patchesOrigTeTfB )
                if ( tester < bestValLoss ):
                    print("MyIT " + str( myrs ) + " IS BEST!! " + str( tester ) + myofn, flush=True )
                    bestValLoss = tester
                    tf.keras.models.save_model( mdl, myofn )
                    training_path[myrs,2]=1
                pp = mdl.predict( patchesResamTeTfB, batch_size = 1 )
                myssimSR = tf.image.psnr( pp * 220, patchesOrigTeTfB* 220, max_val=255 )
                myssimSR = tf.reduce_mean( myssimSR ).numpy()
                myssimBI = tf.image.psnr( patchesUpTeTfB * 220, patchesOrigTeTfB* 220, max_val=255 )
                myssimBI = tf.reduce_mean( myssimBI ).numpy()
                print( myofn + " : " + "PSNR Lin: " + str( myssimBI ) + " SR: " + str( myssimSR ), flush=True  )
                training_path[myrs,3]=myssimSR # psnr
                training_path[myrs,4]=myssimBI # psnrlin
                pd.DataFrame(training_path, columns = colnames ).to_csv( output_prefix + "_training.csv" )
    training_path = pd.DataFrame(training_path, columns = colnames )
    return training_path


def binary_dice_loss(y_true, y_pred):
  smoothing_factor = 1e-4
  K = tf.keras.backend
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return -1 * (2 * intersection + smoothing_factor)/(K.sum(y_true_f) +
        K.sum(y_pred_f) + smoothing_factor)

def train_seg(
    mdl,
    filenames_train,
    filenames_test,
    target_patch_size,
    target_patch_size_low,
    output_prefix,
    n_test = 8,
    learning_rate=5e-5,
    feature_layer = 6,
    feature = 2,
    tv = 0.1,
    dice = 0.5,
    max_iterations = 1000,
    batch_size = 1,
    save_all_best = False,
    feature_type = 'grader',
    check_eval_data_iteration = 20,
    verbose = False  ):
    colnames = ['train_loss','test_loss','best','eval_psnr','eval_psnr_lin','eval_msq','eval_dice']
    training_path = np.zeros( [ max_iterations, len(colnames) ] )
    training_weights = np.zeros( [1,4] )
    if verbose:
        print("begin get feature extractor")
    if feature_type == 'grader':
        feature_extractor = get_grader_feature_network( feature_layer )
    elif feature_type == 'vggrandom':
        feature_extractor = pseudo_3d_vgg_features( target_patch_size, feature_layer, pretrained=False )
    else:
        feature_extractor = pseudo_3d_vgg_features_unbiased( target_patch_size, feature_layer  )
    if verbose:
        print("begin train generator")
    mydatgen = seg_generator(
        filenames_train,
        nPatches=1,
        target_patch_size=target_patch_size,
        target_patch_size_low=target_patch_size_low,
        istest=False , verbose=False)
    if verbose:
        print("begin test generator")
    mydatgenTest = seg_generator( filenames_test, nPatches=1,
        target_patch_size=target_patch_size,
        target_patch_size_low=target_patch_size_low,
        istest=True, verbose=True)
    patchesResamTeTf, patchesOrigTeTf, patchesUpTeTf = next( mydatgenTest )
    if len( patchesOrigTeTf.shape ) == 5:
            tdim = 3
            myax = [1,2,3,4]
    if len( patchesOrigTeTf.shape ) == 4:
            tdim = 2
            myax = [1,2,3]
    if verbose:
        print("begin train generator #2 at dim: " + str( tdim))
    mydatgenTest = seg_generator( filenames_test, nPatches=1,
        target_patch_size=target_patch_size,
        target_patch_size_low=target_patch_size_low,
        istest=True, verbose=True)
    patchesResamTeTfB, patchesOrigTeTfB, patchesUpTeTfB = next( mydatgenTest )
    for k in range( n_test - 1 ):
        mydatgenTest = seg_generator( filenames_test, nPatches=1,
            target_patch_size=target_patch_size,
            target_patch_size_low=target_patch_size_low,
            istest=True, verbose=True)
        temp0, temp1, temp2 = next( mydatgenTest )
        patchesResamTeTfB = tf.concat( [patchesResamTeTfB,temp0],axis=0)
        patchesOrigTeTfB = tf.concat( [patchesOrigTeTfB,temp1],axis=0)
        patchesUpTeTfB = tf.concat( [patchesUpTeTfB,temp2],axis=0)
    if verbose:
        print("begin auto_weight_loss_seg")
    wts_csv = output_prefix + "_training_weights.csv"
    if exists( wts_csv ):
        wtsdf = pd.read_csv( wts_csv )
        wts = [wtsdf['msq'][0], wtsdf['feat'][0], wtsdf['tv'][0], wtsdf['dice'][0]]
        if verbose:
            print( "preset weights:" )
    else:
        wts = auto_weight_loss_seg( mdl, feature_extractor, patchesResamTeTf, patchesOrigTeTf,
            feature=feature, tv=tv, dice=dice )
        for k in range(len(wts)):
            training_weights[0,k]=wts[k]
        pd.DataFrame(training_weights, columns = ["msq","feat","tv","dice"] ).to_csv( wts_csv )
        if verbose:
            print( "automatic weights:" )
    if verbose:
        print( wts )
    def my_loss_6( y_true, y_pred, msqwt = wts[0], fw = wts[1], tvwt = wts[2], dicewt=wts[3], mybs = batch_size ):
        if len( y_true.shape ) == 5:
            tdim = 3
            myax = [1,2,3,4]
        if len( y_true.shape ) == 4:
            tdim = 2
            myax = [1,2,3]
        y_intensity = tf.split( y_true, 2, axis=tdim+1 )[0]
        y_seg = tf.split( y_true, 2, axis=tdim+1 )[1]
        y_intensity_p = tf.split( y_pred, 2, axis=tdim+1 )[0]
        y_seg_p = tf.split( y_pred, 2, axis=tdim+1 )[1]
        squared_difference = tf.square(y_intensity - y_intensity_p)
        msqTerm = tf.reduce_mean(squared_difference, axis=myax)
        temp1 = feature_extractor(y_intensity)
        temp2 = feature_extractor(y_intensity_p)
        feature_difference = tf.square(temp1-temp2)
        featureTerm = tf.reduce_mean(feature_difference, axis=myax)
        loss = msqTerm * msqwt + featureTerm * fw
        mytv = tf.cast( 0.0, 'float32')
        if tdim == 3:
            for k in range( mybs ): # BUG not sure why myr fails .... might be old TF version
                sqzd = y_pred[k,:,:,:,0]
                mytv = mytv + tf.reduce_mean( tf.image.total_variation( sqzd ) ) * tvwt
        if tdim == 2:
            mytv = tf.reduce_mean( tf.image.total_variation( y_pred[:,:,:,0] ) ) * tvwt
        dicer = tf.reduce_mean( dicewt * binary_dice_loss( y_seg, y_seg_p ) )
        return( loss + mytv + dicer )
    if verbose:
        print("begin model compilation")
    opt = tf.keras.optimizers.Adam( learning_rate=learning_rate )
    mdl.compile(optimizer=opt, loss=my_loss_6)
    # set up some parameters for tracking performance
    bestValLoss=1e12
    bestSSIM=0.0
    bestQC0 = -1000
    bestQC1 = -1000
    if verbose:
        print( "begin training", flush=True  )
    for myrs in range( max_iterations ):
        tracker = mdl.fit( mydatgen,  epochs=2, steps_per_epoch=4, verbose=1,
            validation_data=(patchesResamTeTf,patchesOrigTeTf),
            workers = 1, use_multiprocessing=False )
        training_path[myrs,0]=tracker.history['loss'][0]
        training_path[myrs,1]=tracker.history['val_loss'][0]
        training_path[myrs,2]=0
        print( "ntrain: " + str(myrs) + " loss " + str( tracker.history['loss'][0] ) + ' val-loss ' + str(tracker.history['val_loss'][0]), flush=True  )
        if myrs % check_eval_data_iteration == 0:
            with tf.device("/cpu:0"):
                myofn = output_prefix + "_best_mdl.h5"
                if save_all_best:
                    myofn = output_prefix + "_" + str(myrs)+ "_mdl.h5"
                tester = mdl.evaluate( patchesResamTeTfB, patchesOrigTeTfB )
                if ( tester < bestValLoss ):
                    print("MyIT " + str( myrs ) + " IS BEST!! " + str( tester ) + myofn, flush=True )
                    bestValLoss = tester
                    tf.keras.models.save_model( mdl, myofn )
                    training_path[myrs,2]=1
                pp = mdl.predict( patchesResamTeTfB, batch_size = 1 )
                pp = tf.split( pp, 2, axis=tdim+1 )
                y_orig = tf.split( patchesOrigTeTfB, 2, axis=tdim+1 )
                y_up = tf.split( patchesUpTeTfB, 2, axis=tdim+1 )
                myssimSR = tf.image.psnr( pp[0] * 220, y_orig[0]* 220, max_val=255 )
                myssimSR = tf.reduce_mean( myssimSR ).numpy()
                myssimBI = tf.image.psnr( y_up[0] * 220, y_orig[0]* 220, max_val=255 )
                myssimBI = tf.reduce_mean( myssimBI ).numpy()
                squared_difference = tf.square(y_orig[0] - pp[0])
                msqTerm = tf.reduce_mean(squared_difference).numpy()
                dicer = binary_dice_loss( y_orig[1], pp[1] )
                dicer = tf.reduce_mean( dicer ).numpy()
                print( myofn + " : " + "PSNR Lin: " + str( myssimBI ) + " SR: " + str( myssimSR ) + " MSQ: " + str(msqTerm) + " DICE: " + str(dicer), flush=True  )
                training_path[myrs,3]=myssimSR # psnr
                training_path[myrs,4]=myssimBI # psnrlin
                training_path[myrs,5]=msqTerm # psnr
                training_path[myrs,6]=dicer # psnrlin
                pd.DataFrame(training_path, columns = colnames ).to_csv( output_prefix + "_training.csv" )
    training_path = pd.DataFrame(training_path, columns = colnames )
    return training_path


def read_srmodel(srfilename, custom_objects=None):
    """
    Load a super-resolution model (h5, .keras, or SavedModel format),
    and determine its upsampling factor.

    Parameters
    ----------
    srfilename : str
        Path to the model file (.h5, .keras, or a SavedModel folder).
    custom_objects : dict, optional
        Dictionary of custom objects used in the model (e.g. {'TFOpLambda': tf.keras.layers.Lambda(...)})

    Returns
    -------
    model : tf.keras.Model
        The loaded model.
    upsampling_factor : list of int
        List describing the upsampling factor:
        - For 3D input: [x_up, y_up, z_up, channels]
        - For 2D input: [x_up, y_up, channels]

    Example
    -------
    >>> mdl, up = read_srmodel("mymodel.keras")
    >>> mdl, up = read_srmodel("my_weights.h5", custom_objects={"TFOpLambda": tf.keras.layers.Lambda(tf.identity)})
    """

    # Expand path and detect format
    srfilename = os.path.expanduser(srfilename)
    ext = os.path.splitext(srfilename)[1].lower()

    if os.path.isdir(srfilename):
        # SavedModel directory
        model = tf.keras.models.load_model(srfilename, custom_objects=custom_objects, compile=False)
    elif ext in ['.h5', '.keras']:
        model = tf.keras.models.load_model(srfilename, custom_objects=custom_objects, compile=False)
    else:
        raise ValueError(f"Unsupported model format: {ext}")

    # Determine channel index
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    chanindex = 3 if len(input_shape) == 4 else 4
    nchan = int(input_shape[chanindex])

    # Run dummy input to compute upsampling factor
    try:
        if len(input_shape) == 5:  # 3D
            dummy_input = np.zeros([1, 8, 8, 8, nchan])
        else:  # 2D
            dummy_input = np.zeros([1, 8, 8, nchan])

        # Handle named inputs if necessary
        try:
            output = model(dummy_input)
        except Exception:
            output = model({model.input_names[0]: dummy_input})

        outshp = output.shape
        if len(input_shape) == 5:
            return model, [int(outshp[1]/8), int(outshp[2]/8), int(outshp[3]/8), nchan]
        else:
            return model, [int(outshp[1]/8), int(outshp[2]/8), nchan]

    except Exception as e:
        raise RuntimeError(f"Could not infer upsampling factor. Error: {e}")


def simulate_image( shaper=[32,32,32], n_levels=10, multiply=False ):
    """
    generate an image of given shape and number of levels

    Arguments
    ---------
    shaper : [x,y,z] or [x,y]

    n_levels : int

    multiply : boolean

    Returns
    -------

    ants.image

    """
    img = ants.from_numpy( np.random.normal( 0, 1.0, size=shaper ) ) * 0
    for k in range(n_levels):
        temp = ants.from_numpy( np.random.normal( 0, 1.0, size=shaper ) )
        temp = ants.smooth_image( temp, n_levels )
        temp = ants.threshold_image( temp, "Otsu", 1 )
        if multiply:
            temp = temp * k
        img = img + temp
    return img

def optimize_upsampling_shape( spacing, modality='T1', roundit=False, verbose=False ):
    minspc = min( list( spacing ) )
    maxspc = max( list( spacing ) )
    ratio = maxspc/minspc
    if ratio == 1.0:
        ratio = 0.5
    roundratio = np.round( ratio )
    tarshaperaw = []
    tarshape = []
    tarshaperound = []
    for k in range( len( spacing ) ):
        locrat = spacing[k]/minspc
        newspc = spacing[k] * roundratio
        tarshaperaw.append( locrat )
        if modality == "NM":
            if verbose:
                print("Using minspacing: 0.25")
            if newspc < 0.25 :
                locrat = spacing[k]/0.25
        elif modality == "DTI":
            if verbose:
                print("Using minspacing: 1.0")
            if newspc < 1.0 :
                locrat = spacing[k]/1.0
        else: # assume T1
            if verbose:
                print("Using minspacing: 0.35")
            if newspc < 0.35 :
                locrat = spacing[k]/0.35
        myint = int( locrat )
        if ( myint == 0 ):
            myint = 1
        if myint == 5:
            myint = 4
        if ( myint > 6 ):
            myint = 6
        tarshape.append( str( myint ) )
        tarshaperound.append( str( int(np.round( locrat )) ) )
    if verbose:
        print("before emendation:")
        print( tarshaperaw )
        print( tarshaperound )
        print( tarshape )
    allone = True
    if roundit:
        tarshape = tarshaperound
    for k in range( len( tarshape ) ):
        if tarshape[k] != "1":
            allone=False
    if allone:
        tarshape = ["2","2","2"] # default
    return "x".join(tarshape)

def compare_models( model_filenames, img, n_classes=3,
    poly_order='hist',
    identifier=None, noise_sd=0.1,verbose=False ):
    """
    generate a dataframe computing some basic intensity metrics PSNR and SSIM

    by default, will add noise to downsampled image.

    NOTE: when evaluating a 2-channel (segmentation) model - the focus should
    be on the segmentation arm (DICE) alone ... the intensity component can
    be evaluated on its own separately.
    """
    padding=4
    mydf = pd.DataFrame()
    for k in range( len( model_filenames ) ):
        srmdl, upshape = read_srmodel( model_filenames[k] )
        if verbose:
            print( model_filenames[k] )
            print( upshape )
        tarshape = []
        inspc = ants.get_spacing(img)
        for j in range(len(img.shape)):
            tarshape.append( float(upshape[j]) * inspc[j] )
        # uses linear interp
        dimg=ants.resample_image( img, tarshape, use_voxels=False, interp_type=0 )
        dimg = ants.add_noise_to_image( dimg,'additivegaussian', [0,noise_sd] )
        import math
        dicesr=math.nan
        dicenn=math.nan
        if upshape[3] == 2:
            seghigh = ants.threshold_image( img,"Otsu",n_classes)
            seglow = ants.resample_image( seghigh, tarshape, use_voxels=False, interp_type=1 )
            dimgup=inference( dimg, srmdl, segmentation = seglow, poly_order=poly_order, verbose=verbose )
            dimgupseg = dimgup['super_resolution_segmentation']
            dimgup = dimgup['super_resolution']
            segblock = ants.resample_image_to_target( seghigh, dimgupseg, interp_type='nearestNeighbor'  )
            segimgnn = ants.resample_image_to_target( seglow, dimgupseg, interp_type='nearestNeighbor' )
            segblock[ dimgupseg == 0 ] = 0
            segimgnn[ dimgupseg == 0 ] = 0
            dicenn = ants.label_overlap_measures(segblock, segimgnn)['MeanOverlap'][0]
            dicesr = ants.label_overlap_measures(segblock, dimgupseg)['MeanOverlap'][0]
        else:
            dimgup=inference( dimg, srmdl, poly_order=poly_order, verbose=verbose )
        dimglin = ants.resample_image_to_target( dimg, dimgup, interp_type='linear' )
        imgblock = ants.resample_image_to_target( img, dimgup, interp_type='linear'  )
        dimgup[ imgblock == 0.0 ]=0.0
        dimglin[ imgblock == 0.0 ]=0.0
        padder = []
        dimwarning=False
        for jj in range(img.dimension):
            padder.append( padding )
            if img.shape[jj] != imgblock.shape[jj]:
                dimwarning=True
        if dimwarning:
            print("NOTE: dimensions of downsampled to upsampled image do not match!!!")
            print("we force them to match but this suggests results may not be reliable.")
        temp = os.path.basename( model_filenames[k] )
        temp = re.sub( "siq_default_sisr_", "", temp )
        temp = re.sub( "_best_mdl.h5", "", temp )
        if verbose and dimwarning:
            print( "original img shape" )
            print( img.shape )
            print( "resampled img shape" )
            print( imgblock.shape )
        a=[]
        imgshape = []
        for aa in range(len(upshape)):
            a.append( str(upshape[aa]) )
            if aa < len(imgblock.shape):
                imgshape.append( str( imgblock.shape[aa] ) )
        if identifier is None:
            identifier=temp
        mydict = {
            "identifier":identifier,
            "imgshape":"x".join(imgshape),
            "mdl": temp,
            "mdlshape":"x".join(a),
            "PSNR.LIN": antspynet.psnr( imgblock, dimglin ),
            "PSNR.SR": antspynet.psnr( imgblock, dimgup ),
            "SSIM.LIN": antspynet.ssim( imgblock, dimglin ),
            "SSIM.SR": antspynet.ssim( imgblock, dimgup ),
            "DICE.NN": dicenn,
            "DICE.SR": dicesr,
            "dimwarning": dimwarning }
        if verbose:
            print( mydict )
        temp = pd.DataFrame.from_records( [mydict], index=[0] )
        mydf = pd.concat( [mydf,temp], axis=0 )
        # end loop
    return mydf




def region_wise_super_resolution(image, mask, super_res_model, dilation_amount=4, verbose=False):
    """
    Apply super-resolution model to each labeled region in the mask independently.

    Arguments
    ---------
    image : ANTsImage
        Input image.

    mask : ANTsImage
        Integer-labeled segmentation mask with non-zero regions to upsample.

    super_res_model : tf.keras.Model
        Trained super-resolution model.

    dilation_amount : int
        Number of morphological dilations applied to each label region before cropping.

    verbose : bool
        If True, print detailed status.

    Returns
    -------
    ANTsImage : Full-size super-resolved image with per-label inference and stitching.
    """
    import ants
    import numpy as np
    from antspynet import apply_super_resolution_model_to_image

    upFactor = []
    input_shape = super_res_model.inputs[0].shape
    test_shape = [1, 8, 8, 1] if len(input_shape) == 4 else [1, 8, 8, 8, 1]
    test_input = np.zeros(test_shape, dtype=np.float32)
    test_output = super_res_model(test_input)

    for k in range(len(test_shape) - 2):  # ignore batch + channel
        upFactor.append(int(test_output.shape[k + 1] / test_input.shape[k + 1]))

    original_size = mask.shape  # e.g., (x, y, z)
    new_size = tuple(int(s * f) for s, f in zip(original_size, upFactor))
    upsampled_mask = ants.resample_image(mask, new_size, use_voxels=True, interp_type=1)
    upsampled_image = ants.resample_image(image, new_size, use_voxels=True, interp_type=0)

    unique_labels = list(np.unique(upsampled_mask.numpy()))
    if 0 in unique_labels:
        unique_labels.remove(0)

    outimg = ants.image_clone(upsampled_image)

    for lab in unique_labels:
        if verbose:
            print(f"Processing label: {lab}")
        regionmask = ants.threshold_image(mask, lab, lab).iMath("MD", dilation_amount)
        cropped = ants.crop_image(image, regionmask)
        if cropped.shape[0] == 0:
            continue
        subimgsr = apply_super_resolution_model_to_image(
            cropped, super_res_model, target_range=[0, 1], verbose=verbose
        )
        stitched = ants.decrop_image(subimgsr, outimg)
        outimg[upsampled_mask == lab] = stitched[upsampled_mask == lab]

    return outimg


def region_wise_super_resolution_blended(image, mask, super_res_model, dilation_amount=4, verbose=False):
    """
    Apply super-resolution model to labeled regions with smooth blending to minimize stitching artifacts.

    This version uses a weighted-averaging scheme based on distance transforms
    to create seamless transitions between super-resolved regions and the background.

    Arguments
    ---------
    image : ANTsImage
        Input low-resolution image.

    mask : ANTsImage
        Integer-labeled segmentation mask.

    super_res_model : tf.keras.Model
        Trained super-resolution model.

    dilation_amount : int
        Number of morphological dilations applied to each label region before cropping.
        This provides context to the SR model.

    verbose : bool
        If True, print detailed status.

    Returns
    -------
    ANTsImage : Full-size, super-resolved image with seamless blending.
    """
    import ants
    import numpy as np
    from antspynet import apply_super_resolution_model_to_image
    epsilon32 = np.finfo(np.float32).eps
    normalize_weight_maps = True  # Default behavior to normalize weight maps
    # --- Step 1: Determine upsampling factor and prepare initial images ---
    upFactor = []
    input_shape = super_res_model.inputs[0].shape
    test_shape = [1, 8, 8, 1] if len(input_shape) == 4 else [1, 8, 8, 8, 1]
    test_input = np.zeros(test_shape, dtype=np.float32)
    test_output = super_res_model(test_input)
    for k in range(len(test_shape) - 2):
        upFactor.append(int(test_output.shape[k + 1] / test_input.shape[k + 1]))

    original_size = image.shape
    new_size = tuple(int(s * f) for s, f in zip(original_size, upFactor))

    # The initial upsampled image will serve as our background
    background_sr_image = ants.resample_image(image, new_size, use_voxels=True, interp_type=0)

    # --- Step 2: Initialize accumulator and weight sum canvases ---
    # These must be float type for accumulation
    accumulator = ants.image_clone(background_sr_image).astype('float32') * 0.0
    weight_sum = ants.image_clone(accumulator)

    unique_labels = [l for l in np.unique(mask.numpy()) if l != 0]

    for lab in unique_labels:
        if verbose:
            print(f"Blending label: {lab}")

        # --- Step 3: Super-resolve a dilated patch (provides context to the model) ---
        region_mask_dilated = ants.threshold_image(mask, lab, lab).iMath("MD", dilation_amount)
        cropped_lowres = ants.crop_image(image, region_mask_dilated)
        if cropped_lowres.shape[0] == 0:
            continue
            
        # Apply the model to the cropped low-res patch
        sr_patch = apply_super_resolution_model_to_image(
            cropped_lowres, super_res_model, target_range=[0, 1]
        )
        
        # Place the super-resolved patch back onto a full-sized canvas
        sr_patch_full_size = ants.decrop_image(sr_patch, accumulator)

        # --- Step 4: Create a smooth weight map for this region ---
        # We use the *non-dilated* mask for the weight map to ensure a sharp focus on the target region.
        region_mask_original = ants.threshold_image(mask, lab, lab)
        
        # Resample the original region mask to the high-res grid
        weight_map = ants.resample_image(region_mask_original, new_size, use_voxels=True, interp_type=0)
        weight_map = ants.smooth_image(weight_map, sigma=2.0,
                                        sigma_in_physical_coordinates=False)
        if normalize_weight_maps:
            weight_map = ants.iMath(weight_map, "Normalize")
        # --- Step 5: Accumulate the weighted values and the weights themselves ---
        accumulator += sr_patch_full_size * weight_map
        weight_sum += weight_map

    # --- Step 6: Final Combination ---
    # Normalize the accumulator by the total weight at each pixel
    weight_sum_np = weight_sum.numpy()
    accumulator_np = accumulator.numpy()
    
    # Create a mask of pixels where blending occurred
    blended_mask = weight_sum_np > 0.0 # Use a small epsilon for float safety

    # Start with the original upsampled image as the base
    final_image_np = background_sr_image.numpy()
    
    # Perform the weighted average only where weights are non-zero
    final_image_np[blended_mask] = accumulator_np[blended_mask] / weight_sum_np[blended_mask]
    
    # Re-insert any non-blended background regions that were processed
    # This handles cases where regions overlap; the weighted average takes care of it.
    
    return ants.from_numpy(final_image_np, origin=background_sr_image.origin, 
                           spacing=background_sr_image.spacing, direction=background_sr_image.direction)
     

def inference(
    image,
    mdl,
    truncation=None,
    segmentation=None,
    target_range=[1, 0],
    poly_order='hist',
    dilation_amount=0,
    verbose=False):
    """
    Perform super-resolution inference on an input image, optionally guided by segmentation.

    This function uses a trained deep learning model to enhance the resolution of a medical image.
    It optionally applies label-wise inference if a segmentation mask is provided.

    Parameters
    ----------
    image : ants.ANTsImage
        Input image to be super-resolved.

    mdl : keras.Model
        Trained super-resolution model, typically from ANTsPyNet.

    truncation : tuple or list of float, optional
        Percentile values (e.g., [0.01, 0.99]) for intensity truncation before model input.
        If None, no truncation is applied.

    segmentation : ants.ANTsImage, optional
        A labeled segmentation mask. If provided, super-resolution is performed per label
        using `region_wise_super_resolution` or `super_resolution_segmentation_per_label`.

    target_range : list of float
        Intensity range used for scaling the input before applying the model.
        Default is [1, 0] (internal default for `apply_super_resolution_model_to_image`).

    poly_order : int, str or None
        Determines how to match intensity between the super-resolved image and the original.
        Options:
          - 'hist' : use histogram matching
          - int >= 1 : perform polynomial regression of this order
          - None : no intensity adjustment

    dilation_amount : int
        Number of dilation steps applied to each segmentation label during
        region-based super-resolution (if segmentation is provided).

    verbose : bool
        If True, print progress and status messages.

    Returns
    -------
    ANTsImage or dict
        - If `segmentation` is None, returns a single ANTsImage (super-resolved image).
        - If `segmentation` is provided, returns a dictionary with:
            - 'super_resolution': ANTsImage
            - other entries may include label-wise results or metadata.

    Examples
    --------
    >>> import ants
    >>> import antspynet
    >>> from siq import inference
    >>> img = ants.image_read("lowres.nii.gz")
    >>> model = antspynet.get_pretrained_network("dbpn", target_suffix="T1")
    >>> srimg = inference(img, model, truncation=[0.01, 0.99], verbose=True)

    >>> seg = ants.image_read("mask.nii.gz")
    >>> sr_result = inference(img, model, segmentation=seg)
    >>> srimg = sr_result['super_resolution']
    """
    import ants
    import numpy as np
    import antspynet
    import antspyt1w
    from siq import region_wise_super_resolution

    def apply_intensity_match(sr_image, reference_image, order, verbose=False):
        if order is None:
            return sr_image
        if verbose:
            print("Applying intensity match with", order)
        if order == 'hist':
            return ants.histogram_match_image(sr_image, reference_image)
        else:
            return ants.regression_match_image(sr_image, reference_image, poly_order=order)

    pimg = ants.image_clone(image)
    if truncation is not None:
        pimg = ants.iMath(pimg, 'TruncateIntensity', truncation[0], truncation[1])

    input_shape = mdl.inputs[0].shape
    num_channels = int(input_shape[-1])

    if segmentation is not None:
        if num_channels == 1:
            if verbose:
                print("Using region-wise super resolution due to single-channel model with segmentation.")
            sr = region_wise_super_resolution_blended(
                pimg, segmentation, mdl,
                dilation_amount=dilation_amount,
                verbose=verbose
            )
            ref = ants.resample_image_to_target(pimg, sr)
            return apply_intensity_match(sr, ref, poly_order, verbose)
        else:
            mynp = segmentation.numpy()
            mynp = list(np.unique(mynp)[1:len(mynp)].astype(int))
            upFactor = []
            if len(input_shape) == 5:
                testarr = np.zeros([1, 8, 8, 8, 2])
                testarrout = mdl(testarr)
                for k in range(3):
                    upFactor.append(int(testarrout.shape[k + 1] / testarr.shape[k + 1]))
            elif len(input_shape) == 4:
                testarr = np.zeros([1, 8, 8, 2])
                testarrout = mdl(testarr)
                for k in range(2):
                    upFactor.append(int(testarrout.shape[k + 1] / testarr.shape[k + 1]))
            temp = antspyt1w.super_resolution_segmentation_per_label(
                pimg,
                segmentation,
                upFactor,
                mdl,
                segmentation_numbers=mynp,
                target_range=target_range,
                dilation_amount=dilation_amount,
                poly_order=poly_order,
                max_lab_plus_one=True
            )
            imgsr = temp['super_resolution']
            ref = ants.resample_image_to_target(pimg, imgsr)
            return apply_intensity_match(imgsr, ref, poly_order, verbose)

    # Default path: no segmentation
    imgsr = antspynet.apply_super_resolution_model_to_image(
        pimg, mdl, target_range=target_range, regression_order=None, verbose=verbose
    )
    ref = ants.resample_image_to_target(pimg, imgsr)
    return apply_intensity_match(imgsr, ref, poly_order, verbose)

