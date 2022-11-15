import os
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"
from os.path import exists
import siq
import glob
import ants
fns=glob.glob( os.path.expanduser( "~/.antspyt1w/2*T1w*gz" ) )
import tensorflow as tf
mdl = siq.default_dbpn( [2,2,2], 
    nChannelsIn=2, nChannelsOut=2, 
    sigmoid_second_channel=True) # should match ratio of high to low size patches
# import numpy as np
# testarr = np.zeros( [1,8,8,8,2 ] )
# testarrout = mdl( testarr )
myoutprefix = '/tmp/XXX'
mdlfn = myoutprefix + "_best_mdl.h5"
if exists( mdlfn ):
        print( mdlfn + " exists! " )
        mdl=tf.keras.models.load_model( mdlfn, compile=False )
if True:
    training_path = siq.train_seg(
        mdl, 
        fns[0:3], 
        fns[0:3], 
        output_prefix=myoutprefix,
        target_patch_size=[32,32,32],
        target_patch_size_low=[16,16,16],
        n_test=2, 
        learning_rate=5e-05, 
        feature_layer=6, 
        feature=2, 
        tv=0.1,
        dice=0.5,
        max_iterations=2, 
        verbose=True)
    training_path.to_csv( myoutprefix + "_training.csv" )

n=64
image = ants.image_read( fns[0] ).iMath("Normalize")
image = ants.resample_image( image, [n,n,n], use_voxels=True ).iMath("Normalize")
segmentation = ants.threshold_image( image, "Otsu", 2 ).threshold_image(2,2)
test = siq.inference( image, mdl, segmentation=segmentation, verbose=True )
