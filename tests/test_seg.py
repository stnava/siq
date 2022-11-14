import os
import siq
import glob
import ants
fns=glob.glob( os.path.expanduser( "~/.antspyt1w/2*T1w*gz" ) )
import tensorflow as tf
mdl = siq.default_dbpn( [2,2,2], nChannelsIn=2, nChannelsOut=2 ) # should match ratio of high to low size patches
myoutprefix = '/tmp/XXX'
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
image = ants.image_read( fns[0] ).iMath("Normalize")
image = ants.resample_image( image, [48,48,48] )
seg = ants.threshold_image( image, 0.5, 1.0 )
test = siq.inference( image, mdl, segmentation=seg )
