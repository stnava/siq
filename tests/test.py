import os
import siq
import glob
import ants
fns=glob.glob( os.path.expanduser( "~/.antspyt1w/2*T1w*gz" ) )
import tensorflow as tf
ofn = os.path.expanduser("~/code/DPR/models/dsr3d_2up_64_256_6_3_v0.0zzz.h5")
if os.path.exists( ofn ):
    print("existing model")
    mdl = tf.keras.models.load_model( ofn, compile=False )
else:
    print("default model")
    mdl = siq.default_dbpn( [2,2,2] ) # should match ratio of high to low size patches
myoutprefix = '/tmp/XXX'
training_path = siq.train(
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
    max_iterations=2, 
    verbose=True)
training_path.to_csv( myoutprefix + "_training.csv" )
image = ants.image_read( fns[0] )
image = ants.resample_image( image, [48,48,48] )
test = siq.inference( image, mdl )
