import os
import siq
import glob
import ants
fns=glob.glob( os.path.expanduser( "~/.antspyt1w/2*T1w*gz" ) )
print( fns )

if False:
    x,y, notused = siq.image_patch_training_data_from_filenames( 
        filenames=fns,
        target_patch_size=[32,32,32],
        target_patch_size_low=[16,16,16],
        nPatches = 3, 
        istest   = False,
        patch_scaler=True, 
        verbose = True )

# ants.plot( ants.from_numpy( y[1,:,:,:,0] ) )

import tensorflow as tf
ofn = "/Users/stnava/code/DPR/models/dsr3d_2up_64_256_6_3_v0.0.h5" 
mdl = tf.keras.models.load_model( ofn, compile=False )
# myfe = siq.get_grader_feature_network( 6 )
# siq.auto_weight_loss( mdl, myfe, x, y )
# mdl = siq.default_dbpn( strider )
# training_path, evaluation_results = siq.train( mdl, x, y, xte, yte, lin, lte )
a, b = siq.train(
    mdl, 
    fns[0:3], fns[0:3], 
    output_prefix='/tmp/XXX',
    target_patch_size=[32,32,32],
    target_patch_size_low=[16,16,16],
    n_test=2, 
    learning_rate=5e-05, feature_layer=6, 
    feature=2, tv=0.1, max_iterations=5, verbose=True)
