import sys
patch=[64,64,64]
patchlow=[32,32,32]
nchan=1
nthreads='8'
rootdir = "FIXME"
if len(sys.argv) > 1:
        nthreads=sys.argv[1]
        patch=[int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])]
        patchlow=[int(sys.argv[5]),int(sys.argv[6]),int(sys.argv[7])]
        rootdir = sys.argv[8]
import os
os.environ["TF_NUM_INTEROP_THREADS"] = nthreads
os.environ["TF_NUM_INTRAOP_THREADS"] = nthreads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
from os.path import exists
import siq
import random
import glob
import ants
import tensorflow as tf
globber = rootdir + "*nii.gz"
imgfns = glob.glob( globber )
upper = []
uppername=''
for k in range( len( patch ) ):
        upper.append( int(patch[k]/patchlow[k]) )
        ext=''
        if k < len(patch)-1 :
                ext='x'
        uppername=uppername+str(upper[k])+ext
print( upper )
print( uppername )
mdl = siq.default_dbpn( upper,
	nChannelsIn=nchan, nChannelsOut=nchan,
	sigmoid_second_channel=False ) # should match ratio of high to low size patches
f=6 # which feature layer
myoutprefix = './models/siq_default_sisr_'+uppername+'_' + str(nchan) + 'chan_feat'+str(f)
print( myoutprefix )
mdlfn = myoutprefix + "_best_mdl.h5"
if len( imgfns ) < 10:
        raise Exception("Too few images")
random.shuffle(imgfns)
# 90\% training data
n = round( len( imgfns ) * 0.9 )
fnsTrain = imgfns[0:n]      # just start small
fnsTest = imgfns[(n+1):len(imgfns)]    # just a few test for now
small=1e-6
# first set up to do MSQ
training_path = siq.train(
        mdl,
        fnsTrain,
        fnsTest,
        output_prefix=myoutprefix,
        target_patch_size=patch,
        target_patch_size_low=patchlow,
        n_test=6,
        learning_rate=5e-05,
        feature_layer=f,
        feature=small,
        tv=0.1 * small,
        max_iterations=200,
        verbose=True)
training_path.to_csv( myoutprefix + "_pretraining.csv" )
# then reweight and do full training
training_path = siq.train(
        mdl,
        fnsTrain,
        fnsTest,
        output_prefix=myoutprefix,
        target_patch_size=patch,
        target_patch_size_low=patchlow,
        n_test=6,
        learning_rate=5e-05,
        feature_layer=f,
        feature=2.0,
        tv=0.1,
        max_iterations=20000,
        verbose=True)
training_path.to_csv( myoutprefix + "_training.csv" )
