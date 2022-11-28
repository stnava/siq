import os
import siq
import glob
import ants
import tensorflow as tf
import sys
if len(sys.argv) <= 1:
        print("usage:")
        print(sys.argv[0]+ " n  paramfile.csv rootdir  cudanum index output_prefix ")
        print( "n : number of threads for multi-threaded systems")
        print( "paramfile.csv : see tests/make_training_params.R")
        print( "rootdir : path from which to glob *nii.gz on which we train")
        print( "cudanum : optional value for CUDA_VISIBLE_DEVICES (default -1) ")
        print( "index : optional index to the paramfile row (default 0) ")
        print( "output_prefix : optional output prefix (default models/fromparamfile)")
        exit()
index=0
cudanum='-1'
outimageflename = ""
if len(sys.argv) > 1:
        nthreads=sys.argv[1]
        paramfilename = sys.argv[2]
        rootdir = sys.argv[3]
if len(sys.argv) > 4:
        cudanum = sys.argv[4]
if len(sys.argv) > 5:
        index = sys.argv[5]
if len(sys.argv) > 6:
        outimageflename = sys.argv[6]
# see https://www.intel.com/content/www/us/en/developer/articles/technical/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference.html
import subprocess
cpu_sockets =  int(subprocess.check_output('cat /proc/cpuinfo | grep "physical id" | sort -u | wc -l', shell=True))
if cpu_sockets == 0:
	cpu_sockets = nthreads # or should we try another way?  or set to 1?
os.environ["CUDA_VISIBLE_DEVICES"]=cudanum
os.environ["TF_ENABLE_MKL_NATIVE_FORMAT"]="1"
os.environ["KMP_SETTINGS"]="true"
os.environ["KMP_BLOCKTIME"]="0"
os.environ["KMP_AFFINITY"]="granularity=fine,verbose,compact,1,0"
os.environ["TF_ENABLE_ONEDNN_OPTS"]="1"
os.environ["OMP_NUM_THREADS"]=nthreads
os.environ["TF_NUM_INTEROP_THREADS"] = str(cpu_sockets)
os.environ["TF_NUM_INTRAOP_THREADS"] = nthreads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
if "SLUM_ARRAY_TASK_ID" in os.environ:
        index=int( os.environ['SLUM_ARRAY_TASK_ID'] )
print("script called as:")
print(sys.argv[0]+ " \n nthreads : " + nthreads +  " \n paramfilename : " + paramfilename + " \n rootdir : " + rootdir + " \n cudanum : " + cudanum + " \n index : " + str(index) + " \n output_prefix : " + outimageflename )
from os.path import exists
pfn = os.path.expanduser(paramfilename)
rootdir = os.path.expanduser(rootdir)
import pandas as pd
if exists( pfn ):
        myparams = pd.read_csv( pfn )
else:
        print( pfn + " does not exist -- exiting ")
        exit()
paramrow = myparams.iloc[[index]]
print( paramrow )
import numpy as np
patch=[int(paramrow['sz']),int(paramrow['sz']),int(paramrow['sz'])]
patchlow=[
        int(np.round(paramrow['sz'])/int(paramrow['x'])),
        int(np.round(paramrow['sz'])/int(paramrow['y'])),
        int(np.round(paramrow['sz'])/int(paramrow['z']))
        ]
nchan=1
doseg=int(paramrow['seg']) == 1
if doseg:
        nchan=2
import siq
import random
import glob
import ants
import tensorflow as tf
random.seed(0) # reproducibly sample images
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
feet=str(paramrow['feat'].iloc[0])
f=int(paramrow['layer'].iloc[0])
feetext='chan_feat'+feet+"L"+str(f)
myoutprefix = './models/siq_default_sisr_'+uppername+'_' + str(nchan) + feetext
if outimageflename == "":
        outimageflename=myoutprefix
print( outimageflename )
if len( imgfns ) < 10:
        raise Exception("Too few images")
random.shuffle(imgfns)
# 90\% training data
n = round( len( imgfns ) * 0.9 )
fnsTrain = imgfns[0:n]      # just start small
fnsTest = imgfns[(n+1):len(imgfns)]    # just a few test for now
small=1e-6
if not doseg:
        mdl = siq.default_dbpn( upper,
                nChannelsIn=nchan, nChannelsOut=nchan,
                sigmoid_second_channel=False ) # should match ratio of high to low size patches
        # first set up to do MSQ
        training_path = siq.train(
                mdl,
                fnsTrain,
                fnsTest,
                output_prefix=outimageflename+"_pretraining",
                target_patch_size=patch,
                target_patch_size_low=patchlow,
                n_test=6,
                learning_rate=5e-05,
                feature_type=feet,
                feature_layer=f,
                feature=small,
                tv=0.1 * small,
                max_iterations=200,
                verbose=True)
        # then reweight and do full training
        training_path = siq.train(
                mdl,
                fnsTrain,
                fnsTest,
                output_prefix=outimageflename,
                target_patch_size=patch,
                target_patch_size_low=patchlow,
                n_test=6,
                learning_rate=5e-05,
                feature_type=feet,
                feature_layer=f,
                feature=2.0,
                tv=0.1,
                max_iterations=20000,
                verbose=True)
else:
        mdl = siq.default_dbpn( upper,
        	nChannelsIn=nchan, nChannelsOut=nchan,
	        sigmoid_second_channel=True ) # should match ratio of high to low size patches
        training_path = siq.train_seg(
                mdl,
                fnsTrain,
                fnsTest,
                output_prefix=outimageflename+"_pretraining",
                target_patch_size=patch,
                target_patch_size_low=patchlow,
                n_test=6,
                learning_rate=5e-05,
                feature_type=feet,
                feature_layer=f,
                feature=small,
                tv=0.1 * small,
                dice=0.1,
                max_iterations=200,
                verbose=True)
        # then reweight and do full training
        training_path = siq.train_seg(
                mdl,
                fnsTrain,
                fnsTest,
                output_prefix=outimageflename,
                target_patch_size=patch,
                target_patch_size_low=patchlow,
                n_test=6,
                learning_rate=5e-05,
                feature_type=feet,
                feature_layer=f,
                feature=2.0,
                tv=0.1,
                dice=1.0,
                max_iterations=20000,
                verbose=True)
