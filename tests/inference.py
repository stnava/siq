import os
import siq
import glob
import ants
import tensorflow as tf
import sys
nthreads='8'
if len(sys.argv) > 1:
        nthreads=sys.argv[1]
        modelfilename = sys.argv[2]
        imagefllename = sys.argv[3]
        outimagefllename = sys.argv[4]
import os
os.environ["TF_NUM_INTEROP_THREADS"] = nthreads
os.environ["TF_NUM_INTRAOP_THREADS"] = nthreads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
from os.path import exists
ofn = os.path.expanduser(modelfilename)
mdl = tf.keras.models.load_model( ofn, compile=False )
image = ants.image_read( os.path.expanduser(imagefllename) )

# image = ants.crop_image( image )
with tf.device("/cpu:0"):
	test = siq.inference( image, mdl )
ants.image_write( test, os.path.expanduser(outimagefllename) )

