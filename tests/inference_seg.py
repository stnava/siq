import os
import glob
import ants
import antspynet
import siq
import numpy as np
import tensorflow as tf
import re as re
import pandas as pd
import antspyt1w
n=48
if not 'img1' in locals():
    img1 = siq.simulate_image( [n,n,n], 10, False )
mdlfn='siq_smallshort_train_2x2x4_2chan_featvggL6_postseg_best_mdl.h5'
mdl, upfactor = siq.read_srmodel( mdlfn  )
iseg = ants.threshold_image( img1, 'Otsu', 3 )
img1d=ants.resample_image( img1, upfactor[0:3] )
isegd = ants.resample_image( iseg, upfactor[0:3] )
isegdup=ants.resample_image_to_target( isegd, img1 )
print("BilIn")
print( ants.label_overlap_measures(isegdup, iseg ) )
myinf = siq.inference( img1d, mdl, segmentation=isegd, poly_order=None, verbose=True )
print("SR int-match is None"  )
print( ants.label_overlap_measures(myinf['super_resolution_segmentation'] ,iseg ) )
for mypo in [1,2,'hist']:
    myinf = siq.inference( img1d, mdl, segmentation=isegd, poly_order=mypo, verbose=True )
    print("SR int-match is " + str( mypo ) )
    print( ants.label_overlap_measures(myinf['super_resolution_segmentation'] ,iseg ) )
