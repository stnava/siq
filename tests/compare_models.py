import os
import glob
import ants
import antspynet
import siq
import numpy as np
import tensorflow as tf
import re as re
import pandas as pd
mfns=glob.glob("models/siq*1chan*h5")+glob.glob("uvamodels/siq*h5")
mfns=glob.glob("../DPR/models/siq_default_sisr_2x2x2_2chan_featvggL6_best_mdl.h5")
mfns=glob.glob("/Users/stnava/code/DPR/models/siq_default_sisr_2x2x4_1chan_feat6_best_mdl.h5")
mfns.sort()
print( mfns )
img1 = siq.simulate_image( [32,32,32], 10, True )
img2 = siq.simulate_image( [32,32,32], 10, False )
ants.plot( img1 )
ants.plot( img2 )
mycomp = siq.compare_models( mfns, img1, verbose=True )
mycomp.to_csv("srmodel_comparison_results_img1.csv")
mycomp = siq.compare_models( mfns, img2, verbose=True )
mycomp.to_csv("srmodel_comparison_results_img2.csv")
