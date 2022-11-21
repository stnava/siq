# SIQ - super-resolution image quantification

## deep perceptual resampling and super-resolution for (medical) imaging

install by calling (within the source directory):

```
python setup.py install
```

or install via `pip install siq`

# what this will do

facilitates:

* creating training and testing data for deep networks

* generating and testing perceptual losses in 2D and 3D

* general training and inference functions for deep networks

* intuitive weighting of multiple losses

* anisotropic super-resolution

* evaluation strategies for the above

# first time setup

```python
import antspyt1w
antspyt1w.get_data( force_download=True )
# import siq     # FIXME - for later
# siq.get_data( force_download=True )
```

NOTE: `get_data` has a `force_download` option to make sure the latest
package data is installed.

# example processing

```python
import os
import siq
import glob
import ants
fns=glob.glob( os.path.expanduser( "~/.antspyt1w/2*T1w*gz" ) )
import tensorflow as tf
ofn = os.path.expanduser("~/code/DPR/models/dsr3d_2up_64_256_6_3_v0.0zzz.h5")
if os.path.exists( ofn ):
    print("existing model") # should always initialize with pre-trained model
    mdl = tf.keras.models.load_model( ofn, compile=False )
else:
    print("default model - initialized with random weights")
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
image = ants.resample_image( image, [48,48,48] ) # downsample for speed in testing
test = siq.inference( image, mdl )
```

see also: the training scripts in `tests`.

## todo

1. numpy read/write

2. test/fix 2D

## your compute environment

```bash
export TF_ENABLE_ONEDNN_OPTS=1 # for CPU

total_cpu_cores=$(nproc)
number_sockets=$(($(grep "^physical id" /proc/cpuinfo | awk '{print $4}' | sort -un | tail -1)+1))
number_cpu_cores=$(( (total_cpu_cores/2) / number_sockets))

echo "number of CPU cores per socket: $number_cpu_cores";
echo "number of socket: $number_sockets";

echo "Physical cores:"
egrep '^core id' /proc/cpuinfo | sort -u | wc -l

echo "Logical cores:"

egrep '^processor' /proc/cpuinfo | sort -u | wc -l

echo "Physical cpus (separate chips):"

egrep '^physical id' /proc/cpuinfo | sort -u | wc -l

```

## to publish a release

```
rm -r -f build/ antspymm.egg-info/ dist/
python3 setup.py sdist bdist_wheel
python3 -m twine upload -u username -p password  dist/*
```
