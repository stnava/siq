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

## problems reading pre-trained models

see `tests/translate_models_to_keras3.py` for some insights into handling hdf5 reading with different versions of keras or tensorflow.

## starter models 

[link](https://figshare.com/articles/software/SIQ_reference_super_resolution_models/27079987) to models

note - may be issues loading/reading - see the comments about keras versions above

currently -- with tf >= 2.17 -- this works:

1. run `tests/export_legacy_keras_weights.py` on the legacy h5 files.

2. run `tests/migrate_and_resave_models.py` to get the new `.keras` models.

```python
import os
seed=4
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_USE_LEGACY_KERAS'] = '0'
mfn=os.path.expanduser('~/.antspymm/siq_smallshort_train_1x1x2_1chan_featgraderL6_best.keras')
import siq
a, b = siq.read_srmodel(mfn)
```


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

```bash
rm -r -f build/ antspymm.egg-info/ dist/
python3 -m  build .
python3 -m twine upload --repository siq dist/*
```


## notes on cpu environment

```
# dd=/home/ubuntu/miniconda3/condabin/conda
# conda update -n base -c defaults conda
# conda init bash
# conda create -n ai3 python=3.9
# conda activate ai3 
# pip3 install --upgrade pip
py=python3 # "sudo /opt/parallelcluster/pyenv/versions/3.7.10/envs/awsbatch_virtualenv/bin/python3.7"

$py -m pip install --upgrade pip

# python3.7 -m pip uninstall tensorflow antspynet dipy patsy tensorboard tensorflow-probability -y
$py -m pip install nibabel PyNomaly scipy 
$py -m pip install antspyx 
$py -m pip install dipy 
$py -m pip install antspyt1w 
$py -m pip install antspymm 
$py -m pip install antspynet
$py -m pip install siq
$py -m pip uninstall tensorflow -y
$py -m pip install intel-tensorflow # -avx512==2.9.1
$py -m pip install tensorflow_probability
$py -m pip install keras
```
