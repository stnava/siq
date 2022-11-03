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

* evaluation strategies for the above

# first time setup

```python
import siq
siq.get_data()
```

NOTE: `get_data` has a `force_download` option to make sure the latest
package data is installed.

# example processing

```python
import os
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"
import glob
import siq
fns=glob.glob( os.path.expanduser( "~/.antspyt1w/2*T1w*gz" ) ) # antspyt1w.get_data
strider = [2,2,2]
psz = [32,32,32]
pszlo = [16,16,16]
x,y = siq.image_patch_training_data_from_filenames(
    filenames=fns,
    target_patch_size=psz,
    target_patch_size_low=pszlo,
    nPatches = 3,
    istest   = False,
    patch_scaler=True,
    verbose = True )

xte,yte = siq.image_patch_training_data_from_filenames(
    filenames=fns,
    target_patch_size=psz,
    target_patch_size_low=pszlo,
    nPatches = 3,
    istest   = False,
    patch_scaler=True,
    verbose = True )

# write these to numpy - then we can train in a reproducible way

mdl = siq.default_dbpn( strider )

training_path, evaluation_results = siq.train( mdl, x, y, xte, yte, lin, lte )

siq.write_training( '/tmp/test_output', mdl, training_path,   
    evaluation_results )

image = ants.image_read( example_fn )
siq.inference( image, mdl )

```


## to publish a release

```
rm -r -f build/ antspymm.egg-info/ dist/
python3 setup.py sdist bdist_wheel
python -m twine upload -u username -p password  dist/*
```
