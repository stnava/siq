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
fns=glob.glob( os.path.expanduser( "~/.antspyt1w/2*T1w*gz" ) )[0:3]
mdl = siq.default_dbpn( [2,2,2] ) # should match ratio of high to low size patches
training_path, evaluation_results = siq.train(
    mdl,
    fns[0:3], # train files
    fns[0:3], # test files
    target_patch_size=[32,32,32],      # patch sizes at high res
    target_patch_size_low=[16,16,16],  # patch sizes at low res
    n_test=2,                          # number of test examples
    learning_rate=5e-05,               #
    feature_layer=6,                   #
    feature=2,
    tv=0.1,
    max_iterations=5,
    verbose=True)
siq.write_training( '/tmp/test_output', mdl, training_path, evaluation_results )
image = ants.image_read( example_fn )
siq.inference( image, mdl )

```


## to publish a release

```
rm -r -f build/ antspymm.egg-info/ dist/
python3 setup.py sdist bdist_wheel
python -m twine upload -u username -p password  dist/*
```
