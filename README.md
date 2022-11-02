# ANTsPyDPR

## deep perceptual resampling and super-resolution for (medical) imaging

install by calling (within the source directory):

```
python setup.py install
```

or install via `pip install antspydpr`

# what this will do

facilitates:

* creating training and testing data for deep networks

* generating and testing perceptual losses in 2D and 3D

* general training and inference functions for deep networks

* evaluation strategies for the above


# first time setup

```python
import antspydpr
antspydpr.get_data()
```

NOTE: `get_data` has a `force_download` option to make sure the latest
package data is installed.

# example processing

```python
import os
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"

import ants
import antspynet
import siq

strider = [2,2,2]
x, y, lin = siq.get_data( "HCPT1T2", "train", strider )
xte, yte, linte = siq.get_data( "HCPT1T2", "test", strider )

mdl = siq.default_dbpn( strider )

training_path, evaluation_results = siq.train( mdl, x, y, xte, yte, lin, lte )

siq.write_training( '/tmp/test_output', mdl, training_path,   
    evaluation_results )

image = ants.image_read( example_fn )
siq.inference( image, mdl )

```


## to publish a release

```
python3 -m build
python -m twine upload -u username -p password  dist/*
```
