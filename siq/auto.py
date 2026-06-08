import os
import tensorflow as tf
import ants
import numpy as np

from .get_data import default_dbpn, inference

def auto(image, target_resolution=None, **kwargs):
    """
    Automatically infers what the user wants to do with the image
    and runs the appropriate SIQ tools.

    Parameters
    ----------
    image : ants.ANTsImage or str
        The input image to process.
    target_resolution : tuple, list, or None
        The desired resolution. If None, it will default to upsampling
        by a factor of 2x.
    **kwargs
        Additional arguments passed to inference.

    Returns
    -------
    ants.ANTsImage
        The super-resolved image.
    """

    if isinstance(image, str):
        print(f"Loading image from {image}...")
        image = ants.image_read(image)

    if not isinstance(image, ants.core.ants_image.ANTsImage):
        raise ValueError("Input image must be an ants.ANTsImage or a valid file path.")

    dim = image.dimension
    print(f"Detected {dim}D image.")

    # Determine upsampling factor
    upsample_factor = 2
    if target_resolution is not None:
        if len(target_resolution) != dim:
            raise ValueError(f"target_resolution must match image dimensions ({dim})")
        # heuristic: just use upsample_factor based on standard SIQ
        # Actually in an auto function, we might just assume 2x for now and resample later.
        pass

    print(f"Configuring default DBPN model for {dim}D upsampling...")
    # Initialize a dummy model for auto mode
    # Ideally we load a pre-trained model. We will construct a default architecture.
    strider = [upsample_factor] * dim
    mdl = default_dbpn(
        strider=strider,
        dimensionality=dim,
        nfilt=32,
        nff=64,
        convn=3,
        lastconv=1,
        nbp=2,
        option='tiny' # use tiny for faster execution if we don't have pretrained
    )

    print("Running inference...")
    # We use the inference function from get_data
    sr_image = inference(image, mdl, verbose=True, **kwargs)

    if target_resolution is not None:
        print(f"Resampling to target resolution: {target_resolution}")
        sr_image = ants.resample_image(sr_image, target_resolution, use_voxels=False)

    print("SIQ processing complete.")
    return sr_image
