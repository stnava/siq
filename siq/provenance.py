import os
import json
import glob
import numpy as np
import keras


def read_srmodel(srfilename, custom_objects=None):  # pragma: no cover
    """
    Load a super-resolution model (h5, .keras, or SavedModel format),
    and determine its upsampling factor.

    Parameters
    ----------
    srfilename : str
        Path to the model file (.h5, .keras, or a SavedModel folder).
    custom_objects : dict, optional
        Dictionary of custom objects used in the model.

    Returns
    -------
    model : keras.Model
        The loaded model.
    upsampling_factor : list of int
        List describing the upsampling factor:
        - For 3D input: [x_up, y_up, z_up, channels]
        - For 2D input: [x_up, y_up, channels]
    """
    srfilename = os.path.expanduser(srfilename)
    ext = os.path.splitext(srfilename)[1].lower()

    if os.path.isdir(srfilename):
        model = keras.models.load_model(srfilename, custom_objects=custom_objects, compile=False)
    elif ext in ['.h5', '.keras']:
        model = keras.models.load_model(srfilename, custom_objects=custom_objects, compile=False)
    else:
        raise ValueError(f"Unsupported model format: {ext}")

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    chanindex = 3 if len(input_shape) == 4 else 4
    nchan = int(input_shape[chanindex])

    try:
        if len(input_shape) == 5:  # 3D
            dummy_input = np.zeros([1, 8, 8, 8, nchan])
        else:  # 2D
            dummy_input = np.zeros([1, 8, 8, nchan])

        try:
            output = model(dummy_input)
        except Exception:
            output = model({model.input_names[0]: dummy_input})

        outshp = output.shape
        if len(input_shape) == 5:
            return model, [int(outshp[1] / 8), int(outshp[2] / 8), int(outshp[3] / 8), nchan]
        else:
            return model, [int(outshp[1] / 8), int(outshp[2] / 8), nchan]

    except Exception as e:
        raise RuntimeError(f"Could not infer upsampling factor. Error: {e}")


def save_siq_model(  # pragma: no cover
    model_path,
    model,
    config,
    loss_weights=None,
    archive=False,
    tag="",
    notes="",
    verbose=True,
):
    """Save a siq model and its mandatory provenance config together.

    Always use this instead of ``model.save()`` directly so that the
    companion ``_config.json`` is always present alongside the weights.

    Parameters
    ----------
    model_path : str
        Destination path, must end in ``.keras``.
    model : keras.Model
        Trained model to save.
    config : dict
        Provenance dict as returned by :func:`load_siq_model` or built by
        the training reporter. Must contain at minimum:
        ``normalization``, ``input_patch_shape``, ``inference``.
    loss_weights : dict, optional
        Dictionary of loss weights at save time to store in provenance config.
    archive : bool
        If ``True``, also copy the model into the immutable ``model_archive/``
        directory and register it in ``model_registry.json``. Default ``False``.
    tag : str
        Human-readable label for the registry entry. Only used when ``archive=True``.
    notes : str
        Free-form notes stored in the registry entry. Only used when ``archive=True``.
    verbose : bool
        Print save path.
    """
    model.save(model_path)
    if loss_weights is not None:
        config["loss_weights"] = {k: float(v) for k, v in loss_weights.items()}
    config_path = model_path.replace('.keras', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    if verbose:
        print(f"Saved model:  {model_path}")
        print(f"Saved config: {config_path}")
    if archive:
        from .model_registry import archive_model as _archive
        _archive(model_path, config, tag=tag, notes=notes, verbose=verbose)


def load_siq_model(model_path, custom_objects=None, verbose=True):  # pragma: no cover
    """Load a siq model AND its mandatory provenance config.

    Parameters
    ----------
    model_path : str
        Path to the ``.keras`` file.
    custom_objects : dict, optional
        Custom Keras objects required to deserialize the model.
    verbose : bool
        Print loaded shapes and normalization method.

    Returns
    -------
    model : keras.Model
        The loaded model, ready for inference.
    config : dict
        Provenance dict.
    """
    model_path = os.path.expanduser(model_path)
    config_path = model_path.replace('.keras', '_config.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No provenance config found at:\n  {config_path}\n"
            f"Cannot safely run inference without knowing normalization method.\n"
            f"Use siq.save_siq_model(path, model, config) to save models with provenance,\n"
            f"or manually create a _config.json using the siq.default_siq_config() template."
        )

    with open(config_path) as f:
        config = json.load(f)

    keras.config.enable_unsafe_deserialization()
    model = keras.models.load_model(
        model_path, custom_objects=custom_objects, compile=False, safe_mode=False
    )

    if verbose:
        print(f"[siq] Loaded model: {model_path}")
        print(f"  Input patch:    {config.get('input_patch_shape')}")
        print(f"  Output patch:   {config.get('output_patch_shape')}")
        print(f"  Upsample:       {config.get('upsample_factor')}×")
        if 'normalization' in config:
            print(f"  Normalization:  {config['normalization'].get('method')} → {config['normalization'].get('output_range')}")
        if 'val_metrics' in config:
            m = config['val_metrics']
            print(f"  Val metrics:    PSNR={m.get('val_psnr', float('nan')):.3f} "
                  f"SSIM={m.get('val_ssim', float('nan')):.4f} "
                  f"GMSD={m.get('val_gmsd', float('nan')):.4f} "
                  f"HFEN={m.get('val_hfen', float('nan')):.4f}")
        if 'loss_weights' in config:
            lw = config['loss_weights']
            print(f"  Loss weights:   L1={lw.get('l1', 0.0):.4f} "
                  f"Feat={lw.get('feat', 0.0):.2e} "
                  f"TV={lw.get('tv', 0.0):.4f} "
                  f"GMS={lw.get('gms', 0.0):.2f} "
                  f"CBI={lw.get('cbi', 0.0):.2f}")
    return model, config


def default_siq_config(model=None):  # pragma: no cover
    """Return a default provenance config dict, optionally populated from a model."""
    cfg = {
        "model_type": "asdbpn_3d",
        "siq_version": "unknown",
        "saved_at": "unknown",
        "input_patch_shape": [64, 64, 64, 1],
        "output_patch_shape": [128, 128, 128, 1],
        "upsample_factor": 2,
        "normalization": {
            "method": "volume",
            "truncate_quantiles": [0.001, 0.999],
            "output_range": [0.0, 1.0],
            "note": (
                "Apply ants.iMath(vol,'TruncateIntensity',0.001,0.999) then "
                "ants.iMath(vol,'Normalize') to the WHOLE volume. "
                "Do NOT normalize per-patch."
            )
        },
        "inference": {
            "preferred_method": "direct_single_patch_if_fits",
            "patch_overlap": 16,
            "output_clip": [0.0, 1.0],
            "antspynet_wrapper": False,
            "anti_checkerboard": "auto",
            "anti_checkerboard_sigma": "auto",
        },
    }
    if model is not None:
        in_s = list(model.input_shape[1:])
        out_s = list(model.output_shape[1:])
        ndim = len(in_s) - 1
        if any(s is None for s in in_s[:-1]):
            cfg["input_patch_shape"] = [64] * ndim + [in_s[-1] if in_s[-1] is not None else 1]
        else:
            cfg["input_patch_shape"] = in_s
        if any(s is None for s in out_s[:-1]):
            cfg["output_patch_shape"] = [128] * ndim + [out_s[-1] if out_s[-1] is not None else 1]
        else:
            cfg["output_patch_shape"] = out_s
        try:
            dim_factors = []
            for o, i in zip(out_s[:-1], in_s[:-1]):
                if o is not None and i is not None:
                    dim_factors.append(int(round(o / i)))
            if dim_factors and len(dim_factors) == len(out_s) - 1:
                cfg["upsample_factor"] = dim_factors[0] if len(set(dim_factors)) == 1 else dim_factors
            else:
                ndim = len(in_s) - 1
                dummy_shape = [1] + [8] * ndim + [in_s[-1] if in_s[-1] is not None else 1]
                dummy = keras.ops.zeros(dummy_shape)
                out_dummy = model(dummy)
                dim_factors = [int(round(out_dummy.shape[d + 1] / dummy.shape[d + 1])) for d in range(ndim)]
                cfg["upsample_factor"] = dim_factors[0] if len(set(dim_factors)) == 1 else dim_factors
        except Exception:
            cfg["upsample_factor"] = 2
    return cfg


def extract_siq_loss_weights(source, verbose=True):
    """
    Extract balanced training loss weights from a saved siq model, its companion
    _config.json, or an associated training weights CSV.
    """
    import pandas as pd

    weights = None

    if isinstance(source, dict):
        if "loss_weights" in source and isinstance(source["loss_weights"], dict):
            weights = dict(source["loss_weights"])
    elif isinstance(source, str):
        src_path = os.path.expanduser(source)
        cfg_path = None
        if src_path.endswith(".keras"):
            cfg_path = src_path.replace(".keras", "_config.json")
        elif src_path.endswith(".json"):
            cfg_path = src_path

        if cfg_path and os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
                if "loss_weights" in cfg and isinstance(cfg["loss_weights"], dict):
                    weights = dict(cfg["loss_weights"])
            except Exception:
                pass

        if weights is None:
            search_dirs = [
                os.path.dirname(src_path),
                os.path.join(os.path.dirname(src_path), ".."),
                os.getcwd()
            ]
            for d in search_dirs:
                if not d or not os.path.exists(d):
                    continue
                csv_candidates = glob.glob(os.path.join(d, "*refined_training_weights.csv"))
                for c in csv_candidates:
                    try:
                        df = pd.read_csv(c)
                        if "l1" in df.columns:
                            row = df.iloc[-1]
                            weights = {col: float(row[col]) for col in df.columns if pd.notnull(row[col])}
                            break
                    except Exception:
                        pass
                if weights is not None:
                    break

    if weights is not None:
        normalized = {
            "l1": float(weights.get("l1", 0.0)),
            "feat": float(weights.get("feat", 0.0)),
            "tv": float(weights.get("tv", 0.0)),
            "msq": float(weights.get("msq", 0.0)),
            "edge": float(weights.get("edge", 0.0)),
            "gms": float(weights.get("gms", 0.0)),
            "cbi": float(weights.get("cbi", 0.0)),
        }
        if verbose:
            print(f"[siq] Extracted loss weights: L1={normalized['l1']:.6f}, "
                  f"Feat={normalized['feat']:.6e}, TV={normalized['tv']:.6f}, "
                  f"GMS={normalized['gms']:.4f}, CBI={normalized['cbi']:.4f}, Edge={normalized['edge']:.4f}")
        return normalized

    if verbose:
        print("[siq] No saved loss weights found in source model or companion files.")
    return None
