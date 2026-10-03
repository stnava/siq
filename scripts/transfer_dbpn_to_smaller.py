#!/usr/bin/env python3
"""
transfer_dbpn_to_smaller.py

Transfers trained weights from a large/refined 3D DBPN (e.g. dbpn_3d_refined.keras)
to a smaller 3D DBPN (option='small', 9.88M parameters) via structured channel slicing.

Transfers:
  - 100% of Conv3D layers (38 intermediate projection units + final reconstruction layer)
  - 100% of PReLU layers (38 parametric activation units)
"""

import os
import argparse
import numpy as np
import keras
from keras import ops

import siq


def transfer_dbpn_to_smaller(src_model, dst_model, verbose=True):
    """
    Transfers weights from large DBPN to small DBPN via channel slicing.
    """
    src_convs = [l for l in src_model.layers if isinstance(l, keras.layers.Conv3D)]
    dst_convs = [l for l in dst_model.layers if isinstance(l, keras.layers.Conv3D)]
    src_prelu = [l for l in src_model.layers if isinstance(l, keras.layers.PReLU)]
    dst_prelu = [l for l in dst_model.layers if isinstance(l, keras.layers.PReLU)]

    if verbose:
        print(f"Source model: {len(src_convs)} Conv3D layers, {len(src_prelu)} PReLU layers ({src_model.count_params():,} params)")
        print(f"Target model: {len(dst_convs)} Conv3D layers, {len(dst_prelu)} PReLU layers ({dst_model.count_params():,} params)")

    transferred_conv = 0
    # 1. Intermediate Conv3D projection layers
    for i in range(len(dst_convs) - 1):
        s_l = src_convs[i]
        d_l = dst_convs[i]
        s_w = s_l.get_weights()
        d_w = d_l.get_weights()
        if len(s_w) > 0 and len(d_w) > 0:
            s_k, d_k = s_w[0], d_w[0]
            cin = min(s_k.shape[3], d_k.shape[3])
            cout = min(s_k.shape[4], d_k.shape[4])
            scale = np.sqrt(float(s_k.shape[3]) / float(cin))
            new_k = np.copy(d_k)
            new_k[:, :, :, :cin, :cout] = s_k[:, :, :, :cin, :cout] * scale
            new_w = [new_k]
            if len(d_w) > 1 and len(s_w) > 1:
                new_b = np.copy(d_w[1])
                new_b[:cout] = s_w[1][:cout]
                new_w.append(new_b)
            d_l.set_weights(new_w)
            transferred_conv += 1

    # 2. Final Reconstruction Conv3D layer
    s_last = src_convs[-1]
    d_last = dst_convs[-1]
    s_w, d_w = s_last.get_weights(), d_last.get_weights()
    cin = min(s_w[0].shape[3], d_w[0].shape[3])
    scale = float(s_w[0].shape[3]) / float(cin)
    new_k = np.copy(d_w[0])
    new_k[:, :, :, :cin, :] = s_w[0][:, :, :, :cin, :] * scale
    new_w = [new_k]
    if len(d_w) > 1 and len(s_w) > 1:
        new_w.append(s_w[1])
    d_last.set_weights(new_w)
    transferred_conv += 1

    # 3. PReLU Activation layers
    transferred_prelu = 0
    for i in range(min(len(src_prelu), len(dst_prelu))):
        s_l = src_prelu[i]
        d_l = dst_prelu[i]
        s_w = s_l.get_weights()
        d_w = d_l.get_weights()
        if len(s_w) > 0 and len(d_w) > 0:
            c = min(s_w[0].shape[-1], d_w[0].shape[-1])
            new_a = np.copy(d_w[0])
            new_a[:, :, :, :c] = s_w[0][:, :, :, :c]
            d_l.set_weights([new_a])
            transferred_prelu += 1

    if verbose:
        print(f"Transferred: {transferred_conv}/{len(dst_convs)} Conv3D layers and {transferred_prelu}/{len(dst_prelu)} PReLU layers.")

    return transferred_conv, transferred_prelu


def main():
    parser = argparse.ArgumentParser(description="Transfer weights from large DBPN to small DBPN")
    parser.add_argument("--source", type=str, default="dbpn_3d_refined.keras", help="Path to source trained model")
    parser.add_argument("--output", type=str, default="dbpn_small_3d_from_refined.keras", help="Path to save small model")
    parser.add_argument("--factor", type=int, nargs="+", default=[2, 2, 2], help="Scaling factor (default: 2 2 2)")
    args = parser.parse_args()

    factor_tuple = tuple(args.factor) if len(args.factor) == 3 else (args.factor[0], args.factor[0], args.factor[0])

    print("=" * 70)
    print("DBPN Weight Transfer: Large / Refined -> Small DBPN")
    print(f"  Source Model: {args.source}")
    print(f"  Target Model: {args.output}")
    print(f"  Factor:       {factor_tuple}")
    print("=" * 70)

    # 1. Load source model
    if not os.path.exists(args.source):
        raise FileNotFoundError(f"Source model not found at {args.source}")
    src_model = keras.models.load_model(args.source, compile=False)

    # 2. Build small DBPN
    dst_model = siq.default_dbpn(
        strider=list(factor_tuple),
        dimensionality=3,
        nChannelsIn=1,
        nChannelsOut=1,
        sigmoid_second_channel=False,
        option="small",
    )

    # 3. Transfer weights
    transfer_dbpn_to_smaller(src_model, dst_model, verbose=True)

    # 4. Save with companion config
    cfg = siq.default_siq_config(dst_model)
    cfg["model_type"] = "dbpn_small_3d"
    cfg["source_model"] = os.path.basename(args.source)
    cfg["transferred_from_params"] = src_model.count_params()
    cfg["transferred_to_params"] = dst_model.count_params()

    siq.save_siq_model(args.output, dst_model, cfg)
    print(f"\nSaved small DBPN model to: {args.output}")
    print(f"Saved companion config to: {args.output.replace('.keras', '_config.json')}")


if __name__ == "__main__":
    main()
