#!/usr/bin/env python3
"""
train_smallshort_blind_vgg6.py

Mimics siq_smallshort_train_2x2x2_1chan_featgraderL6_best_mdl.h5 using:
  1. Our procedural Blind Super-Resolution generator (siq.blind_sr_generator)
  2. Pseudo-3D VGG19 Layer 6 perceptual feature loss (pseudo_3d_vgg_features_unbiased)
  3. Small 3D DBPN architecture (9.88M parameters, option='small')
  4. Two-phase short training schedule (200 warmup MSE steps + 800 VGG6 perceptual steps)
  5. Companion _config.json provenance saving via siq.save_siq_model()

Usage:
  python scripts/train_smallshort_blind_vgg6.py
  python scripts/train_smallshort_blind_vgg6.py --factor 2 --batch-size 2 --iterations 800
"""

import os
import sys
import argparse
import numpy as np
import keras
from keras import ops

import siq


def main():
    parser = argparse.ArgumentParser(
        description="Train Small 3D DBPN Blind SR with Pseudo-3D VGG19 Layer 6 perceptual loss"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="siq_smallshort_train_2x2x2_1chan_featvggL6_blind",
        help="Output prefix for model and config (default: siq_smallshort_train_2x2x2_1chan_featvggL6_blind)",
    )
    parser.add_argument(
        "--factor",
        type=int,
        nargs="+",
        default=[2, 2, 2],
        help="Super-resolution scaling factor (default: 2 2 2)",
    )
    parser.add_argument(
        "--pretrain-iters",
        type=int,
        default=200,
        help="Warmup MSE-only pretraining iterations (default: 200, matching smallshort_train)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=800,
        help="Main perceptual training iterations (default: 800, matching smallshort_train)",
    )
    parser.add_argument(
        "--lr-patch-size",
        type=int,
        default=32,
        help="Low-resolution cubic patch size (default: 32 -> 64x64x64 HR patch)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per training step (default: 2)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Adam optimizer learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--model-type",
        choices=["dbpn-small", "dbpn-large", "espcn-residual"],
        default="dbpn-small",
        help="Model architecture: dbpn-small (9.88M), dbpn-large (66.86M), or espcn-residual (default: dbpn-small)",
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default="dbpn_small_3d_from_refined.keras",
        help="Path to initial weights (e.g. transferred small model) (default: dbpn_small_3d_from_refined.keras if exists)",
    )
    parser.add_argument(
        "--msq-weight",
        type=float,
        default=3.86,
        help="MSE reconstruction loss weight (default: 3.86)",
    )
    parser.add_argument(
        "--feat-weight",
        type=float,
        default=1.14e-4,
        help="VGG19 Layer 6 perceptual loss weight (default: 1.14e-4)",
    )
    parser.add_argument(
        "--tv-weight",
        type=float,
        default=0.39,
        help="Total variation loss weight (default: 0.39)",
    )

    args = parser.parse_args()

    # Normalize factor tuple
    if len(args.factor) == 1:
        factor_tuple = (args.factor[0], args.factor[0], args.factor[0])
    elif len(args.factor) == 3:
        factor_tuple = tuple(args.factor)
    else:
        raise ValueError(f"Invalid factor: {args.factor}. Must be 1 or 3 values.")

    print("=" * 70)
    print("Blind 3D Super-Resolution Training (VGG19 Layer 6)")
    print(f"  Model Type:        {args.model_type}")
    print(f"  Scaling Factor:    {factor_tuple}")
    print(f"  Warmup Iterations: {args.pretrain_iters} (pure MSE)")
    print(f"  Perceptual Iters:  {args.iterations} (VGG19 Layer 6 + MSE + TV)")
    print(f"  Batch Size:        {args.batch_size} (LR {args.lr_patch_size}^3 -> HR {[args.lr_patch_size * f for f in factor_tuple]}^3)")
    print(f"  Loss Weights:      MSE={args.msq_weight}, Feat={args.feat_weight}, TV={args.tv_weight}")
    print(f"  Output Prefix:     {args.output_prefix}")
    print("=" * 70)

    # 1. Instantiate or Load Model Architecture
    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading transferred model from {args.load_model}...")
        model, _ = siq.load_siq_model(args.load_model)
    elif args.model_type == "dbpn-small":
        print("Instantiating Small 3D DBPN (option='small', 9.88M parameters)...")
        model = siq.default_dbpn(
            strider=list(factor_tuple),
            dimensionality=3,
            nChannelsIn=1,
            nChannelsOut=1,
            sigmoid_second_channel=False,
            option="small",
        )
    elif args.model_type == "dbpn-large":
        print("Instantiating Large/Default 3D DBPN (option='large', 66.86M parameters)...")
        model = siq.default_dbpn(
            strider=list(factor_tuple),
            dimensionality=3,
            nChannelsIn=1,
            nChannelsOut=1,
            sigmoid_second_channel=False,
            option="large",
        )
    elif args.model_type == "espcn-residual":
        print("Instantiating Residual 3D ESPCN (sub-pixel blind model)...")
        model = siq.create_espcn_3d_residual(
            input_shape=(None, None, None, 1),
            factor=factor_tuple,
            n_filters=128,
            n_res_blocks=8,
        )

    print(f"Model instantiated successfully with {model.count_params():,} parameters.")

    # 2. Train using Kitchen Sink Blind SR Pipeline with VGG19 Layer 6
    trained_model = siq.train_blind_sr_kitchen_sink(
        output_prefix=args.output_prefix,
        factor=factor_tuple,
        iterations=args.iterations,
        pretrain_iterations=args.pretrain_iters,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        lr_patch_size=args.lr_patch_size,
        feature_type="vgg",
        feature_layer=6,
        model=model,
        msq_weight=args.msq_weight,
        feat_weight=args.feat_weight,
        tv_weight=args.tv_weight,
        use_cache=False,
    )

    print("\nTraining complete!")
    print(f"Final model checkpoint: {args.output_prefix}_best.keras")
    print(f"Companion config:       {args.output_prefix}_best_config.json")


if __name__ == "__main__":
    main()
