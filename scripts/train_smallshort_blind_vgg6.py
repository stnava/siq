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

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

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
        "--dim",
        type=int,
        choices=[2, 3],
        default=3,
        help="Dimensionality (default 3). --dim 2 runs the identical blind pipeline, "
             "DBPN-small architecture and VGG layer-6 perceptual loss on 2D patches, "
             "validated on the head-cropped ANTs r16 slice, for fast iteration.",
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
        "--feature-type",
        choices=["vgg", "grader"],
        default="vgg",
        help="Perceptual feature extractor backend: vgg or grader (default: vgg)",
    )
    parser.add_argument(
        "--feature-layer",
        type=int,
        default=6,
        help="Feature layer index (default: 6)",
    )
    parser.add_argument(
        "--l1-weight",
        type=float,
        default=3.86,
        help="L1 (MAE) reconstruction loss weight (default: 3.86 for sharp edges)",
    )
    parser.add_argument(
        "--msq-weight",
        type=float,
        default=0.0,
        help="MSE reconstruction loss weight (default: 0.0)",
    )
    parser.add_argument(
        "--feat-weight",
        type=float,
        default=None,
        help="Perceptual loss weight (default: 1.14e-4 for vgg, 565.0 for grader)",
    )
    parser.add_argument(
        "--tv-weight",
        type=float,
        default=0.39,
        help="Total variation loss weight (default: 0.39)",
    )
    parser.add_argument(
        "--blur-lam",
        type=float,
        default=0.2,
        help="Poisson blur lambda (default: 0.2, biased toward minimal blur)",
    )
    parser.add_argument(
        "--blur-scale",
        type=float,
        default=0.5,
        help="Poisson blur scale factor (default: 0.5)",
    )
    parser.add_argument(
        "--val-image",
        type=str,
        default=None,
        help="Path to real MRI validation image (default: sub-BLAST022 or OASIS)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=50,
        help="Validation evaluation and HTML dashboard refresh frequency (default: 50 steps)",
    )
    parser.add_argument(
        "--no-html-report",
        action="store_true",
        help="Disable interactive HTML dashboard generation",
    )

    args = parser.parse_args()

    # Determine default feat_weight if not explicitly provided
    if args.feat_weight is None:
        feat_weight = 565.0 if args.feature_type == "grader" else 1.14e-4
    else:
        feat_weight = args.feat_weight

    # Normalize factor tuple
    if len(args.factor) == 1:
        factor_tuple = tuple([args.factor[0]] * args.dim)
    elif len(args.factor) == args.dim:
        factor_tuple = tuple(args.factor)
    elif args.dim == 2 and args.factor == [2, 2, 2]:
        factor_tuple = (2, 2)  # untouched 3D default
    else:
        raise ValueError(f"Invalid factor: {args.factor}. Must be 1 or {args.dim} values.")

    # 2D defaults: no 3D-transfer weights, validate on r16
    if args.dim == 2:
        if args.load_model == "dbpn_small_3d_from_refined.keras":
            args.load_model = None
        if args.val_image is None:
            args.val_image = "r16"

    print("=" * 70)
    print(f"Blind {args.dim}D Super-Resolution Training")
    print(f"  Model Type:        {args.model_type}")
    print(f"  Scaling Factor:    {factor_tuple}")
    print(f"  Feature Backend:   {args.feature_type} Layer {args.feature_layer}")
    print(f"  Warmup Iterations: {args.pretrain_iters} (pure MSE/L1)")
    print(f"  Perceptual Iters:  {args.iterations} ({args.feature_type.upper()} Layer {args.feature_layer} + L1 + TV)")
    print(f"  Batch Size:        {args.batch_size} (LR {args.lr_patch_size}^3 -> HR {[args.lr_patch_size * f for f in factor_tuple]})")
    print(f"  Loss Weights:      L1={args.l1_weight}, MSE={args.msq_weight}, Feat={feat_weight}, TV={args.tv_weight}")
    print(f"  Blur Settings:     Poisson(lam={args.blur_lam}, scale={args.blur_scale})")
    print(f"  Output Prefix:     {args.output_prefix}")
    print("=" * 70)

    # 1. Instantiate or Load Model Architecture
    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading transferred model from {args.load_model}...")
        model, _ = siq.load_siq_model(args.load_model)
    elif args.model_type == "dbpn-small":
        print(f"Instantiating Small {args.dim}D DBPN (option='small')...")
        model = siq.default_dbpn(
            strider=list(factor_tuple),
            dimensionality=args.dim,
            nChannelsIn=1,
            nChannelsOut=1,
            sigmoid_second_channel=False,
            option="small",
        )
    elif args.model_type == "dbpn-large":
        print("Instantiating Large/Default 3D DBPN (option='large', 66.86M parameters)...")
        model = siq.default_dbpn(
            strider=list(factor_tuple),
            dimensionality=args.dim,
            nChannelsIn=1,
            nChannelsOut=1,
            sigmoid_second_channel=False,
            option="large",
        )
    elif args.model_type == "espcn-residual":
        if args.dim != 3:
            raise ValueError("espcn-residual is 3D only")
        print("Instantiating Residual 3D ESPCN (sub-pixel blind model)...")
        model = siq.create_espcn_3d_residual(
            input_shape=(None, None, None, 1),
            factor=factor_tuple,
            n_filters=128,
            n_res_blocks=8,
        )

    print(f"Model instantiated successfully with {model.count_params():,} parameters.")

    # 2. Train using Kitchen Sink Blind SR Pipeline
    blur_sigma_cfg = {"type": "poisson", "lam": args.blur_lam, "scale": args.blur_scale}
    trained_model = siq.train_blind_sr_kitchen_sink(
        output_prefix=args.output_prefix,
        factor=factor_tuple,
        iterations=args.iterations,
        pretrain_iterations=args.pretrain_iters,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        lr_patch_size=args.lr_patch_size,
        feature_type=args.feature_type,
        feature_layer=args.feature_layer,
        model=model,
        msq_weight=args.msq_weight,
        l1_weight=args.l1_weight,
        feat_weight=feat_weight,
        tv_weight=args.tv_weight,
        use_cache=False,
        blur_sigma_range=blur_sigma_cfg,
        eval_freq=args.eval_freq,
        val_image=args.val_image,
        enable_html_report=not args.no_html_report,
        dimensionality=args.dim,
    )

    print("\nTraining complete!")
    print(f"Final model checkpoint: {args.output_prefix}_best.keras")
    print(f"Companion config:       {args.output_prefix}_best_config.json")


if __name__ == "__main__":
    main()
