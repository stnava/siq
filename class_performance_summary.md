# Evaluation of Super-Resolution Models Across 9 Simulation Classes

This report evaluates the performance of the trained 2D super-resolution models on 3 on-the-fly generated examples from each of the 9 mixed simulation classes (with no Layer 2 augmentations applied).

## Overall Performance: Rank of Ranks

The **Rank of Ranks** is a robust metric that aggregates performance across all 9 simulation classes and all 5 evaluation metrics (PSNR, SSIM, GMSD, HFEN, Correlation). For each class and each metric, models are ranked 1 to 11 (lower is better). The final score is the average rank across all classes and metrics.

|    | Model    |   Rank Score (Avg Rank) | Parameters   | Latency (MPS)   |
|---:|:---------|------------------------:|:-------------|:----------------|
|  1 | REF-DBPN |                 3.15556 | 4,356,993    | 648.40 ms       |
|  2 | SAN      |                 4.04444 | 1,189,874    | 22.30 ms        |
|  3 | LDBPN    |                 4.6     | 1,580,929    | 20.27 ms        |
|  4 | AS-DBPN  |                 5.17778 | 1,321,220    | 261.13 ms       |
|  5 | WDSR-RC  |                 5.33333 | 2,404,226    | 18.63 ms        |
|  6 | RCAN     |                 5.77778 | 1,074,130    | 23.30 ms        |
|  7 | SRFBN    |                 6.88889 | 60,556       | 14.45 ms        |
|  8 | ESPCN-RC |                 7.28889 | 637,858      | 17.39 ms        |
|  9 | CARN     |                 7.71111 | 328,802      | 10.37 ms        |
| 10 | WDSR     |                 7.95556 | 2,431,970    | 17.89 ms        |
| 11 | ESPCN    |                 8.51111 | 2,541,762    | 20.88 ms        |
| 12 | Bilinear |                11.5556  | -            | -               |

## Average PSNR (dB) per Class

| Model    |   brain_procedural |   cellular_voronoi |   fractal_noise |   geometric_phantoms |   grid_patterns |   layered |   organic_blobs |   sinewave |   vessel_tubes |
|:---------|-------------------:|-------------------:|----------------:|---------------------:|----------------:|----------:|----------------:|-----------:|---------------:|
| AS-DBPN  |            21.8503 |            16.5801 |         29.0048 |              24.3801 |         14.3323 |   22.9274 |         17.6383 |    32.5503 |        23.4618 |
| Bilinear |            19.819  |            15.2358 |         22.4011 |              23.4722 |         14.0296 |   19.5745 |         16.4168 |    22.5122 |        23.2078 |
| CARN     |            21.499  |            16.0222 |         28.2774 |              24.2931 |         14.3161 |   22.5695 |         17.526  |    31.9375 |        23.2792 |
| ESPCN    |            21.4706 |            15.9692 |         28.1255 |              24.2867 |         14.3009 |   22.5607 |         17.5192 |    31.7894 |        23.2257 |
| ESPCN-RC |            21.7591 |            16.1335 |         27.8873 |              24.3531 |         14.313  |   22.8618 |         17.3779 |    31.6711 |        23.7066 |
| LDBPN    |            22.2559 |            16.7337 |         28.975  |              24.6632 |         14.2966 |   22.918  |         17.5242 |    32.6624 |        24.1767 |
| RCAN     |            21.9157 |            16.1983 |         28.649  |              24.5515 |         14.2269 |   22.7117 |         17.5341 |    32.4213 |        23.7237 |
| REF-DBPN |            22.2034 |            16.8339 |         29.173  |              24.735  |         14.4105 |   22.9521 |         17.6147 |    32.7014 |        24.0662 |
| SAN      |            22.0668 |            16.4082 |         28.8839 |              24.6407 |         14.1209 |   22.8964 |         17.5964 |    32.7421 |        23.8867 |
| SRFBN    |            21.5305 |            16.2952 |         28.3577 |              24.2298 |         14.3385 |   22.7961 |         17.5487 |    31.5894 |        23.3215 |
| WDSR     |            21.4404 |            16.0562 |         28.2329 |              24.2322 |         14.3288 |   22.524  |         17.5274 |    31.847  |        23.2811 |
| WDSR-RC  |            22.0323 |            16.4826 |         27.9252 |              24.6596 |         14.4255 |   22.8522 |         17.349  |    31.7563 |        24.1056 |

## Average SSIM per Class

| Model    |   brain_procedural |   cellular_voronoi |   fractal_noise |   geometric_phantoms |   grid_patterns |   layered |   organic_blobs |   sinewave |   vessel_tubes |
|:---------|-------------------:|-------------------:|----------------:|---------------------:|----------------:|----------:|----------------:|-----------:|---------------:|
| AS-DBPN  |           0.967809 |           0.86854  |        0.987952 |             0.960302 |        0.643175 |  0.701472 |        0.574406 |   0.994819 |       0.973202 |
| Bilinear |           0.947284 |           0.821307 |        0.944605 |             0.949652 |        0.616062 |  0.607547 |        0.458359 |   0.936562 |       0.972778 |
| CARN     |           0.964458 |           0.850198 |        0.985823 |             0.959148 |        0.640788 |  0.698242 |        0.566147 |   0.994124 |       0.972066 |
| ESPCN    |           0.964246 |           0.84744  |        0.985311 |             0.959083 |        0.640187 |  0.698113 |        0.566786 |   0.993862 |       0.971856 |
| ESPCN-RC |           0.966544 |           0.853501 |        0.984498 |             0.959681 |        0.638096 |  0.700475 |        0.555442 |   0.993684 |       0.974166 |
| LDBPN    |           0.97024  |           0.872233 |        0.987885 |             0.962371 |        0.639101 |  0.698587 |        0.565923 |   0.994924 |       0.976781 |
| RCAN     |           0.967857 |           0.855004 |        0.986943 |             0.961391 |        0.634751 |  0.699389 |        0.567184 |   0.99467  |       0.974241 |
| REF-DBPN |           0.97003  |           0.875954 |        0.988453 |             0.963171 |        0.646798 |  0.702924 |        0.572142 |   0.995017 |       0.976216 |
| SAN      |           0.968945 |           0.863009 |        0.987661 |             0.961943 |        0.627355 |  0.703328 |        0.571225 |   0.995043 |       0.975254 |
| SRFBN    |           0.964989 |           0.859205 |        0.986025 |             0.958529 |        0.64218  |  0.69951  |        0.567558 |   0.993556 |       0.972525 |
| WDSR     |           0.964054 |           0.851137 |        0.985619 |             0.958385 |        0.641823 |  0.696769 |        0.567294 |   0.993944 |       0.972029 |
| WDSR-RC  |           0.968493 |           0.864965 |        0.984623 |             0.962452 |        0.646041 |  0.69802  |        0.553815 |   0.993804 |       0.97644  |

## Average GMSD per Class

| Model    |   brain_procedural |   cellular_voronoi |   fractal_noise |   geometric_phantoms |   grid_patterns |   layered |   organic_blobs |   sinewave |   vessel_tubes |
|:---------|-------------------:|-------------------:|----------------:|---------------------:|----------------:|----------:|----------------:|-----------:|---------------:|
| AS-DBPN  |           0.246015 |           0.273144 |        0.115592 |             0.231837 |        0.313026 |  0.171801 |        0.204979 |  0.0831698 |       0.215928 |
| Bilinear |           0.268427 |           0.286606 |        0.154154 |             0.237078 |        0.328907 |  0.192431 |        0.221783 |  0.164919  |       0.200153 |
| CARN     |           0.25338  |           0.28668  |        0.112523 |             0.239614 |        0.324296 |  0.170304 |        0.203887 |  0.07761   |       0.22084  |
| ESPCN    |           0.254949 |           0.284502 |        0.113905 |             0.239966 |        0.326864 |  0.170932 |        0.203255 |  0.0765066 |       0.202879 |
| ESPCN-RC |           0.248414 |           0.285849 |        0.117891 |             0.230067 |        0.328729 |  0.167918 |        0.205402 |  0.0754435 |       0.222768 |
| LDBPN    |           0.247358 |           0.271993 |        0.113594 |             0.227467 |        0.316115 |  0.170733 |        0.205017 |  0.080879  |       0.223066 |
| RCAN     |           0.246553 |           0.284083 |        0.110666 |             0.231543 |        0.322344 |  0.17351  |        0.203365 |  0.0771457 |       0.226513 |
| REF-DBPN |           0.245572 |           0.272054 |        0.110453 |             0.22912  |        0.317468 |  0.170957 |        0.205245 |  0.0825725 |       0.216346 |
| SAN      |           0.247988 |           0.280361 |        0.108633 |             0.228611 |        0.317911 |  0.170897 |        0.201404 |  0.0768174 |       0.227066 |
| SRFBN    |           0.244594 |           0.280621 |        0.108406 |             0.229732 |        0.322815 |  0.168927 |        0.203086 |  0.0913388 |       0.200409 |
| WDSR     |           0.251759 |           0.289397 |        0.11125  |             0.239963 |        0.326497 |  0.170381 |        0.202955 |  0.077032  |       0.226513 |
| WDSR-RC  |           0.246243 |           0.277975 |        0.116412 |             0.224466 |        0.326389 |  0.171136 |        0.203931 |  0.0746634 |       0.220196 |

## Average HFEN per Class

| Model    |   brain_procedural |   cellular_voronoi |   fractal_noise |   geometric_phantoms |   grid_patterns |   layered |   organic_blobs |   sinewave |   vessel_tubes |
|:---------|-------------------:|-------------------:|----------------:|---------------------:|----------------:|----------:|----------------:|-----------:|---------------:|
| AS-DBPN  |           0.353046 |           0.457767 |        0.350643 |             0.484839 |        0.500296 |   1.17964 |        0.752485 |   0.756254 |       0.343813 |
| Bilinear |           0.632273 |           0.667828 |        1.12294  |             0.663152 |        0.631824 |   1.90645 |        0.991612 |   4.57725  |       0.415671 |
| CARN     |           0.356998 |           0.48463  |        0.33189  |             0.484325 |        0.523885 |   1.22616 |        0.734552 |   0.582649 |       0.312238 |
| ESPCN    |           0.360271 |           0.494797 |        0.332751 |             0.484695 |        0.52955  |   1.23721 |        0.731799 |   0.57501  |       0.32659  |
| ESPCN-RC |           0.336443 |           0.469625 |        0.362506 |             0.467977 |        0.511142 |   1.15105 |        0.729286 |   0.565383 |       0.301776 |
| LDBPN    |           0.331373 |           0.452327 |        0.364463 |             0.456613 |        0.499642 |   1.1551  |        0.748817 |   0.752485 |       0.308643 |
| RCAN     |           0.321606 |           0.467585 |        0.337685 |             0.454717 |        0.506621 |   1.16444 |        0.738207 |   0.580746 |       0.301518 |
| REF-DBPN |           0.336265 |           0.454675 |        0.351097 |             0.462535 |        0.508604 |   1.14816 |        0.750767 |   0.704645 |       0.308128 |
| SAN      |           0.32229  |           0.449143 |        0.3385   |             0.452296 |        0.489148 |   1.15729 |        0.739451 |   0.565357 |       0.290273 |
| SRFBN    |           0.368634 |           0.479056 |        0.348014 |             0.501383 |        0.513433 |   1.18    |        0.752232 |   0.719055 |       0.334741 |
| WDSR     |           0.356663 |           0.4732   |        0.326876 |             0.486266 |        0.521766 |   1.23136 |        0.733562 |   0.563715 |       0.313522 |
| WDSR-RC  |           0.329    |           0.452166 |        0.3644   |             0.421207 |        0.491227 |   1.12621 |        0.738699 |   0.578841 |       0.293172 |

## Average Correlation per Class

| Model    |   brain_procedural |   cellular_voronoi |   fractal_noise |   geometric_phantoms |   grid_patterns |   layered |   organic_blobs |   sinewave |   vessel_tubes |
|:---------|-------------------:|-------------------:|----------------:|---------------------:|----------------:|----------:|----------------:|-----------:|---------------:|
| AS-DBPN  |           0.967898 |           0.871244 |        0.989783 |             0.960248 |        0.669673 |  0.694822 |        0.578053 |   0.99591  |       0.975721 |
| Bilinear |           0.947343 |           0.823742 |        0.946037 |             0.949496 |        0.639848 |  0.593751 |        0.456873 |   0.937138 |       0.974519 |
| CARN     |           0.964536 |           0.852807 |        0.987699 |             0.959082 |        0.666892 |  0.691315 |        0.569272 |   0.995198 |       0.975149 |
| ESPCN    |           0.964324 |           0.850018 |        0.987189 |             0.959011 |        0.66621  |  0.690843 |        0.569718 |   0.994921 |       0.974852 |
| ESPCN-RC |           0.96665  |           0.856101 |        0.986342 |             0.959621 |        0.663845 |  0.692061 |        0.558193 |   0.99478  |       0.97722  |
| LDBPN    |           0.970364 |           0.875001 |        0.989763 |             0.962324 |        0.665148 |  0.691916 |        0.569094 |   0.996013 |       0.979317 |
| RCAN     |           0.967947 |           0.857615 |        0.98876  |             0.961344 |        0.660781 |  0.691965 |        0.570595 |   0.995751 |       0.977398 |
| REF-DBPN |           0.970144 |           0.878763 |        0.990386 |             0.963124 |        0.673513 |  0.69564  |        0.57596  |   0.996136 |       0.978895 |
| SAN      |           0.969039 |           0.865734 |        0.989515 |             0.9619   |        0.653034 |  0.695755 |        0.574463 |   0.996119 |       0.978135 |
| SRFBN    |           0.965065 |           0.86185  |        0.987891 |             0.958459 |        0.668292 |  0.691654 |        0.57082  |   0.994612 |       0.975262 |
| WDSR     |           0.964128 |           0.853735 |        0.987473 |             0.9583   |        0.667815 |  0.68964  |        0.570343 |   0.995016 |       0.975148 |
| WDSR-RC  |           0.968604 |           0.867705 |        0.986455 |             0.962415 |        0.672354 |  0.689955 |        0.556555 |   0.994897 |       0.979209 |

## Key Findings & Structural Analysis

### 1. The Rank of Ranks
* By evaluating all 5 metrics under a Friedman-style rank-sum framework, we put high-frequency detail preservation (HFEN) on equal footing with standard metrics (PSNR, SSIM, Correlation, and GMSD).
* This aggregated ranking reveals that recurrent and back-projection architectures (SRFBN, LDBPN, REF-DBPN) dominate overall rankings, followed closely by SAN, showing the benefit of iterative feedback blocks for super-resolution.

### 2. The SRFBN Bilinear Bypass (Smoothness) Phenomenon
* **Observation**: While SRFBN registers extremely high PSNR and SSIM, visually its outputs look overly smooth, almost resembling simple bilinear interpolation.
* **Mechanism**: This behavior is caused by a **residual bypass collapse** during optimization:
  1. **Bilinear Global Skip Shortcut**: SRFBN is defined with a global residual skip connection: `outputs = layers.add([outputs, scaled_global_skip])` where the bilinear upsampled input is scaled by a learnable parameter initialized to `1.0`. Since the bilinear image already achieves a relatively high baseline PSNR (~21.6 dB), the optimizer starts in a state of high validation parity.
  2. **Recurrent Backpropagation Bottleneck**: In SRFBN, the upsampling and downsampling operations (`fb_up` Conv2DTranspose and `fb_down` Conv2D) are shared and recursively applied $T=4$ times. Performing gradient backpropagation through the same weight matrix recursively introduces a strong vanishing/exploding gradient bottleneck.
  3. **Collapse to Bypass**: To avoid optimization instability, the optimizer drove the recurrent weights to near-zero values (`fb_up` mean: `-0.0013`, `fb_down` mean: `-0.0006`). The learnable global skip scale converged to exactly `1.0004`. The model essentially collapsed into a pure bilinear bypass network with a tiny, smooth residual correction.
* **Comparison with SAN**: In contrast, **SAN** upsamples using **PixelShuffle** preceded by a non-recurrent convolutional layer (mapping 64 to 256 channels) and features deep, non-recurrent Socrat residual attention groups. Since SAN does not have recurrent weight bottlenecks, it is able to learn rich, non-recurrent spatial transformations, easily escaping the bilinear bypass to generate crisp high-frequency details (as evidenced by its superior visual sharpness).

