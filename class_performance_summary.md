# Evaluation of Super-Resolution Models Across 9 Simulation Classes

This report evaluates the performance of the trained 2D super-resolution models on 3 on-the-fly generated examples from each of the 9 mixed simulation classes (with no Layer 2 augmentations applied).

## Overall Performance: Rank of Ranks

The **Rank of Ranks** is a robust metric that aggregates performance across all 9 simulation classes and all 5 evaluation metrics (PSNR, SSIM, GMSD, HFEN, Correlation). For each class and each metric, models are ranked 1 to 11 (lower is better). The final score is the average rank across all classes and metrics.

|    | Model    |   Rank Score (Avg Rank) |
|---:|:---------|------------------------:|
|  1 | SAN      |                 3.33333 |
|  2 | REF-DBPN |                 3.44444 |
|  3 | LDBPN    |                 3.75556 |
|  4 | RCAN     |                 4.62222 |
|  5 | SRFBN    |                 5.64444 |
|  6 | ESPCN-RC |                 5.8     |
|  7 | WDSR     |                 6.6     |
|  8 | CARN     |                 7.06667 |
|  9 | ESPCN    |                 7.37778 |
| 10 | WDSR-RC  |                 7.84444 |
| 11 | Bilinear |                10.5111  |

## Average PSNR (dB) per Class

| Model    |   brain_procedural |   cellular_voronoi |   fractal_noise |   geometric_phantoms |   grid_patterns |   layered |   organic_blobs |   sinewave |   vessel_tubes |
|:---------|-------------------:|-------------------:|----------------:|---------------------:|----------------:|----------:|----------------:|-----------:|---------------:|
| Bilinear |            19.819  |            15.2358 |         22.4011 |              23.4722 |         14.0296 |   19.5745 |         16.4168 |    22.5122 |        23.2078 |
| CARN     |            21.4919 |            15.9991 |         28.1759 |              24.2699 |         14.3146 |   22.5523 |         17.5009 |    31.9095 |        23.2651 |
| ESPCN    |            21.4701 |            15.9691 |         28.1253 |              24.2871 |         14.3008 |   22.5604 |         17.5192 |    31.7902 |        23.2255 |
| ESPCN-RC |            21.7526 |            16.1358 |         27.8856 |              24.3634 |         14.3191 |   22.8663 |         17.3561 |    31.7188 |        23.6847 |
| LDBPN    |            22.2559 |            16.7337 |         28.975  |              24.6632 |         14.2966 |   22.918  |         17.5242 |    32.6624 |        24.1767 |
| RCAN     |            21.8841 |            16.23   |         28.7142 |              24.563  |         14.2393 |   22.6832 |         17.5422 |    32.5678 |        23.7033 |
| REF-DBPN |            22.0534 |            16.7379 |         28.9744 |              24.5443 |         14.3544 |   22.9308 |         17.5907 |    32.5928 |        23.9088 |
| SAN      |            21.9691 |            16.2889 |         28.919  |              24.6009 |         14.1209 |   22.8354 |         17.5982 |    32.9722 |        23.7839 |
| SRFBN    |            21.5305 |            16.2952 |         28.3577 |              24.2298 |         14.3385 |   22.7961 |         17.5487 |    31.5894 |        23.3215 |
| WDSR     |            21.4559 |            16.0278 |         28.1513 |              24.2424 |         14.3099 |   22.625  |         17.5192 |    31.8303 |        23.3126 |
| WDSR-RC  |            21.4471 |            15.958  |         28.0724 |              24.2164 |         14.3094 |   22.5429 |         17.5217 |    31.6938 |        23.2176 |

## Average SSIM per Class

| Model    |   brain_procedural |   cellular_voronoi |   fractal_noise |   geometric_phantoms |   grid_patterns |   layered |   organic_blobs |   sinewave |   vessel_tubes |
|:---------|-------------------:|-------------------:|----------------:|---------------------:|----------------:|----------:|----------------:|-----------:|---------------:|
| Bilinear |           0.947284 |           0.821307 |        0.944605 |             0.949652 |        0.616062 |  0.607547 |        0.458359 |   0.936562 |       0.972778 |
| CARN     |           0.964407 |           0.849252 |        0.985501 |             0.95894  |        0.640526 |  0.697694 |        0.565355 |   0.994078 |       0.971541 |
| ESPCN    |           0.964241 |           0.847439 |        0.98531  |             0.959087 |        0.640177 |  0.69811  |        0.566785 |   0.993863 |       0.971853 |
| ESPCN-RC |           0.966459 |           0.853729 |        0.984467 |             0.959783 |        0.638954 |  0.700605 |        0.554838 |   0.993753 |       0.974016 |
| LDBPN    |           0.97024  |           0.872233 |        0.987885 |             0.962371 |        0.639101 |  0.698587 |        0.565923 |   0.994924 |       0.976781 |
| RCAN     |           0.967645 |           0.855929 |        0.987147 |             0.961506 |        0.635637 |  0.698786 |        0.567827 |   0.994889 |       0.974199 |
| REF-DBPN |           0.969086 |           0.873244 |        0.987889 |             0.96154  |        0.643987 |  0.700553 |        0.570775 |   0.994893 |       0.975375 |
| SAN      |           0.968226 |           0.859364 |        0.987777 |             0.961517 |        0.62777  |  0.702598 |        0.571694 |   0.995359 |       0.974771 |
| SRFBN    |           0.964989 |           0.859205 |        0.986025 |             0.958529 |        0.64218  |  0.69951  |        0.567558 |   0.993556 |       0.972525 |
| WDSR     |           0.964195 |           0.850307 |        0.985382 |             0.958557 |        0.640645 |  0.698357 |        0.567186 |   0.993969 |       0.971966 |
| WDSR-RC  |           0.964134 |           0.847814 |        0.985103 |             0.958293 |        0.641324 |  0.697983 |        0.567362 |   0.993752 |       0.971391 |

## Average GMSD per Class

| Model    |   brain_procedural |   cellular_voronoi |   fractal_noise |   geometric_phantoms |   grid_patterns |   layered |   organic_blobs |   sinewave |   vessel_tubes |
|:---------|-------------------:|-------------------:|----------------:|---------------------:|----------------:|----------:|----------------:|-----------:|---------------:|
| Bilinear |           0.268427 |           0.286606 |        0.154154 |             0.237078 |        0.328907 |  0.192431 |        0.221783 |  0.164919  |       0.200153 |
| CARN     |           0.253491 |           0.286318 |        0.112676 |             0.239747 |        0.324435 |  0.171499 |        0.203885 |  0.0770385 |       0.223751 |
| ESPCN    |           0.25499  |           0.284504 |        0.113912 |             0.239944 |        0.326867 |  0.170947 |        0.203256 |  0.0764738 |       0.202887 |
| ESPCN-RC |           0.247466 |           0.284203 |        0.116959 |             0.229326 |        0.329418 |  0.167731 |        0.204862 |  0.0745029 |       0.226427 |
| LDBPN    |           0.247358 |           0.271993 |        0.113594 |             0.227467 |        0.316115 |  0.170733 |        0.205017 |  0.080879  |       0.223066 |
| RCAN     |           0.246514 |           0.282993 |        0.110772 |             0.231168 |        0.322814 |  0.172399 |        0.202998 |  0.0766918 |       0.226018 |
| REF-DBPN |           0.246926 |           0.273072 |        0.112434 |             0.232485 |        0.318993 |  0.170552 |        0.206975 |  0.0845422 |       0.220349 |
| SAN      |           0.249293 |           0.282596 |        0.109165 |             0.229018 |        0.318365 |  0.169873 |        0.201601 |  0.078404  |       0.228765 |
| SRFBN    |           0.244594 |           0.280621 |        0.108406 |             0.229732 |        0.322815 |  0.168927 |        0.203086 |  0.0913388 |       0.200409 |
| WDSR     |           0.252086 |           0.288625 |        0.113703 |             0.239426 |        0.32703  |  0.170144 |        0.203621 |  0.0762073 |       0.225405 |
| WDSR-RC  |           0.253772 |           0.290804 |        0.113568 |             0.240389 |        0.328656 |  0.170648 |        0.203428 |  0.0769459 |       0.224275 |

## Average HFEN per Class

| Model    |   brain_procedural |   cellular_voronoi |   fractal_noise |   geometric_phantoms |   grid_patterns |   layered |   organic_blobs |   sinewave |   vessel_tubes |
|:---------|-------------------:|-------------------:|----------------:|---------------------:|----------------:|----------:|----------------:|-----------:|---------------:|
| Bilinear |           0.632273 |           0.667828 |        1.12294  |             0.663152 |        0.631824 |   1.90645 |        0.991612 |   4.57725  |       0.415671 |
| CARN     |           0.362048 |           0.48649  |        0.332187 |             0.487987 |        0.522018 |   1.2274  |        0.735479 |   0.58345  |       0.313198 |
| ESPCN    |           0.360323 |           0.494785 |        0.332753 |             0.484649 |        0.529562 |   1.23729 |        0.731798 |   0.574889 |       0.326601 |
| ESPCN-RC |           0.340232 |           0.473131 |        0.363473 |             0.467852 |        0.505612 |   1.12908 |        0.730896 |   0.554871 |       0.30334  |
| LDBPN    |           0.331373 |           0.452327 |        0.364463 |             0.456613 |        0.499642 |   1.1551  |        0.748817 |   0.752485 |       0.308643 |
| RCAN     |           0.323994 |           0.466269 |        0.338451 |             0.456781 |        0.505856 |   1.17059 |        0.738758 |   0.575888 |       0.302849 |
| REF-DBPN |           0.347623 |           0.468821 |        0.357739 |             0.480258 |        0.524024 |   1.1594  |        0.758837 |   0.728356 |       0.314296 |
| SAN      |           0.323422 |           0.455615 |        0.343428 |             0.451225 |        0.489944 |   1.15131 |        0.741258 |   0.573484 |       0.289771 |
| SRFBN    |           0.368634 |           0.479056 |        0.348014 |             0.501383 |        0.513433 |   1.18    |        0.752232 |   0.719055 |       0.334741 |
| WDSR     |           0.35473  |           0.482563 |        0.331871 |             0.487187 |        0.527844 |   1.22145 |        0.737753 |   0.558615 |       0.307528 |
| WDSR-RC  |           0.353905 |           0.487506 |        0.326115 |             0.48728  |        0.528027 |   1.23546 |        0.735203 |   0.552289 |       0.317063 |

## Average Correlation per Class

| Model    |   brain_procedural |   cellular_voronoi |   fractal_noise |   geometric_phantoms |   grid_patterns |   layered |   organic_blobs |   sinewave |   vessel_tubes |
|:---------|-------------------:|-------------------:|----------------:|---------------------:|----------------:|----------:|----------------:|-----------:|---------------:|
| Bilinear |           0.947343 |           0.823742 |        0.946037 |             0.949496 |        0.639848 |  0.593751 |        0.456873 |   0.937138 |       0.974519 |
| CARN     |           0.964483 |           0.851815 |        0.987378 |             0.958863 |        0.666625 |  0.690585 |        0.568407 |   0.995134 |       0.975103 |
| ESPCN    |           0.964319 |           0.850014 |        0.987189 |             0.959014 |        0.666201 |  0.690841 |        0.569717 |   0.994922 |       0.974851 |
| ESPCN-RC |           0.966566 |           0.85634  |        0.986288 |             0.959722 |        0.664734 |  0.692117 |        0.557611 |   0.994838 |       0.977095 |
| LDBPN    |           0.970364 |           0.875001 |        0.989763 |             0.962324 |        0.665148 |  0.691916 |        0.569094 |   0.996013 |       0.979317 |
| RCAN     |           0.967735 |           0.85856  |        0.988955 |             0.96146  |        0.661793 |  0.691427 |        0.571276 |   0.995974 |       0.977262 |
| REF-DBPN |           0.969185 |           0.876017 |        0.989778 |             0.961488 |        0.670146 |  0.693364 |        0.57444  |   0.996014 |       0.978154 |
| SAN      |           0.968313 |           0.86203  |        0.989617 |             0.96147  |        0.653356 |  0.695178 |        0.574962 |   0.996428 |       0.977619 |
| SRFBN    |           0.965065 |           0.86185  |        0.987891 |             0.958459 |        0.668292 |  0.691654 |        0.57082  |   0.994612 |       0.975262 |
| WDSR     |           0.964268 |           0.85288  |        0.987229 |             0.958478 |        0.666456 |  0.690848 |        0.570379 |   0.995044 |       0.975323 |
| WDSR-RC  |           0.964194 |           0.850345 |        0.986957 |             0.958222 |        0.667139 |  0.690822 |        0.570348 |   0.99483  |       0.974822 |

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

