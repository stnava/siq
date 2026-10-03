### r16c

| method | psnr | ssim | gmsd | cbi | cqs | pcs | pcsc | acutance | laplacian |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| nearest | 23.39 | 0.8961 | 0.1709 | 0.0402 | 0.6850 | 2.0381 | 0.3318 | 1.0110 | 1.6953 |
| bilinear | 28.76 | 0.9671 | 0.1255 | 0.0255 | 0.8161 | 1.5339 | 0.5339 | 0.8271 | 0.6086 |
| bspline | 29.63 | 0.9733 | 0.1126 | 0.0245 | 0.8362 | 1.6808 | 0.6808 | 0.9707 | 0.7185 |
| windowed-sinc | 29.66 | 0.9722 | 0.1154 | 0.0245 | 0.8323 | 1.7039 | 0.7039 | 0.9918 | 0.7515 |
| linear-LS REAL (K=9) | 28.69 | 0.9608 | 0.1404 | 0.0257 | 0.7947 | 1.6006 | 0.6006 | 0.7971 | 0.8148 |
| linear-LS oracle in-sample (K=9) | 30.49 | 0.9731 | 0.1140 | 0.0243 | 0.8348 | 1.6189 | 0.6189 | 0.8926 | 0.6754 |
| MODEL real_mse_matched | 23.35 | 0.9600 | 0.1171 | 0.0241 | 0.8188 | 1.5241 | 0.5241 | 0.7619 | 0.6486 |
| MODEL real_curr_aa_latest | 23.89 | 0.8918 | 0.1830 | 0.0496 | 0.6591 | 2.5065 | -0.1882 | 1.8324 | 1.8622 |
| MODEL real_curr_aa_best | 23.87 | 0.8914 | 0.1831 | 0.0496 | 0.6587 | 2.5065 | -0.1891 | 1.8340 | 1.8616 |
| PUBLIC eugenesiow/edsr-base | 27.32 | 0.9528 | 0.1334 | 0.0394 | 0.7801 | 2.0824 | 0.4777 | 1.3010 | 1.3037 |
| PUBLIC eugenesiow/msrn-bam | 26.98 | 0.9513 | 0.1348 | 0.0424 | 0.7741 | 2.1191 | 0.4291 | 1.3264 | 1.3635 |
| bil+unsharp best-CQS (s=2.0,a=0.25) | 29.00 | 0.9701 | 0.1191 | 0.0256 | 0.8254 | 1.6925 | 0.6925 | 0.9991 | 0.7351 |
| bil+unsharp best-PCSc (s=1.0,a=1.0) | 27.78 | 0.9602 | 0.1274 | 0.0288 | 0.8040 | 1.8867 | 0.7214 | 1.1497 | 1.0157 |

**Verdicts (CQS; PSNR is informational):**

| method | Δ vs bilinear | Δ vs best classical | Δ vs linear ceiling | beats classical | beats ceiling |
|:--|:-:|:-:|:-:|:-:|:-:|
| MODEL real_mse_matched | +0.0027 | -0.0174 | -0.0161 | no | no |
| MODEL real_curr_aa_latest | -0.1570 | -0.1771 | -0.1757 | no | no |
| MODEL real_curr_aa_best | -0.1574 | -0.1775 | -0.1761 | no | no |
| PUBLIC eugenesiow/edsr-base | -0.0360 | -0.0561 | -0.0547 | no | no |
| PUBLIC eugenesiow/msrn-bam | -0.0420 | -0.0621 | -0.0607 | no | no |

### heldout

| method | psnr | ssim | gmsd | cbi | cqs | pcs | pcsc | acutance | laplacian |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| nearest | 24.30 | 0.8593 | 0.1700 | 0.0835 | 0.6058 | 1.6955 | 0.5091 | 1.0057 | 1.1736 |
| bilinear | 28.26 | 0.9337 | 0.1348 | 0.0594 | 0.7396 | 1.3199 | 0.3199 | 0.7117 | 0.4489 |
| bspline | 28.40 | 0.9354 | 0.1289 | 0.0602 | 0.7462 | 1.4809 | 0.4809 | 0.9054 | 0.5641 |
| windowed-sinc | 28.29 | 0.9335 | 0.1304 | 0.0608 | 0.7423 | 1.5062 | 0.5062 | 0.9388 | 0.5890 |
| linear-LS REAL (K=9) | 27.92 | 0.9250 | 0.1402 | 0.0604 | 0.7244 | 1.3655 | 0.3655 | 0.7060 | 0.5761 |
| linear-LS oracle in-sample (K=9) | 28.83 | 0.9362 | 0.1303 | 0.0607 | 0.7452 | 1.4169 | 0.4053 | 0.7922 | 0.5512 |
| MODEL real_mse_matched | 28.33 | 0.9370 | 0.1276 | 0.0562 | 0.7532 | 1.3841 | 0.3841 | 0.7546 | 0.5073 |
| MODEL real_curr_aa_latest | 22.82 | 0.8260 | 0.2047 | 0.1039 | 0.5174 | 2.5547 | -0.5200 | 2.5064 | 1.5682 |
| MODEL real_curr_aa_best | 22.82 | 0.8262 | 0.2048 | 0.1037 | 0.5177 | 2.5517 | -0.5162 | 2.5043 | 1.5636 |
| PUBLIC eugenesiow/edsr-base | 24.89 | 0.8977 | 0.1591 | 0.0989 | 0.6397 | 2.1803 | 0.0991 | 1.7847 | 1.2964 |
| PUBLIC eugenesiow/msrn-bam | 24.91 | 0.8984 | 0.1584 | 0.0997 | 0.6403 | 2.1819 | 0.0988 | 1.7796 | 1.3035 |
| bil+unsharp best-CQS (s=2.0,a=0.25) | 28.17 | 0.9359 | 0.1313 | 0.0599 | 0.7446 | 1.4935 | 0.4935 | 0.9425 | 0.5552 |
| bil+unsharp best-PCSc (s=1.0,a=0.5) | 27.97 | 0.9334 | 0.1332 | 0.0609 | 0.7394 | 1.5244 | 0.5244 | 0.9513 | 0.6186 |

**Verdicts (CQS; PSNR is informational):**

| method | Δ vs bilinear | Δ vs best classical | Δ vs linear ceiling | beats classical | beats ceiling |
|:--|:-:|:-:|:-:|:-:|:-:|
| MODEL real_mse_matched | +0.0136 | +0.0070 | +0.0080 | yes | yes |
| MODEL real_curr_aa_latest | -0.2222 | -0.2288 | -0.2278 | no | no |
| MODEL real_curr_aa_best | -0.2219 | -0.2285 | -0.2275 | no | no |
| PUBLIC eugenesiow/edsr-base | -0.0999 | -0.1065 | -0.1055 | no | no |
| PUBLIC eugenesiow/msrn-bam | -0.0993 | -0.1059 | -0.1049 | no | no |

### heldout_aa

| method | psnr | ssim | gmsd | cbi | cqs | pcs | pcsc | acutance | laplacian |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| nearest | 25.13 | 0.8621 | 0.1843 | 0.0684 | 0.6093 | 1.1831 | 0.1831 | 0.4505 | 0.6971 |
| bilinear | 26.60 | 0.8946 | 0.1775 | 0.0644 | 0.6527 | 0.9545 | -0.0455 | 0.3779 | 0.2256 |
| bspline | 27.43 | 0.9149 | 0.1607 | 0.0633 | 0.6909 | 1.0424 | 0.0424 | 0.4443 | 0.2587 |
| windowed-sinc | 27.51 | 0.9167 | 0.1582 | 0.0632 | 0.6953 | 1.0580 | 0.0580 | 0.4541 | 0.2713 |
| linear-LS REAL (K=9) | 26.70 | 0.8937 | 0.1775 | 0.0644 | 0.6518 | 0.9980 | -0.0020 | 0.3785 | 0.3139 |
| linear-LS oracle in-sample (K=9) | 29.81 | 0.9492 | 0.1251 | 0.0614 | 0.7627 | 1.4617 | 0.4617 | 0.8282 | 0.5699 |
| MODEL real_mse_matched | 26.48 | 0.8974 | 0.1749 | 0.0631 | 0.6594 | 0.9772 | -0.0228 | 0.3847 | 0.2509 |
| MODEL real_curr_aa_latest | 30.19 | 0.9498 | 0.1180 | 0.0580 | 0.7739 | 1.4898 | 0.4898 | 0.8243 | 0.6076 |
| MODEL real_curr_aa_best | 30.20 | 0.9500 | 0.1181 | 0.0580 | 0.7740 | 1.4905 | 0.4905 | 0.8260 | 0.6071 |
| PUBLIC eugenesiow/edsr-base | 29.45 | 0.9382 | 0.1327 | 0.0581 | 0.7474 | 1.1761 | 0.1761 | 0.5196 | 0.3379 |
| PUBLIC eugenesiow/msrn-bam | 29.45 | 0.9382 | 0.1329 | 0.0581 | 0.7472 | 1.1757 | 0.1757 | 0.5200 | 0.3370 |
| bil+unsharp best-CQS (s=2.0,a=1.0) | 28.05 | 0.9346 | 0.1463 | 0.0626 | 0.7257 | 1.3293 | 0.3293 | 0.7882 | 0.4190 |
| bil+unsharp best-PCSc (s=2.0,a=1.0) | 28.05 | 0.9346 | 0.1463 | 0.0626 | 0.7257 | 1.3293 | 0.3293 | 0.7882 | 0.4190 |

**Verdicts (CQS; PSNR is informational):**

| method | Δ vs bilinear | Δ vs best classical | Δ vs linear ceiling | beats classical | beats ceiling |
|:--|:-:|:-:|:-:|:-:|:-:|
| MODEL real_mse_matched | +0.0067 | -0.0664 | -0.1033 | no | no |
| MODEL real_curr_aa_latest | +0.1212 | +0.0482 | +0.0112 | yes | yes |
| MODEL real_curr_aa_best | +0.1213 | +0.0483 | +0.0113 | yes | yes |
| PUBLIC eugenesiow/edsr-base | +0.0947 | +0.0216 | -0.0153 | yes | no |
| PUBLIC eugenesiow/msrn-bam | +0.0945 | +0.0215 | -0.0154 | yes | no |

