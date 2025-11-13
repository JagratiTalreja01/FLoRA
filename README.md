# FLoRA
FLoRA: Fusion-Latent for Optical Reconstruction &amp; Area Segmentation is a multi-task deep learning architecture designed for multi-modal SAR–Optical data fusion in flood mapping and environmental monitoring.

The code is built on (PyTorch) and tested on Ubuntu 20.04.6 environment (Python 3.10.13, PyTorch >= 1.1.0) with NVIDIA RTX A4000 with 16GB RAM. 
## Contents
1. [Introduction](#introduction)
2. [Key Highlights](#keyhighlights)
3. [Dependencies](#dependencies)
4. [Train](#train)
5. [Test](#test)
6. [Results](#results)
7. [Acknowledgements](#acknowledgements)

## Introduction

This repository provides the official implementation of FLoRA — Fusion-Latent Optical Reconstruction & Area Segmentation — a unified cross-modal framework for flood mapping that jointly performs SAR-to-Optical translation and flood-region segmentation.

Instead of treating translation and segmentation as isolated tasks, FLoRA builds a single multi-task pipeline where Sentinel-1 VV/VH features are aligned with optical priors through a fusion-latent space. A lightweight optical teacher network guides SAR feature learning using RGB and NDVI cues, while multi-scale windowed cross-attention and FiLM conditioning enforce spectral fidelity, structural consistency, and hydrologic realism.

Designed for disaster response workflows, FLoRA stabilizes joint optimization through gradient decoupling and uses a GAN-enhanced reconstruction objective (Charbonnier + SSIM + FFT) combined with Dice/BCE segmentation losses. Trained on SEN1Floods11, SEN12MS, and DeepFlood, the model produces perceptually accurate optical reconstructions and reliable flood-extent predictions—even under heavy noise and clutter typical of SAR imagery.



![FLoRA (Architecture)](./Figures/architecture.PNG)


## Key Highlights:

 * Dual-Polarization Processing: Independent stems for SAR VV and VH bands, extracting complementary scattering features.

 * Local Cross-Attention Fusion (LCAF): Adaptive fusion mechanism that leverages interactions between VV and VH polarizations for richer representation.

 * Hybrid GAN–Transformer Framework: Combines the perceptual realism of GANs with the global contextual modeling power of Transformers.

 * SE-Gated Skip Connections: Noise-aware skip links that selectively pass useful features while suppressing SAR-induced artifacts.

 * Efficient and Stable Training: Incorporates Residual Swin Transformer Blocks and multi-loss optimization (L1, SSIM, perceptual, adversarial) for balanced realism and accuracy.

 * Superior Performance: Outperforms state-of-the-art SAR-to-Optical translation models on the DeepFlood dataset in PSNR, SSIM, and LPIPS, while producing interpretable optical-like outputs for real flood events.

 * BiSARFormerGAN bridges SAR and optical modalities, improving the usability of SAR imagery for flood mapping, disaster response, and geospatial analysis.

## Dependencies
* Python 3.10.13
* PyTorch >= 1.1.0
* CUDA 12.2
* numpy
* skimage
* **imageio**
* matplotlib
* tqdm
* cv2 >= 3.xx (Only if you want to use video input/output)

## Train
### Prepare training data 

1. Download DEEPFLOOD Dataset, which includes co-registered Sentinel-1 SAR (VV, VH) and Sentinel-2 optical imagery, along with UAV references and auxiliary layers (NDWI, slope, DTM, flood masks). from [DEEPFLOOD dataset](https://figshare.com/articles/dataset/DEEPFLOOD_DATASET_High-Resolution_Dataset_for_Accurate_Flood_Mappingand_Segmentation/28328339).

2. Use SAR_VH, SAR_VV  for Dual-Polarization input and UAV tiles for Target Optical

3. Create train, test and validation sets 70%, 15% & 15%

4. Specify '--dir_data' based on the image's path. 

For more information, please refer to [MT_GAN (PyTorch)](https://github.com/NUAA-RS/MT_GAN).

### Begin to train

Cd to 'src', run the following script to train models.

 **Use the train.py file in the src folder to begin training of the model**

    ```bash
    # Example Training
    python train.py 
    ```
## Test
### Quick start
1. Download DEEPFLOOD DATASET from [GEMS Lab](https://figshare.com/articles/dataset/DEEPFLOOD_DATASET_High-Resolution_Dataset_for_Accurate_Flood_Mappingand_Segmentation/28328339) and split the dataset into 70% train, 15% validation, & 15% test set.


Cd to 'src', run the following scripts.

 **Use the validation.py file in the src folder to begin training of the model**
 **Use the test.py file in the src folder to begin training of the model**

    ```bash
    
    # Example for Validation set
    python validation.py  
    ```
    ```bash
    
    # Example for Test set
    python test.py  
    ```


## Results
### Visual Patches

![Elizabeth Florence](./Figures/Elizabeth_Florence.PNG)

![Grifton Matthew](./Figures/Grifton_Matthew.PNG)

![Princeville Florence](./Figures/Princeville_Matthew.PNG)

![Kinston Matthew](./Figures/Kinston_Matthew.PNG)

![Lumberton Florence](./Figures/Lumberton_Florence.PNG)

![Washington Florence](./Figures/Washington_Florence.PNG)

### Quantitative Results

![Number of Parameters](./Figures/Number_of_Params.PNG)

![PSNR Convergence](./Figures/PSNR_Convergence.PNG)

![Loss Convergence](./Figures/Loss_Convergence.PNG)

![NDWI_Comparisons](./Figures/NDWI_Comparisons.PNG)

![LPIPS_Comparisons](./Figures/LPIPS_Comparison.PNG)


## Acknowledgements
This code is built on [MT_GAN (PyTorch)](https://github.com/NUAA-RS/MT_GAN/tree/main) and [CycleGAN-PyTorch](https://junyanz.github.io/CycleGAN/). We thank the authors for sharing their codes.
