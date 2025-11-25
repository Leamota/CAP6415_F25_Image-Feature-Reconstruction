# CAP6415_F25_Image-Feature-Reconstruction

##  Project Description

### Abstract
This project explores the inversion of deep neural network representations: given feature maps extracted from intermediate layers of a pretrained backbone (ResNet‑50), the goal is to reconstruct the original input image. By analyzing features from multiple depths—early, mid, deep, and very deep layers—we investigate how much visual information is preserved at different stages of the network.

---

###  Objectives
- Train learned decoder networks to reconstruct images directly from feature maps.
- Compare reconstructions across at least four different layers of ResNet‑50.
- Evaluate results using quantitative metrics (PSNR, SSIM) and qualitative visual comparisons.
- Ensure reproducibility through version control, logging, and documented workflows.

---

###  Dataset

As a result of computational and storage constraints, the project utilizes the **imageNet ILSVRC2012 validation** split (50,000 labeled across 1,000 classes). This 
images where obtained directly from the [official ImageNet website](https://www.image-net.org/). Our evalution and class referencing are facilitated by  ground truth lables ILSVRC2012_validation_ground_truth.txt available on [ILSVRC2012 Validation Ground Truth File on GitHub](https://github.com/Spiritator/machine-learning-dataset-tool/blob/master/ILSVRC2012_validation_ground_truth.txt).

In this subset, practical training and evalution are enabled while preserving the relevance ot imageNet-1K-trained models. The initial attempt to use the full imageNet-1K was restricted by storage and compute capability available to me.

---

### Files

#### requirements.txt
Lists all Python package dependencies required to run this project:
- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `numpy>=1.20.0`
- `matplotlib>=3.4.0`
- `scikit-image>=0.19.0`

To install all required libraries, run:

pip install -r requirements.txt


#### feature_extractor.py
Implements the modular feature extraction pipeline.  
This file contains the `FeatureExtractor` class, which uses forward hooks on multiple ResNet-50 layers to extract intermediate features at several depths.  
It supports both multi-layer and single-layer feature extraction workflows for downstream image reconstruction and analysis.

---

#### extract_and_concat_features.py
Provides a utility function to process and combine feature maps from different network depths.  
This script upsamples each extracted feature map to a common spatial size and concatenates them along the channel dimension, yielding a tensor suitable for decoders that require joint multi-layer input.

---

#### Training and Results
Trained the decoder for 32 epochs on the feature-based reconstruction task, with average training loss decreasing from about 0.97 (epoch 1) to about 0.88 (epoch 32).
Evaluated reconstruction quality using MSE, PSNR, and SSIM on a representative batch: MSE ≈ 1.0028, PSNR ≈ 1.25 dB, SSIM ≈ 0.1511.
Saved side-by-side visualizations of original vs reconstructed images for qualitative inspection.
Experimented with decoder width, depth, and layer/pooling combinations, and compared baseline vs modified models using the same metrics.

---

### Results

The `results` folder contains original images and their corresponding reconstructed images from the model.

#### Filenames Pattern
- `original_X.jpg` → Original test image, where **X** is the sample index  
- `reconstruction_X.jpg` → Model output for the same index  

This structure allows direct visual comparison between ground-truth and reconstructed samples for both qualitative and quantitative evaluation.

---

#### Example (Text Table)

| Original Image   | Reconstruction       |
|------------------|----------------------|
| original_0.png   | reconstruction_0.png |
| original_1.png   | reconstruction_1.png |
| original_2.png   | reconstruction_2.png |
| ...              | ...                  |

For example:  
`original_1.jpg` and `reconstruction_1.jpg` show the input and its reconstructed output.

These pairs demonstrate the **feature extraction pipeline** and **decoder output quality**.

---

#### Visual Examples (Markdown Thumbnails)

Below are sample pairs of original and reconstructed images:

| Original Image | Reconstruction |
|----------------|----------------|
| ![original_0](original_0.png) | ![reconstruction_0](reconstruction_0.png) |
| ![original_1](original_1.png) | ![reconstruction_1](reconstruction_1.png) |
| ![original_2](original_2.png) | ![reconstruction_2](reconstruction_2.png) |
| ... | ... |

Each row shows the **input image** (left) and its **decoder output** (right).  

---

**Note:**  
You can view these files in the `results` directory of the repository or download them for further analysis.

---



**Citation:**  
I have cited the main ImageNet paper (Deng et al., CVPR 2009) as the source of this dataset.  
If you use ImageNet data in your own work, please also cite the following reference:


Deng, Jia; Dong, Wei; Socher, Richard; Li, Li‑Jia; Li, Kai; and Fei‑Fei, Li.  
*ImageNet: A Large‑Scale Hierarchical Image Database.*  
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2009, pp. 248–255.

---



```bibtex
@inproceedings{deng2009imagenet,
  title={Imagenet: A large-scale hierarchical image database},
  author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
  booktitle={2009 IEEE conference on computer vision and pattern recognition},
  pages={248--255},
  year={2009},
  organization={IEEE}
}
```


**Maintained:** by CAP6415 Project Team (Fall 2025 – Computer Vision), Florida Atlantic University(FAU).
