# CAP6415_F25_Image-Feature-Reconstruction

##  Project Description

### Image Feature Reconstruction 
This project explores the inversion of deep neural network representations: given feature maps extracted from intermediate layers of a pretrained backbone (ResNet‑50), the goal is to reconstruct the original input image. By analyzing features from multiple depths—early, mid, deep, and very deep layers—we investigate how much visual information is preserved at different stages of the network.

---

###  Objectives
- Implement optimization‑based inversion methods using feature losses and natural image priors.
- Train learned decoder networks to reconstruct images directly from feature maps.
- Compare reconstructions across at least four different layers of ResNet‑50.
- Evaluate results using quantitative metrics (PSNR, SSIM, LPIPS) and qualitative visual comparisons.
- Ensure reproducibility through version control, logging, and documented workflows.

---

###  Dataset
The project uses the **ImageNet‑1K dataset** (downloaded via HuggingFace), which contains ~1.28M labeled images across 1,000 object classes.  
This benchmark dataset provides diverse natural images and aligns with the pretrained ResNet‑50 backbone used for feature extraction.

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


**Maintained:** by CAP6415 Project Team (Fall 2025 – Computer Vision)
