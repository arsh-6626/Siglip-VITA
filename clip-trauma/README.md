# Zero-Shot Trauma Detection via Keypoint-Guided Segmentation and CLIP based Classification  
**Unified Pipeline for Multi-Human, Multi-Injury Scene Understanding using Pose-Guided Segmentation and CLIP**

---

## Overview  

This project presents the **first unified, annotation-light pipeline** for detecting and classifying physical trauma (injury, amputation, non-injury) from RGB images containing multiple humans. Unlike conventional systems requiring dense pixel-wise labels and real-world training, our approach uses:

- **Zero-shot person detection** via YOLOv9-s  
- **Transformer-based pose estimation** with ViTPose  
- **Pose-densified Gaussian segmentation** to identify body parts  
- **LoRA-finetuned CLIP classifier** to categorize trauma types  

The system is trained **entirely on synthetic data** and generalizes well to real-world images, achieving high accuracy and macro-F1 scores, even in complex, cluttered scenes.

---

## Key Contributions

- **No pixel annotations required** – segmentation is generated analytically using pose keypoints  
- **Real-time performance** – 71 FPS on edge GPUs (TensorRT-accelerated)  
- **Domain-robustness** – trained on 329 synthetic images, performs well on real-world trauma photos  
- **Modular & interpretable** – each stage is self-contained and customizable  

---

## Pipeline Components  

| Stage        | Module             | Description                              |
|--------------|--------------------|------------------------------------------|
| Detection    | YOLOv9             | Fast zero-shot human detection           |
| Pose         | ViTPose            | 17-keypoint Transformer-based pose       |
| Segmentation | Gaussian Generator | Keypoint-driven analytic region masks    |
| Classification | CLIP + LoRA     | Trauma type prediction via vision-language model |

---

## Links  

[Link to required models](https://drive.google.com/drive/folders/1MLY02-5aVPl5mP5_NFZr9ktiTRYWvK-r?usp=drive_link)

---

## Performance  

| Dataset      | Accuracy | Macro-F1 | Avg TP/Image |
|--------------|----------|----------|--------------|
| Synthetic    | 89.5%    | 0.84     | 2.82         |
| Real Images  | 83.4%    | 0.77     | 2.65         |

> The model demonstrates strong generalization, with only a 6% drop across synthetic-to-real domain shift.

---

## IMPORTANT

**Local MMCV-all setup in the same folder as the codes is required to run the provided codes.**
