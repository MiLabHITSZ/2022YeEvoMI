# Evolutionary Model Inversion Attack

Implementation of paper *Z. Ye, W, Luo and Z. Zhang, Evolutionary Model Inversion Attack.*

# Requirements

I have tested on:

- PyTorch 1.11.0
- CUDA 11.4

# Usage

## If there is already a trained StyleGAN and target classifier, attack from label-0 to label-9:

> python main.py --init_label 0 --final_label 10 path_to_StyleGAN path_to_classifier path_for_saving backbone_of_classifier classifier_output_dim
