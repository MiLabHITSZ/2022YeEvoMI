# Evolutionary Model Inversion Attack

Implementation of paper *"Z. Ye, W, Luo and Z. Zhang, Evolutionary Model Inversion Attack."*

# Requirements

I have tested on:

- PyTorch 1.11.0
- CUDA 11.4

# Usage

### If there is already a trained StyleGAN and target classifier, example of attacking from label-0 to label-9:

> python main.py --init_label 0 --final_label 10 path_to_StyleGAN path_to_classifier path_for_saving backbone_of_classifier classifier_output_dim

### or training your own models: 

- train a StyleGAN first, please refer to https://github.com/NVlabs/stylegan2 and https://github.com/rosinality/stylegan2-pytorch
- train target classifier:
  
  1. run `annotation_face_train.py` to prepare the list file of **training data** and **testing data**;
  2. run `train_facenet.py` to train a target classifier;
  3. run `main.py` as above.

### or using our trained models directly:

- StyleGAN2: https://huggingface.co/ZipZip/Model_EvoMI/resolve/main/100000.pt.
- Target Classifiers (dataset & backbone):

  1. CASIA-WebFace & InceptionResNet: https://huggingface.co/ZipZip/Model_EvoMI/resolve/main/CASIA_InceptionResnetV1.pth;
  2. CASIA-WebFace & MobileNet: https://huggingface.co/ZipZip/Model_EvoMI/resolve/main/CASIA_MobileNet.pth;
  3. FaceScrub & InceptionResNet: https://huggingface.co/ZipZip/Model_EvoMI/resolve/main/FaceScrub_InceptionResnetV1.pth;
  4. FaceScrub & MobileNet: https://huggingface.co/ZipZip/Model_EvoMI/resolve/main/FaceScrub_MobileNet.pth.
 
 # REFERENCES
 
 *T. Karras, et al. "Analyzing and improving the image quality of stylegan," in CVPR, 2020.*
 
 *A. G. Howard, et al. "Mobilenets: Efficient convolutional neural networks for mobile vision applications," arXiv preprint arXiv:1704.04861, 2017.*
 
 *C. Szegedy, et al. "Inception-v4, inception-resnet and the impact of residual connections on learning," in AAAI, 2017.*
 
 *https://github.com/rosinality/stylegan2-pytorch*
 
