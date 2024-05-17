# Goal 2
## Architecture
![img](patchcore_architecture.png)
TL;DR: Patchcore compares new, potentially anomalous images to a memory bank of good observations seen during training. If the new image is very far from training samles, the image is marked as anomalous. 

Details: The memory bank contains patches of training images embedded using a pretrained model such as ResNet50 on ImageNet. These patches can be used to create a pixel-wise anomaly score to overlay over an image for explainability. The memory bank is downsampled in a clever way for computational efficiency. 

## Plan
Since this requires no additional trainable parameters other than a threshold, this approach may be well suited for few-shot learning on out datasets with <50 samples.