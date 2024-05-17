# Goal 1
## Architecture
![](ts_architecture.png)
source: https://openaccess.thecvf.com/content/CVPR2022/papers/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.pdf

## Big picture
I aim to apply [Anomaly Detection via Reverse Distillation from One-Class Embedding](https://openaccess.thecvf.com/content/CVPR2022/papers/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.pdf), which trains a student resnet50 model to immitate intermediate activations of a pre-trained teacher resnet50 model. It does this using an encoder-decoder structure, where the teacher is the encoder, and the student is the decoder. In this way, it's similar to a reconstruction loss.

The idea is that the student will only be trained on samples we want to include, so if the student is presented with a sample that is very different from what it was trained on, it would be very different from the teacher response.

However, larger models have a lot of redundant representation capability which makes anomaly detection tricky. Instead, we train a residule block in a one-class manner to compress the representation tuned to the data. Then the model will only work on information specific to the data domain.

## Data
In the first step, I aim to train the model on our existing scavenge port inspection image dataset of <10k images belonging to one of 8 classes, along with a large unannotated dataset called "raw", given the class "raw". Hopefully the model will be able to capture features of both types of images, such that when the Teacher-Student model is trained on only the good images, there'll be a high error for the poor images.

We can then verify the model performance on a test set of ~1k, where some of these images are out-of-class, and measure the accuracy. Carsten says that he prefers higher recall over precision. It is better that we filter too many images than too few.

We may also consider adding an "uncertain" class by thresholding the anomaly signal to some bound, e.g. 5% < "uncertain < 95%