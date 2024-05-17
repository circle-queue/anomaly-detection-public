# Goal 3
## Plan
We failed to train a model from scratch, however, what if we fine tune a (teacher) resnet architecture using unstructured data? Then it should represent all data in our dataset pretty well. When we then freeze the teacher and train the student only, we expect the discreptency to be larger compared to using a general ImageNet teacher.

We train it using the teacher student architecture, having lower learning rate for the higher layers, especially for the freshly initialized student 

## Datasets
- ScavPortClass consists of 8 classes of engine components (Two tasks, either multi-class classification OR anomaly with one vs all)
    - ScavPortOneVsAll The above binary task of predicting is/is-not the one class vs the remaining 7
- ScavPortAnom consists of 2 classes outlier or inlier (==ScavPortClass)
- VesselArchive consists of ScavPort-like data, along with other junk
- Condition consists of 4 sub-datasets: {overview,lock,ring,ring-surface}, with one "good" label and 1+ abnormal labels
- Dimo consists of 5 sub-datasets

## Todo

### Train BackBones (ResNet18)
Here we want to train a model to create "effective" embeddings for "accurately" representing engine components. Terms in "quotes" are vague, but in short the embeddings should improve downstream tasks
#### Dataset + Method
- [x] Pretrained ImageNet1k (no method)
- [x] ScavPort + Reverse Distillation
- [x] ScavPort + SelfSupervisedContrast
- [x] ScavPort + SupervisedContrast (by 8 classes)
- [ ] VesselArchive + Reverse Distillation
- [ ] VesselArchive + SelfSupervisedContrast
- [ ] Dimo, sub-dataset + ???


### Train Anomaly classifier
Here we want to train a model to detect if something doesn't belong. Not-belonging can be defined as e.g. selfies amongst engine-like things, or a piston ring with a deformation amongst clean piston rings 
#### Datasets
- [ ] ScavPort (one vs all)
- [ ] ScavPortAnom (outlier vs inlier)
- [ ] VesselArchive
- [ ] DIMO (metals)

#### Methods
- [ ] Baseline (embedding distance metric)
- [ ] Reverse Distillation
- [ ] PatchCore
- [ ] ...

#### Done Combinations (Task + method + backbone)
- [x] ScavPortAnom + RevD + ImageNet1k (68% ROC)
- [x] ScavPortAnom + RevD + ScavPort[RevD] (70% ROC)
- [x] ScavPortAnom + RevD + ScavPort[SelfSupContrast] (71% ROC)
- [x] ScavPortAnom + RevD + ScavPort[SupContrast] (72% ROC)

- [x] ScavPortAnom + PatchCore + ImageNet1k (93% ROC)
- [ ] ScavPortAnom + PatchCore + ScavPort[RevD]
- [ ] ScavPortAnom + PatchCore + ScavPort[SelfSupContrast]
- [ ] ScavPortAnom + PatchCore + ScavPort[SupContrast]

- [ ] ScavPortOneVsAll + RevD + ImageNet1k
- [ ] ScavPortOneVsAll + RevD + ScavPort[RevD]
- [ ] ScavPortOneVsAll + RevD + ScavPort[SelfSupContrast]
- [ ] ScavPortOneVsAll + RevD + ScavPort[SupContrast]

- [ ] ScavPortOneVsAll + PatchCore + ImageNet1k
- [ ] ScavPortOneVsAll + PatchCore + ScavPort[RevD]
- [ ] ScavPortOneVsAll + PatchCore + ScavPort[SelfSupContrast]
- [ ] ScavPortOneVsAll + PatchCore + ScavPort[SupContrast]

- [ ] MetalicDataset + RevD + ImageNet1k
- [ ] MetalicDataset + RevD + ScavPort[RevD]
- [ ] MetalicDataset + RevD + ScavPort[SelfSupContrast]
- [ ] MetalicDataset + RevD + ScavPort[SupContrast]
- [ ] MetalicDataset + PatchCore + ImageNet1k
- [ ] MetalicDataset + PatchCore + ScavPort[RevD]
- [ ] MetalicDataset + PatchCore + ScavPort[SelfSupContrast]
- [ ] MetalicDataset + PatchCore + ScavPort[SupContrast]

### ScavPort (Supervised, unsupervised, oneclass vs allclass)
- [ ] Train teacher using Reverse Distillation
- - [ ] Train student
- [ ] Train ResNet using SelfSupervisedContrast
- - [ ] Train student
- - [ ] Train PatchCore
- [ ] Train ResNet using SupervisedContrast
- - [ ] Train student
- - [ ] Train PatchCore
### VesselArchive (unsupervised)
- [ ] Train teacher using Reverse Distillation
- - [ ] Train student
- [ ] Train ResNet using SelfSupervisedContrast
- - [ ] Train student
- - [ ] Train PatchCore
- [ ] Train ResNet using SupervisedContrast
- - [ ] Train student
- - [ ] Train PatchCore

### Dimo