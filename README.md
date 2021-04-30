# Crowd Counting on Images with Scale Variation and Isolated Clusters (Accepted by ICCVW-2019)

## Outline

Crowd counting is to estimate the number of objects
(e.g., people or vehicles) in an image of unconstrained congested scenes. Designing a general crowd counting algorithm applicable to a wide range of crowd images is challenging, mainly due to the possibly large variation in object scales and the presence of many isolated small clusters. Previous approaches based on convolution operations
with multi-branch architecture are effective for only some
narrow bands of scales, and have not captured the longrange contextual relationship due to isolated clustering. To
address that, we propose SACANet, a novel scale-adaptive
long-range context-aware network for crowd counting.
SACANet consists of three major modules: the pyramid contextual module which extracts long-range contextual information and enlarges the receptive field, a scale-adaptive self-attention multi-branch module to attain high
scale sensitivity and detection accuracy of isolated clusters, and a hierarchical fusion module to fuse multi-level
self-attention features. We have
conducted extensive experiments using the VisDrone2019
People dataset, the VisDrone2019 Vehicle dataset, and some
other challenging benchmarks. As compared with the stateof-the-art methods, SACANet is shown to be effective, especially for extremely crowded conditions with diverse scales
and scattered clusters, and achieves much lower MAE as
compared with baselines.

### Prerequisites

Python3.6. and the following packages are required to run the scripts:

- [PyTorch-1.1.0 and torchvision](https://pytorch.org)  

- Package [tensorboardX](https://github.com/lanpa/tensorboardX)


- Dataset: please download the dataset and put images into the folder ProcessedData/[name of the dataset, SHHA or SHHB or VisDrone-People or VisDrone-Vehicle]/

- Download Link: [VisDrone-People](https://drive.google.com/file/d/12bCfAWEVurX6Z0RuAbegywkY7Z-UDU19/view?usp=sharing), [VisDrone-Vehicle](https://drive.google.com/file/d/19gh-ZF-FpoTNNtVh_gScRc9pFlqvktpU/view?usp=sharing), [ShanghaiTech PartA and PartB](https://www.kaggle.com/tthien/shanghaitech)

- Pre-Trained Weights: please download the pre-trained weights and put the weights in the folder saves/[SHHA_best_model.pth or SHHB_best_model.pth] 

### Code Structure

There are four parts in the code:
 - model: the main codes of the network architecture.
 - dataloader: the main codes of the dataloader.
 - saves: to put the initialized weights.
 - main.py: the main file to train and evaluate the model.
 - quantification: the main codes of two dimension crowd counting challenges quantification, and the codes to split and preprocess the viscrowd dataset.


### Main Hyper-parameters

We introduce the usual hyper-parameters as below. There are some other hyper-parameters in the code, which are only added to make the code general, but not used for experiments in the paper.

#### Basic Parameters

- `dataset`: The dataset to use. For example, `SHHA` or `SHHB` or `cdpeople` or `cdvehicle`.

- `backbone_class`: The backbone to use, choose `vgg19`.

#### Optimization Parameters

- `max_epoch`: The maximum number of epochs to train the model, default to `500`

- `lr`: The learning rate, default to `0.0001`

- `init_weights`: The path to the init weights

- `batch_size`: The number of inputs for each batch, default to `8`

- `image_size`: The designed input size to preprocess the image, default to `225`

- `prefetch`: The number of workers for dataloader, default to `16`


#### Other Parameters

- `gpu`: To select which GPU device to use, default to `0`.

### Demonstrations on SHHB with SACANet

Train and evaluate SACANet on SHHB:

$ python main.py --dataset SHHB --model_type SACANet --backbone_class vgg19 --max_epoch 500 --lr 0.0001 --gpu 0 --batch_size 8




## References
If you find this work or code useful, please cite:

```
@inproceedings{bai2019crowd,
  title={Crowd counting on images with scale variation and isolated clusters},
  author={Bai, Haoyue and Wen, Song and Gary Chan, S-H},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops},
  pages={0--0},
  year={2019}
}
```




