# Semi-supervised Learning Combined with Regression Pre-training of Siamese Network Based on Superpixel Segmentation for HSI Classification

> **Abstract:** *Despite the success of deep neural networks (DNNs), insufficient labeled samples remain a key challenge for DNN based hyperspectral image (HSI) classification. To perform better with limited labeled samples, some HSI classification methods try to acquire spatial information through superpixel segmentation (SPS). However, the existing methods seldom use multi-band and multiscale SPS simultaneously to extract features, and lack effective means to mine the superpixel-wise information fully. This paper proposes a semi-supervised learning method combined with regression pre-training of siamese network based on superpixel segmentation (SLRSS), which conducts the regression pre-training by using the SPS results to enhance the following semi-supervised learning. First, we design an average edge-weighted graph to generate the similarity labels of each sample pair by using the multi-band multi-scale SPS results obtained from the unlabeled HSI. Next, we randomly sample a proportion of sample pairs with their corresponding similarity labels and add them into the regression training (RT) set. To fully utilize the limited labeled samples in the training set for classification, these limited labeled samples are also used to construct the sample pairs with the similarity labels according to their true labels and added into the RT set. Then, the parameters of the well pre-trained siamese network after the regression training are used to initialize the parameters of the feature extraction module in the classification training. Finally, we design a novel semi-supervised learning strategy named as model-selected labeling to enlarge the training set in the classification training. Experimental results on two HSI datasets show that the proposed SLRSS outperforms several state-of-the-art methods significantly, with only five labeled samples per class. For study replication, the code developed for this study is available at https://github.com/Maike-Feng/SLRSS.git.* 
<hr />

## Network Architecture
<div aligh=center witdh="200"><img src="Network Architecture.png"></div>

## What is this repository for?
This is the PyTorch implementation for **Semi-supervised Learning Combined with Regression Pre-training of Siamese Network Based on Superpixel Segmentation for HSI Classification**.

## Usage

### 1. Create Envirement:

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages:

  ```shell
  cd SLRSS
  pip install -r requirements.txt
  ```

## 2. Data Preparation(Taking the `Salinas` dataset as an example):
```
(1)、Place the dataset ( `Salinas_corrected.mat` )  in the ERS folder.

(2)、Open the ERS folder using Matlab software.

(3)、To achieve multi band entropy superpixel segmentation of IndianPines, execute the file runmore_HSI_seg_main. m, where nC controls the number of superpixels, i.e. the segmentation scale. Then we will obtain files such as `Salinas_255seg30.mat` `Salinas_255seg50.mat` `Salinas_255seg70.mat` in the folder ERS.

(4)、Run the `data_preprocess.py` file to achieve the fusion of the newly obtained superpixel data. And then we will get the file `Salinas_edgelabels.mat`.
```
## 3. Training and Prediction:
Train the model and obtain the results, just run the file `SLRSS.py`
```shell
# Training on Salinas dataset
python SLRSS.py --data_name Salinas
```

## Who do I talk to?
Yi Liu, Xidian University, yiliuxd@foxmail.com
