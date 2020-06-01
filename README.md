# Implemention of SalGAN by Keras

SalGAN is a model to generate saliency map proposed by J.Pan in 2017.
This repository is an implementation of Salgan using Keras framework.
[SalGAN: Visual Saliency Prediction with Generative Adversarial Networks](https://arxiv.org/abs/1701.01081) - Original Paper of SalGAN

## Getting Started

### Download scripts
Clone this repository and add some additional directories.
Your directory should be as shown below.

```
salgan/
- data/
- model/
- prediction/
- train.py
- preprocess.py
- predict.py
- validate.py
- utils/
-- metrics.py
-- model.py
```


### Download datasets

There are several datasets used for saliency map generation.
This implementation uses SALICON datasets which is same with original paper.
SALICON datasets consists of training (10,000), validation (5,000), and test (1,000) datasets.
This data can be downloaded from the site below.
[SALICON challenge 2017](http://salicon.net/challenge-2017/) - HP for SALICON challenge 2017

Set the downloaded images under data directory as shown below.

```
data/
- images/
-- train/
--- ...
-- val/
--- ...
-- test/
--- ...
- maps/
-- train/
--- ...
-- val/
--- ...
```

### Download model weight

If you want to make use of pre-trained model weight, the weight file can be downloaded from link below.
[Pre-trained model weight file](https://drive.google.com/open?id=1A0AovgjQQuNtt-sCg9WREG91ZMus__Lw) - Model weight file uploaded on google drive

Set the downloaded file under model directory.

```
model/
- weights_bce4_15.hdf5
```


## Running the code

### Preprocessing

Convert image and map data to .npy file using the command below.
.npy files will be output under data directory.

```
python preprocess.py
```

### training

There are several parameters which can be checked in train.py
Also, fine-tuning can be done by using --load_model_path flag.

```
python train.py --model_name=salgan --num_epoch=100 --model_save_ratio=1
```

### Prediction

Prediction can be made by loading trained model weights.
```
python predict.py --load_model_path=model/weights_bce4_15.hdf5 --target_data_path=data/Xval.npy --output_data_path=prediction/pred.npy
```


### Validation

Validation of saliency maps using 4 index. (AUC_borji, AUC_shuffled, nss, cc)
Load ground truth and corresponding prediction map file.

```
python validate.py --gt_data_path=data/Yval.npy --prediction_data_path=prediction/pred.npy
```

## Author

* **Ryota Nomura** - *Initial work* - [HomePage](http://ryota-n.info/)


## Acknowledgments

* [SalGAN: Visual Saliency Prediction with Generative Adversarial Networks](https://arxiv.org/abs/1701.01081) - Original Paper of SalGAN
* [SalGAN: code by authors](https://github.com/imatge-upc/salgan) - Implementation of SalGAN in Lasagne (Theano) made by the authors
* [saliency metrics](http://salicon.net/challenge-2017/) - Implemention of saliency metrics in python2
* [GAN sample code in Keras](https://github.com/eriklindernoren/Keras-GAN) - Sample implementation of various GANs in Keras
