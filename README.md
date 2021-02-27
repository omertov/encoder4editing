# Designing an Encoder for StyleGAN Image Manipulation
  <a href="https://arxiv.org/abs/2102.02766"><img src="https://img.shields.io/badge/arXiv-2008.00951-b31b1b.svg"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/omertov/encoder4editing/blob/main/notebooks/inference_playground.ipynb)

> Recently, there has been a surge of diverse methods for performing image editing by employing pre-trained unconditional generators. Applying these methods on real images, however, remains a challenge, as it necessarily requires the inversion of the images into their latent space. To successfully invert a real image, one needs to find a latent code that reconstructs the input image accurately, and more importantly, allows for its meaningful manipulation. In this paper, we carefully study the latent space of StyleGAN, the state-of-the-art unconditional generator. We identify and analyze the existence of a distortion-editability tradeoff and a distortion-perception tradeoff within the StyleGAN latent space. We then suggest two principles for designing encoders in a manner that allows one to control the proximity of the inversions to regions that StyleGAN was originally trained on. We present an encoder based on our two principles that is specifically designed for facilitating editing on real images by balancing these tradeoffs. By evaluating its performance qualitatively and quantitatively on numerous challenging domains, including cars and horses, we show that our inversion method, followed by common editing techniques, achieves superior real-image editing quality, with only a small reconstruction accuracy drop.

<p align="center">
<img src="docs/teaser.jpg" width="800px"/>
</p>

## Description   
Official Implementation of "Designing an Encoder for StyleGAN Image Manipulation" paper for both training and evaluation. 
The e4e encoder is specifically designed to complement existing image manipulation techniques performed over StyleGAN's latent space.

## Getting Started
### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN (CPU may be possible with some modifications, but is not inherently supported)
- Python 3

### Installation
- Clone the repository:
``` 
git clone https://github.com/omertov/encoder4editing.git
cd encoder4editing
```
- Dependencies:  
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/). 
All dependencies for defining the environment are provided in `environment/e4e_env.yaml`.

### Inference Notebook
We provide a Jupyter notebook found in `notebooks/inference_playground.ipynb` that allows one to encode and perform several editings on real images using StyleGAN.   

### Pretrained Models
Please download the pre-trained models from the following links. Each e4e model contains the entire pSp framework architecture, including the encoder and decoder weights.
| Path | Description
| :--- | :----------
|[FFHQ Inversion](https://drive.google.com/file/d/1cUv_reLE6k3604or78EranS7XzuVMWeO/view?usp=sharing)  | FFHQ e4e encoder.
|[Cars Inversion](https://drive.google.com/file/d/17faPqBce2m1AQeLCLHUVXaDfxMRU2QcV/view?usp=sharing)  | Cars e4e encoder.
|[Horse Inversion](https://drive.google.com/file/d/1TkLLnuX86B_BMo2ocYD0kX9kWh53rUVX/view?usp=sharing)  | Horse e4e encoder.
|[Church Inversion](https://drive.google.com/file/d/1-L0ZdnQLwtdy6-A_Ccgq5uNJGTqE7qBa/view?usp=sharing) | Church e4e encoder.

If you wish to use one of the pretrained models for training or inference, you may do so using the flag `--checkpoint_path`.

In addition, we provide various auxiliary models needed for training your own e4e model from scratch.
| Path | Description
| :--- | :----------
|[FFHQ StyleGAN](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing) | StyleGAN model pretrained on FFHQ taken from [rosinality](https://github.com/rosinality/stylegan2-pytorch) with 1024x1024 output resolution.
|[IR-SE50 Model](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing) | Pretrained IR-SE50 model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for use in our ID loss during training.
|[MOCOv2 Model](https://drive.google.com/file/d/18rLcNGdteX5LwT7sv_F7HWr12HpVEzVe/view?usp=sharing) | Pretrained ResNet-50 model trained using MOCOv2 for use in our simmilarity loss for domains other then human faces during training.

By default, we assume that all auxiliary models are downloaded and saved to the directory `pretrained_models`. However, you may use your own paths by changing the necessary values in `configs/path_configs.py`. 

## Training
To train the e4e encoder, make sure the paths to the required models, as well as training and testing data is configured in `configs/path_configs.py` and `configs/data_configs.py`.
#### **Training the e4e Encoder**
```
python scripts/train.py \
--dataset_type cars_encode \
--exp_dir new/experiment/directory \
--start_from_latent_avg \
--use_w_pool \
--w_discriminator_lambda 0.1 \
--progressive_start 20000 \
--id_lambda 0.5 \
--val_interval 10000 \
--max_steps 200000 \
--stylegan_size 512 \
--stylegan_weights path/to/pretrained/stylegan.pt \
--workers 8 \
--batch_size 8 \
--test_batch_size 4 \
--test_workers 4 
```

## Inference
Having trained your model, you can use `scripts/inference.py` to apply the model on a set of images.   
For example, 
```
python scripts/inference.py \
--images_dir=/path/to/images/directory \
--save_dir=/path/to/saving/directory \
path/to/checkpoint.pt 
```

## Acknowledgments
This code borrows heavily from [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel)

## Citation
If you use this code for your research, please cite our paper <a href="https://arxiv.org/abs/2102.02766">Designing an Encoder for StyleGAN Image Manipulation</a>:

```
@article{tov2021designing,
  title={Designing an Encoder for StyleGAN Image Manipulation},
  author={Tov, Omer and Alaluf, Yuval and Nitzan, Yotam and Patashnik, Or and Cohen-Or, Daniel},
  journal={arXiv preprint arXiv:2102.02766},
  year={2021}
}
```
