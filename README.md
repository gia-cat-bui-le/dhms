# README

This repository contains official code for our research paper "Rethinking Sampling Strategies for Long-Term Dance Generation with Diffusion Model" (https://dl.acm.org/doi/10.1145/3581783.3611887).

## Environment setup

Our environment is similar to [EDGE](https://github.com/Stanford-TML/EDGE). You may check them for more details.

we also provide an `requirements.txt` file that you can use with pip:
```
pip install -r requirements.txt
```
## Data preparation

Please visit Google Driver to download the AIST++ dataset and put it in the `data_loaders\\d2m\\` folder.

Process AIST++ dataset using:

```
cd data_loaders\d2m\
python create_dataset.py --extract-baseline --extract-jukebox --datapath [DATA_DIR]
```

## Pre-trained weights
We provide the pretrained models here: [pretrained models](https://drive.google.com/drive/folders/1Lrj5FEt7bFFiv_VnfoDFoQgZzfF4X6RJ?usp=sharing). The 'pretrained.zip' file contains the pretrained model and training configurations used to report metrics in our paper. You can put the pretrained model and training configuration file under `[CHECKPOINT_DIR]`.

## Training
Once the AIST++ dataset is downloaded and processed, run the training script:

```

```

## Generating Dance

You can test the model on custom music by downloading them as .wav files into a directory, e.g. `custom_music/` and running:

```
```
You can also test the pretrained model with test set by putting the musics from test set in to custom music directory and run the same.

## Evaluation

You can use the following command to obtain the result reported in our paper:
```

```

## Acknowledgments
Our code is based on [PCMDM](https://github.com/yangzhao1230/newPCMDM) and [EDGE](https://github.com/Stanford-TML/EDGE). Thanks for their greate work!
