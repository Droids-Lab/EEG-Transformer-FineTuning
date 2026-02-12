<div align="center">

# CBraMod


_A Criss-Cross Brain Foundation Model for EEG Decoding_


[![Paper](https://img.shields.io/badge/arXiv-2412.07236-red)](https://arxiv.org/abs/2412.07236)
[![Paper](https://img.shields.io/badge/Paper-ICLR-008B8B)](https://openreview.net/forum?id=NPNUHgHF2w)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/weighting666/CBraMod)

</div>


<div align="center">
<img src="figure/CBraMod_logo.png" style="width: 15%;" />
</div>


<p align="center">
    🔍&nbsp;<a href="#-about">About</a>
    | 🔨&nbsp;<a href="#-setup">Setup</a>
    | 🚢&nbsp;<a href="#-preprocessing">Preprocessing</a>
    | ⛵&nbsp;<a href="#-finetune">Finetune</a>
    | 🚀&nbsp;<a href="#-model-evaluation">Model Evaluation</a>
</p>

## 🔍 About
**CBraMod** is an EEG foundation model that uses criss-cross transformer for EEG decoding on various BCI tasks. The BCI task to be focused on for this project is autism score binary classification (low, high)
<div align="center">
<img src="figure/model.png" style="width:100%;" />
</div>



## 🔨 Setup

Install other requirements:
```commandline
pip install -r requirements.txt
``` 

## 🚢 Preprocessing (Optional)
The model is already pretrained, pretrained weights are included in the directory.

To preprocess ChongQing data, use the following code:
```commandline
python -m preprocessing.preprocessing_chongqing
```
This step is optional as the directory already includes preprocessed data in "processed/"

## ⛵ Finetune
To finetune CBraMod on the ChongQing dataset, use the following code:
```commandline
python finetune_main.py
```
The default is 5 epochs

## 🚀 Model Evaluation
After finetuning the model, the finetuned model's weights are saved in "new_model/models_weights/". 
A graph of the Training loss and Training accuracy is saved to "figure/"  
