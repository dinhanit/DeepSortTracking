<h1 align="center">FACE RECOGNITION PROJECT</h1>
# Welcome to our Project!

## Table of Contents

- [Changelog](#changelog)
- [Install](#install)
- [Usage](#usage)
	- [Get started](#get-start)


## Changelog 
* Jul. 1, 2023: v0 released.
* Aug. 1, 2023: v0.1 released.
* Aug. 16, 2023: v0.2 released.
* Nov. 4, 2023: v1 released.  

## Install

### Requirements
**Make sure GPU, cuda are available on your local machine.**
* Python ≥ 3.8. Recommended 3.10 on our local
* PyTorch and torchvision that matches the PyTorch installation. You can install them together at pytorch.org to make sure of this
```shell
# Install torch with GPU directly 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirement
pip install -r requirements.txt
```

## Usage
Use need to creat a DataSet folder and a Model folder.
```shell
DataSet/FaceData/processed/
DataSet/FaceData/raw/
Models/
```

### Get Start 
You need to get dataset by:
```shell
python GetDataTraining.py
```
Now, we have 2 options:
Training with FaceNet:
```shell
#Train Model
python FaceNet_Train.py

#Create reference
python FaceNet_Create_Reference.py

#Inference
python FaceNet_Inference.py
```

Training with Face_Recognition:
```shell
#Train Model
python Face_Recognition_Train.py

#Inference
python Face_Recognition_Inference.py
```

Combine with tracking:
```shell
#Run FastAPI
python Face_Recognition_FastAPI.py
```