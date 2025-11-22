# DSSM
DSSM: Dynamic Streaming Pipeline with Spatiotemporal Model toward Remote Sensing Object Detection

## Table of Contents
-  Dataset
-  Conda Installation
-  Demo
-  Copyright
### Dataset
The [EMRS Dataset](https://pan.baidu.com/s/1G30vw2NO3WvHO5Q2EViI6g) can be downloaded from Baidu Netdisk. 
The [VisDrone Dataset](https://pan.baidu.com/s/1G30vw2NO3WvHO5Q2EViI6g) can be downloaded from Baidu Netdisk. 

### Conda Installation
We train our models under`python=3.8,pytorch=2.2.2,cuda=11.8`. 

####  1. Install Pytorch and torchvision.
Follow the instruction on  [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

`conda install -c pytorch pytorch torchvision`

####  2. Install other needed packages
   
`pip install -r requirements.txt`

####  3. For more detailed installation steps, please refer to [Ultralytics](https://github.com/ultralytics/ultralytics)


### Demo

train script 
```sh
python main.py --model_cfg "cfg/dssm-s.yaml" --dataset_path "dataset_path" 
```

val script 
```sh
python main.py --model_cfg "cfg/smvm-s.yaml" --dataset_path "dataset_path" --multimodal "image"  --test_only True --resume "weight_path"
```
