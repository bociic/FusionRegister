# [CVPR 2026] FusionRegister
This is the official PyTorch implementation of the paper **FusionRegister: Every Infrared and Visible Image Fusion Deserves Registration** [[paper]]() [[arxiv]]().

## Contents
- [Requirements and dependencies](#requirements_and_dependencies)
- [Dataset](#dataset)
- [Models](#models)
- [Test](#test)
- [Train](#train)
- [Citation](#citation)
- [Contact](#contact)
- [Acknowledgement](#acknowledgement)


## Requirements and dependencies
The model is built in PyTorch 2.1.0 (Python3.9.19, CUDA12.8).

For installing, follow these intructions:
```
conda create -n pytorch python=3.9
conda activate pytorch
conda install pytorch==2.1.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install requirements.txt
```

Install warmup scheduler
```
cd pytorch-gradual-warmup-lr
python setup.py install
cd ..
```

## Dataset

### Registration dataset
1. Download dataset from the [Registered dataset](https://drive.google.com/drive/folders/1oFTP-aMRJ_IlSP-vdXHi_lacEJskDbOA?usp=sharing) .

2. Unzip files ```dataset``` folder.

3. Make Infrared and Visible image fusion dataset structure be:
```
Registration Dataset
├─ dataset
│ ├─ ir
│ │ ├─ img1.png
│ │ ├─ ......
│ ├─ vi
│ │ ├─ img1.png
│ │ ├─ ......
├─ ......

```

## Models
Pre-trained models can be downloaded from [google drive](https://drive.google.com/drive/folders/1ZRzltkYJpjrbRBWeTYtmZ8wucyftEGQr?usp=sharing).
* *fr.pth*: trained for [MMDRFuse (ACMM 24)](https://github.com/yanglindeng/mmdrfuse) (Main Version).
* *fr_hcl.pth*: trained for [HCLFuse (NeurIPS 25)](https://github.com/LinGuo-H/HCLFuse) (Additional Version).
* *fr_s4.pth*: trained for [S4Fusion (TIP 25)](https://github.com/zipper112/S4Fusion/) (Additional Version).

## Test MMDRFuse Version (Main Version)
1. Clone this github repo
```
git clone https://github.com/bociic/FusionRegister.git
cd FusionRegister
```
2. Prepare testing dataset and modify "input_dir", "target_dir", and "weights" in `./test.py`
3. Run test
```
python test.py --save_result #test MMDRFuse (Main Version)
```
4. The original fused result are saved in `./fused_results`. The registered result are saved in `./registered_results`

## Test S4Fusion Version
1. Download the pretrained weight of image fusion from [S4Fusion](https://github.com/zipper112/S4Fusion/), and download the pretrained weight of [FusionRegister(S4-Version)](https://drive.google.com/file/d/1-ptOkzgcfAv43r-wHsKYZ_yOAA9dwwFz/view?usp=sharing). Put them under `./checkpoints`.
2. Prepare testing dataset and modify "input_dir", "register_weights", and "fusion_weights" in `./test_s4.py`
3. Run test
```
python test_s4.py --save_result #test S4Fusion (TIP 2025)
```
4. The original fused result of S4Fusion are saved in `./fused_results`. The registered result of S4Fusion are saved in `./registered_results`

## Test HCLFuse Version
1. Download the pretrained weight of image fusion from [HCLFuse](https://github.com/LinGuo-H/HCLFuse), and download the pretrained weight of [FusionRegister(HCL-Version)](https://drive.google.com/file/d/1hZ6hRP8xNW4EoA4pmtTLpMyeWpMdeauz/view?usp=sharing). Put them under `./checkpoints`.
2. Prepare testing dataset and modify "input_dir", "register_weights", and "fusion_weights"  in `./test_hcl.py`
3. Run test
```
python test_hcl.py --save_result #test HCLFuse(NeurIPS 2025) 
```
4. The original fused result of HCLFuse are saved in `./fused_results`. The registered result of HCLFuse are saved in `./registered_results`


## Train
1. Clone this github repo
```
git clone https://github.com/ChengxuLiu/MISCFilter.git
cd MISCFilter
pip install requirements.txt
```
2. Prepare training dataset and modify "input_dir" and "fusion_weights" in `./train.py`. If you want to finetune from the pretrained weight you can modify "pretrained_weights" in `./train.py`
3. If you want to train your own version, modify `Load the Fusion Net` and `Get the Original Fused Image` in Line 106 and Line 175 in `train.py`
3. Run training
```
python train.py
```
4. The models are saved in `./experiments`

## Citation
If you find the code and pre-trained models useful for your research, please consider citing our paper. :blush:
```

``` 

## Contact
If you meet any problems, please describe them in issues or contact:
* Congcong Bian: <bociic_jnu_cv@163.com>

## Acknowledgement
The code of FusionRegister is built upon [MISCFilter](https://github.com/ChengxuLiu/MISCFilter),and we express our gratitude to this awesome projects.


