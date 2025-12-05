# TCS-Net
This repository is the `pytorch` code for paper `"From Patch to Pixel: A Transformer-based Hierarchical Framework for Compressive Image Sensing"`.  
## 1. Introduction ##
**1) Datasets**  
111111111
Training set: [`BSDS500`](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html), testing sets: [`McM18`](https://www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.html), [`LIVE29`](http://live.ece.utexas.edu/research/Quality/), [`General100`](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html) and [`OST300`](http://mmlab.ie.cuhk.edu.hk/projects/SFTGAN/).  

**2）Project structure**
```
(TCS-Net)
|-dataset
|    |-train  
|        |-BSDS500 (.jpg)  
|    |-test  
|        |-McM18  
|        |-LIVE29  
|        |-General100  
|        |-OST300  
|-reconstructed_images
|    |-McM18
|        |-grey
|            |-... (Testing results .png)
|        |-rgb
|            |-... (Testing results .png)
|    |-... (Testing sets)
|    |-Res_(...).txt
|-models
|    |-__init__.py  
|    |-net.py  
|    |-modules.py  
|-trained_models  
|    |-1  
|    |-4  
|    |-... (Sampling rates)
|-config 
|    |-__init__.py  
|    |-config.py  
|    |-loader.py  
|-test.py  
|-train.py
|-train.sh
```

**3) Competting methods**  

|Methods|Sources|Year|
|:----|:----|:----|
| ![ReconNet](https://latex.codecogs.com/svg.image?\textbf{ReconNet})| [Conf. Comput. Vis. Pattern Recog.](https://ieeexplore.ieee.org/document/7780424/) | 2016 |
| ![LDIT](https://latex.codecogs.com/svg.image?\textbf{LDIT}) | [Proc. Adv. Neural Inf. Process. Syst.](https://dl.acm.org/doi/10.5555/3294771.3294940) | 2017 |
| ![LDAMP](https://latex.codecogs.com/svg.image?\textbf{LDAMP}) | [Proc. Adv. Neural Inf. Process. Syst.](https://dl.acm.org/doi/10.5555/3294771.3294940) | 2017 |
| ![ISTA-Net (plus)](https://latex.codecogs.com/svg.image?\textbf{ISTA-Net}^{&plus;}) | [Conf. Comput. Vis. Pattern Recog.](https://ieeexplore.ieee.org/document/8578294) | 2018 |
| ![CSGAN](https://latex.codecogs.com/svg.image?\textbf{CSGAN}) | [Proc. Int. Conf. Mach. Learn.](http://proceedings.mlr.press/v97/wu19d.html) | 2019 |
| ![CSNet (plus)](https://latex.codecogs.com/svg.image?\textbf{CSNet}^{&plus;}) | [Trans. Image Process.](https://ieeexplore.ieee.org/document/8765626/) | 2020 |
| ![AMP-Net](https://latex.codecogs.com/svg.image?\textbf{AMP-Net}) | [Trans. Image Process.](https://ieeexplore.ieee.org/document/9298950) | 2021 |
|CSformer| arXiv | 2022 |


**4) Performance demonstrates**  

Visual comparisons of reconstruction images (original images are drawn from dataset `LIVE29`):

<div align=center><img src="https://github.com/CompressiveLab/TCS-Net/blob/main/sample/rgb.png"/></div>  

## 2. Useage ##  
**1) Re-training TCS-Net.**  

* Put the `BSDS500` and `VOC2012` images into `./dataset/train/`.  
* e.g., If you want to train TCS-Net at sampling rate `τ = 0.1` with `GPU No.0`, please run the following command. The train set will be automatically packaged and our model will be trained with its default parameters (please make sure you have enough GPU RAM):  
```
python train.py --rate 0.1 --GPU 0
```
* You can also run our shell script directly as well, it will automatically train the model under all sampling rates, i.e., `τ ∈ {0.01, 0.04, 0.1, 0.25}`:  
```
sh train.sh
```
* The trained models (.pth) will save in the `trained_models` folder.

**2) Testing TCS-Net.**  
* We provide the trained models so that you can put them under `TCS-Net/trained_models/` and use them for testing directly; all trained TCS-Net models can be found in this [GoogleDrive link](https://drive.google.com/drive/folders/15dRG29V51i8rVraz8TkHtev7N3jLkx0U?usp=sharing); Please note that the `folder's names` are the `100 times of sampling rates`, e.g., the folder named `10` includes trained models at `sampling rate = 0.1`.  

* Put the testing folders into `./dataset/test/`.  
* e.g., if you want to test TCS-Net at sampling rate τ = 0.1 with GPU No.0, please run:  
```
python test.py --rate 0.1 --GPU 0
```  
* After that, the reconstructed images, PSNR and SSIM results will be saved to `./reconstructed_images/`.  
## End ##  

We appreciate your reading and attention. For more details about TCS-Net, please refer to our paper.
