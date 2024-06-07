Python  3.8(ubuntu18.04)

RTX 2080 Ti(11GB) * 1

## Getting Started
Clone the repo:
  ```bash
  git clone https://github.com/aurelianocyp/DECA
  cd DECA
  ```  
  ```bash
 pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
  pip install -r requirements.txt
  ```

For visualization, we use our rasterizer that uses pytorch JIT Compiling Extensions. If there occurs a compiling error, you can install [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md) instead and set --rasterizer_type=pytorch3d when running the demos.

### Usage
1. Prepare data（需要albedo就做这一步，不需要就不做）
    run script: 
    ```bash
    bash fetch_data.sh
    ```
    <!-- or manually download data form [FLAME 2020 model](https://flame.is.tue.mpg.de/download.php) and [DECA trained model](https://drive.google.com/file/d/1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje/view?usp=sharing), and put them in ./data  -->  
    (Optional for Albedo)   
    follow the instructions for the [Albedo model](https://github.com/TimoBolkart/BFM_to_FLAME) to get 'FLAME_albedo_from_BFM.npz', put it into ./data

2. Run demos  （下载网盘里的generic model pkl 和deca model tar到data文件夹中）将我的图片放在主目录下，删除TestSamples文件夹
    ```bash
    python demos/demo_reconstruct.py -i myphoto.png  
    ```   
    to visualize the predicted 2D landmanks, 3D landmarks (red means non-visible points), coarse geometry, detailed geometry, and depth.   
    <p align="center">   
    <img src="Doc/images/id04657-PPHljWCZ53c-000565_inputs_inputs_vis.jpg">
    </p>  
    <p align="center">   
    <img src="Doc/images/IMG_0392_inputs_vis.jpg">
    </p>  
    You can also generate an obj file (which can be opened with Meshlab) that includes extracted texture from the input image.  

    Please run `python demos/demo_reconstruct.py --help` for more details. 

    b. **expression transfer**   
    ```bash
    python demos/demo_transfer.py
    ```   
    Given an image, you can reconstruct its 3D face, then animate it by tranfering expressions from other images. 
    Using Meshlab to open the detailed mesh obj file, you can see something like that:
    <p align="center"> 
    <img src="Doc/images/soubhik.gif">
    </p>  
    (Thank Soubhik for allowing me to use his face ^_^)   
    
    Note that, you need to set '--useTex True' to get full texture.   

    c. for the [teaser gif](https://github.com/YadiraF/DECA/results/teaser.gif) (**reposing** and **animation**)
    ```bash
    python demos/demo_teaser.py 
    ``` 
    
    More demos and training code coming soon.

## Evaluation
DECA (ours) achieves 9% lower mean shape reconstruction error on the [NoW Challenge](https://ringnet.is.tue.mpg.de/challenge) dataset compared to the previous state-of-the-art method.  
The left figure compares the cumulative error of our approach and other recent methods (RingNet and Deng et al. have nearly identitical performance, so their curves overlap each other). Here we use point-to-surface distance as the error metric, following the NoW Challenge.  
<p align="left"> 
<img src="Doc/images/DECA_evaluation_github.png">
</p>

For more details of the evaluation, please check our [arXiv paper](https://arxiv.org/abs/2012.04012). 

## Training
1. Prepare Training Data

    a. Download image data  
    In DECA, we use [VGGFace2](https://arxiv.org/pdf/1710.08092.pdf), [BUPT-Balancedface](http://www.whdeng.cn/RFW/Trainingdataste.html) and [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)  

    b. Prepare label  
    [FAN](https://github.com/1adrianb/2D-and-3D-face-alignment) to predict 68 2D landmark  
    [face_segmentation](https://github.com/YuvalNirkin/face_segmentation) to get skin mask  

    c. Modify dataloader   
    Dataloaders for different datasets are in decalib/datasets, use the right path for prepared images and labels. 

2. Download face recognition trained model  
    We use the model from [VGGFace2-pytorch](https://github.com/cydonia999/VGGFace2-pytorch) for calculating identity loss,
    download [resnet50_ft](https://drive.google.com/file/d/1A94PAAnwk6L7hXdBXLFosB_s0SzEhAFU/view),
    and put it into ./data  

3. Start training

    Train from scratch: 
    ```bash
    python main_train.py --cfg configs/release_version/deca_pretrain.yml 
    python main_train.py --cfg configs/release_version/deca_coarse.yml 
    python main_train.py --cfg configs/release_version/deca_detail.yml 
    ```
    In the yml files, write the right path for 'output_dir' and 'pretrained_modelpath'.  
    You can also use [released model](https://drive.google.com/file/d/1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje/view) as pretrained model, then ignor the pretrain step.





