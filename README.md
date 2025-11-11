# Benchmarking Laryngeal Neoplasm Segmentation: A Multicenter Dataset and an Effective Method

*Authors:*  
*Guanghui Yue, Shangjie Wu, Ruxian Tian, Hanhe Lin, Jiaxuan Li, Ting Yuan, Huaiqing Lv, Zhenkun Yu, Ning Mao, and Xicheng Song*

## 1. Preface

- This repository provides code for *[Benchmarking Laryngeal Neoplasm Segmentation: A Multicenter Dataset and an Effective Method](https://ieeexplore.ieee.org/document/11237038)* IEEE TIP 2025.

## 2. Overview

### 2.1 Abstract
While accurate and automatic Laryngeal Neoplasm Segmentation (LNS) can benefit the diagnosis and prevention of laryngeal cancers, existing LNS-related works are very limited due to the lack of public datasets. This paper conducts systematic
research to take the research field a step further. Firstly, we create a multicenter LNS dataset, named as MLN-Seg. Collecting from four hospitals, it has 2,273 laryngeal images with a diversity in resolutions and modalities, where each image is pixel-wise annotated by experienced physicians. Secondly, considering the scarcity of LNS methods and similarity between LNS and Colorectal Polyp Segmentation (CPS) tasks, we collect 15 CPS methods and validate their performance on MLN-Seg. It shows that despite the similarity between the two tasks, existing CPS methods underperform on LNS, especially those with blurry boundaries and camouflaged characteristics. Lastly, considering the LNS challenges, we propose an effective segmentation method, termed Scale-Sensitive Network (S2Net). S2Net scales the feature at each layer of the network up and down and integrates all the scaled features to coarsely localize neoplasm regions. In addition, a Localization Calibration (LC) module is used to refine uncertain areas. By connecting the LC modules from top to down, S2Net can finally accurately segment the laryngeal neoplasms. Extensive tests on MLN-Seg shows that S2Net has better learning ability and generalizability than competing methods. In addition, evaluation on five public datasets shows that S2Net achieves comparable performance in the CPS task.

### 2.2 Framework Overview
<p align="center">
  <img width="2279" height="669" alt="Figure5" src="https://github.com/user-attachments/assets/2c69099f-1c4e-4c16-9a3d-22362d02a897" />
  <br />
  <em>Figure 1: Overview architecture of our proposed S2Net.</em>
</p>

## 3. Proposed Method

The implementation code of our proposed laryngeal neoplasm segmentation method is currently under final organization and validation. We aim to release it along with detailed documentation in the coming weeks. Please check back for updates.

### 3.1. Training/Testing

The training and testing experiments are conducted using [PyTorch](https://pytorch.org/) with one NVIDIA RTX 4090 GPU (or other compatible GPUs).

1. **Configuring your environment (Prerequisites):**
- Installing necessary packages: `pip install -r requirements.txt`.

2. **Downloading necessary data:**
- Downloading MLN-Seg multicenter dataset (see Section 4.1 for details).
- Downloading backbone pretrained weights [PVTv2](https://github.com/whai362/PVT](https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth) and place it in `./models/pretrained/`.

3. **Training:**
- Run the training script: `python S2Net_train.py`.  
- You can adjust parameters (e.g., `--batch_size`, `--lr`, `--epochs`) based on your hardware in the configuration file `options_lary.py`.

4. **Testing:**
- To evaluate the model, run: `python S2Net_test.py --resume ./cpt/S2Net_best.pth.
- Results will be saved in `./results/`.

## 4. Data and Results

### 4.1. MLN-Seg  Dataset
Our proposed MLN-Seg dataset (for laryngeal neoplasm segmentation) is publicly available for academic research. You can download it via the following link:
- **Download Link**: [Baidu Drive](https://pan.baidu.com/s/1mm7Qdpmp1vXxsdQCwaQURQ?pwd=fp63)
- **Extraction Code**: fp63

### 4.2. Experimental Results
All experimental results (including qualitative comparison maps between our S2Net and other state-of-the-art methods). You can download the complete results via:
- **Download Link**: [Baidu Drive](https://pan.baidu.com/s/1sMLALyn6oNVEsCQ2vWrmmA?pwd=z4g2)
- **Extraction Code**: z4g2

## 5. Citation

Please cite our paper if you find the work useful, thanks!

```bibtex
@ARTICLE{11237038,
  author={Yue, Guanghui and Wu, Shangjie and Tian, Ruxian and Lin, Hanhe and Li, Jiaxuan and Yuan, Ting and Lv, Huaiqing and Yu, Zhenkun and Mao, Ning and Song, Xicheng},
  journal={IEEE Transactions on Image Processing}, 
  title={Benchmarking Laryngeal Neoplasm Segmentation: A Multicenter Dataset and an Effective Method}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Neoplasms;Image segmentation;Medical diagnostic imaging;Hospitals;Feature extraction;Image resolution;Benchmark testing;Transformers;Location awareness;Larynx;Laryngoscope;laryngeal neoplasm segmentation;mixed-scale fusion;localization calibration},
  doi={10.1109/TIP.2025.3628504}}
