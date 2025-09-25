# Denoising Diffusion model with Proximal Alternating Linearized Minimization for image fusion

## Abstract
This repository aims to present a fusion fo two models for images fusion. The first is [Fusion of Magnetic Resonance and Ultrasound Images for Endometriosis Detection](https://github.com/TLongin/Fusion-of-Magnetic-Resonance-and-Ultrasound-Images-for-Endometriosis-Detection) and the second is [Denoising Diffusion Model for Multi-Modality Image fusion](https://github.com/Zhaozixiang1228/MMIF-DDFM). We aims to add a diffusion process in PALM or, in the same way, to replace Expectation Maximization algorithm in DDFM by PALM. 

## Report
For further details regarding this work, please refer to the report [Fusion d'images par résonance magnétique et ultrasons](https://drive.google.com/file/d/1EqD42Iw54JGWqdoAzLa9iBXeZQF84VMq/view?usp=drive_link) (french version).

## Installation
We recommend following the instructions provided in the GitHub repository [Denoising Diffusion Model for Multi-Modality Image fusion](https://github.com/Zhaozixiang1228/MMIF-DDFM), as the original code (which we modified for our needs) comes from there [2]. Follow the instructions provided with the files given in our repository. Please note that we used Python version 3.12 and not 3.8. It is advisable to install the packages listed in the Github repository (`requirement.txt` file) one by one and not all at once with the command indicated :
```bash
pip install requirements.txt
```
without specifying the version so that any dependency issues between different packages are automatically resolved. This process is long and tedious, but it is the only way to ensure that all of the packages have the correct versions, without dependency issues.

## Usage
If you want to infer with our DDFM-PALM model and obtain the fusion results in the report, please run in a Jupyter Notebook (or Jupyter Lab) :
```python
!python sampleTLSE.py
```
Or in a command terminal :
```bash
python sampleTLSE.py
```
**Warning** : Please note that the use of a GPU (or GPU cluster) is required.

## Proximal Alternating Linearized Minimization in Python
The Python implementation of the PALM algorithm is a translation of a Matlab code available [here](https://github.com/TLongin/Fusion-of-Magnetic-Resonance-and-Ultrasound-Images-for-Endometriosis-Detection). As Matlab and Python are two programming languages with certain specific features, we had to implement certain Matlab functions that are not available in Python. These functions are contained in the file `matlab_tools` where we used the GitHub [ResizeRight](https://github.com/assafshocher/ResizeRight) repository to implement the Matlab `resize` function in Python. The checkpoint to use DnCNN model and the code are from the GitHub [DnCNN Pytorch](https://github.com/SaoYan/DnCNN-PyTorch) repository and the `fspecial` function has been directly translated from Matlab. If you want to experiment PALM algorithm for image fusion, please run the file `PALM/palm_main.py`. The `PALM` folder contains only the Python implementation of the PALM algorithm and is completely separate from the DDFM model.

## References
[1]  Oumaima El Mansouri, Fabien Vidal, Adrian Basarab, Pierre Payoux, Denis Kouamé, and Jean-Yves Tourneret. Fusion of magnetic resonance and ultrasound images for endometriosis detection. IEEE Transactions on Image Processing, 2020.  [Github repository](https://github.com/TLongin/Fusion-of-Magnetic-Resonance-and-Ultrasound-Images-for-Endometriosis-Detection)

[2] Zixiang Zhao, Haowen Bai, Yuanzhi Zhu, Jiangshe Zhang, Shuang Xu, Yulun Zhang, Kai Zhang, Deyu Meng, Radu Timofte, and Luc Van Gool. Ddfm : Denoising diffusion model for multi-modality image fusion, 2023. [Github repository](https://github.com/Zhaozixiang1228/MMIF-DDFM)

## Citations
If you use the code or dataset, please cite the papers as below :
```bibtex
@misc{zhao2023ddfmdenoisingdiffusionmodel,
      title={DDFM: Denoising Diffusion Model for Multi-Modality Image Fusion}, 
      author={Zixiang Zhao and Haowen Bai and Yuanzhi Zhu and Jiangshe Zhang and Shuang Xu and Yulun Zhang and Kai Zhang and Deyu Meng and Radu Timofte and Luc Van Gool},
      year={2023},
      eprint={2303.06840},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2303.06840}, 
}

@article{9018380,
  author={El Mansouri, Oumaima and Vidal, Fabien and Basarab, Adrian and Payoux, Pierre and Kouamé, Denis and Tourneret, Jean-Yves},
  journal={IEEE Transactions on Image Processing}, 
  title={Fusion of Magnetic Resonance and Ultrasound Images for Endometriosis Detection}, 
  year={2020},
  volume={29},
  number={},
  pages={5324-5335},
  keywords={Spatial resolution;Magnetic resonance imaging;Image fusion;Diseases;Magnetic resonance;Image fusion;magnetic resonance imaging;ultrasound imaging;super-resolution;despeckling;proximal alternating linearized minimization},
  doi={10.1109/TIP.2020.2975977}}
```
