# Denoising Diffusion model with Proximal Alternating Linearized Minimization for image fusion

## Usage
We recommend following the instructions provided in the GitHub repository [Denoising Diffusion Model for Multi-Modality Image fusion](https://github.com/Zhaozixiang1228/MMIF-DDFM), as the original code (which we modified for our needs) comes from there. Follow the instructions provided with the files given in our repository. If you want to infer with our DDFM-PALM model and obtain the fusion results in the report, please run
```python
!python sampleTLSE.py
```

## Proximal Alternating Linearized Minimization in Python
The Python implementation of the PALM algorithm is a translation of a Matlab code available [here](https://github.com/TLongin/Fusion-of-Magnetic-Resonance-and-Ultrasound-Images-for-Endometriosis-Detection). As Matlab and Python are two programming languages with certain specific features, we had to implement certain Matlab functions that are not available in Python. These functions are contained in the file `matlab_tools` where we used the GitHub [ResizeRight](https://github.com/assafshocher/ResizeRight) repository to implement the Matlab `resize` function in Python. The checkpoint to use DnCNN model and the code are from the GitHub [DnCNN Pytorch](https://github.com/SaoYan/DnCNN-PyTorch) repository and the `fspecial` function has been directly translated from Matlab. If you want to experiment PALM algorithm for image fusion, please run the file `PALM/palm_main.py`. The `PALM` folder contains only the Python implementation of the PALM algorithm and is completely separate from the DDFM model.

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
