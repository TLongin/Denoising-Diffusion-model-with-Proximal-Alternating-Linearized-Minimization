#**************************************************************************
# Denoising Diffusion Model for Multi-modality Image Fusion with Proximal
# Alternating Linearized Minimization algorithm
# Author: Tom Longin (2025 June)
# University of Toulouse, IRIT
# Email: tom.longin@irit.fr
#
# Copyright (2025): Tom Longin
# 
# Permission to use, copy, modify, and distribute this software for
# any purpose without fee is hereby granted, provided that this entire
# notice is included in all copies of any software which is or includes
# a copy or modification of this software and in all copies of the
# supporting documentation for such software.
# This software is being provided "as is", without any express or
# implied warranty.  In particular, the authors do not make any
# representation or warranty of any kind concerning the merchantability
# of this software or its fitness for any particular purpose."
#************************************************************************** 

# --- Librairies ---
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import torch

# --- Fichiers ---
from .utils_palm import estimate_c, FusionPALM
from .matlab_tools import load_dncnn
from ResizeRight.resize_right import resize
from ResizeRight.interp_methods import cubic

# --- Fonctions ---
def init_PALM(irm, us):
    #Initialisation des paramètres et des variables de PALM
    # Force en array 2D float64 (au cas où ce sont des tensors ou 3D)
    irm = np.array(irm.squeeze(), dtype=np.float64)
    us = np.array(us.squeeze(), dtype=np.float64)
    d = 6
    # Coefficients polynomiaux
    cest, _ = estimate_c(irm, us, d)
    c = np.abs(cest)
    # Normalisation
    ym = irm / irm.max()
    yu = us / us.max()
    # Débruitage (DnCNN)
    xu0 = load_dncnn(yu)
    # Paramètres PALM
    return {
        "ym": ym, "xu0": xu0, "c": c,
        "tau1": 1e-12, "tau2": 1e-15, "tau3": 1e-4, "tau4": 2e-4,
        "d": d, "m_iteration": 1
    }

def fusion_onestep(f_pre, palm_init=None):
    """
    Args:
        f_pre     : prédiction courante du modèle de diffusion [B, 1, H, W]
        palm_init : dictionnaire contenant les paramètres PALM pré-initialisés
        plot      : bool, affiche l'image fusionnée à t=0
    Returns:
        x_fused   : image fusionnée [B, 1, H, W]
    """
    assert palm_init is not None, "palm_init est requis"

    device = f_pre.device
    batch_size = f_pre.shape[0]

    # Extraction des paramètres depuis palm_init
    ym = palm_init["ym"]       # IRM normalisée
    xu0 = palm_init["xu0"]     # US débruitée
    c = palm_init["c"]
    tau1 = palm_init["tau1"]
    tau2 = palm_init["tau2"]
    tau3 = palm_init["tau3"]
    tau4 = palm_init["tau4"]
    d = palm_init["d"]         # Coefficient de super-résolution
    m_iteration = palm_init["m_iteration"]

    # Exécution de PALM (1 seule image, donc pas vectorisée batch)
    x2 = FusionPALM(ym, xu0, c, tau1, tau2, tau3, tau4, d, m_iteration)

    # Conversion en tenseur PyTorch [B, 1, H, W]
    x2_tensor = torch.from_numpy(x2).float().unsqueeze(0).unsqueeze(0).to(device)
    if batch_size > 1:
        x2_tensor = x2_tensor.repeat(batch_size, 1, 1, 1)

    return x2_tensor