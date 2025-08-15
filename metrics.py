import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from skimage import io, img_as_ubyte
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy
from sklearn.metrics import mutual_info_score
from sewar.full_ref import vifp

def show_images(image_us, image_irm, image_fused_ddfm_em, image_fused_ddfm_palm, image_fused_palm):
    fig, axes = plt.subplots(2, 3, figsize=(10, 10))
    # Afficher l'image 'US' redimensionnée
    axes[0, 0].imshow(image_us, cmap='gray')
    axes[0, 0].set_title("Image US")
    axes[0, 0].axis('off')
    # Afficher l'image 'IRM' redimensionnée
    axes[0, 1].imshow(image_irm, cmap='gray')
    axes[0, 1].set_title("Image IRM")
    axes[0, 1].axis('off')
    # Afficher l'image 'DDFM EM'
    axes[1, 0].imshow(image_fused_ddfm_em, cmap='gray')
    axes[1, 0].set_title("Image Fused (DDFM EM)")
    axes[1, 0].axis('off')
    # Afficher l'image 'DDFM PALM'
    axes[1, 1].imshow(image_fused_ddfm_palm, cmap='gray')
    axes[1, 1].set_title("Image Fused (DDFM PALM)")
    axes[1, 1].axis('off')
    # Afficher l'image 'PALM' redimensionnée
    axes[1, 2].imshow(image_fused_palm, cmap='gray')
    axes[1, 2].set_title("Image Fused (PALM)")
    axes[1, 2].axis('off')
    # Afficher le tout
    plt.tight_layout()
    plt.show()

def compute_entropy(image):
    """
    Entropy (EN) measures the amount of information contained in a fused image on the
    basis of information theory.
    The larger the EN, the more information is contained
    in the fused image and the better the performance of the fusion method. However, EN may
    be influenced by noise; the more noise the fused image contains, the larger the EN. Therefore,
    EN is usually used as an auxiliary metric.
    """
    return shannon_entropy(image)

def compute_sd(image):
    """
    The standard deviation (SD) metric is based on the statistical concept that reflects the
    distribution and contrast of the fused image.
    Regions with high contrast always attract human attention due to the sensitivity of the human
    visual system to contrast.
    Therefore, a fused image with high contrast often results in a large SD, which means that
    the fused image achieves a good visual effect.
    """
    return np.std(image)

def compute_mi(image_us, image_irm, img_fused):
    """
    The mutual information (MI) metric is a quality index that measures the amount of
    information that is transferred from source images to the fused image.
    A large MI metric means that considerable information is transferred from source images
    to the fused image, which indicates a good fusion performance.
    """
    hist_us, _, _ = np.histogram2d(image_us.ravel(), img_fused.ravel(), bins=20)
    hist_irm, _, _ = np.histogram2d(image_irm.ravel(), img_fused.ravel(), bins=20)
    return mutual_info_score(None, None, contingency=hist_us) + mutual_info_score(None, None, contingency=hist_irm)

def compute_vif(image_us, image_irm, fused_image):
    """
    This is a VIF Pixel Based.
    The visual information fidelity (VIF) metric measures the information fidelity of the
    fused image [298], which is consistent with the human visual system.
    """
    return vifp(image_us, fused_image) + vifp(image_irm, fused_image)

def compute_edge_metric(image_us, image_irm, fused_image):
    """
    QAB/F measures the amount of edge information that is transferred from source images
    to the fused image and is based on the assumption that the edge information in the source
    images is preserved in the fused image.
    A large QAB/F means that considerable edge information is transferred to the fused image.
    """
    return np.corrcoef(sobel(fused_image).flatten(), 
                       sobel(image_us).flatten())[0, 1] + np.corrcoef(sobel(fused_image).flatten(), 
                                                                      sobel(image_irm).flatten())[0, 1]

def compute_ssim(image_us, image_irm, fused_image):
    return ssim(image_us, fused_image) + ssim(image_irm, fused_image)

in_folder=r"inputTLSE"
out_folder=r"outputTLSE/recon"   
img_names = sorted(os.listdir(out_folder))
print(img_names)
i = 0
while i + 2 < len(img_names):
    in_base = img_names[i].split("_")[0]
    image_us = io.imread(os.path.join(in_folder, in_base, "us.png"), as_gray=True)
    image_irm = io.imread(os.path.join(in_folder, in_base, "irm.png"), as_gray=True)
    image_fused_ddfm_em = io.imread(os.path.join(out_folder, img_names[i]), as_gray=True)
    image_fused_ddfm_palm = io.imread(os.path.join(out_folder, img_names[i+1]), as_gray=True)
    image_fused_palm = io.imread(os.path.join(out_folder, img_names[i+2]), as_gray=True)
    if(i > 0):
        image_fused_ddfm_em_noise = io.imread(os.path.join(out_folder, img_names[i+2]), as_gray=True)
    scale = 32
    h, w = image_irm.shape
    h = h - h % scale
    w = w - w % scale
    # On resize toutes les images selon notre base (l'image de fusion du modèle DDFM)
    image_us = cv2.resize(image_us, (image_irm.shape[1], image_irm.shape[0]), interpolation=cv2.INTER_CUBIC)
    image_irm = cv2.resize(image_irm, (image_irm.shape[1], image_irm.shape[0]), interpolation=cv2.INTER_CUBIC)
    image_fused_palm = cv2.resize(image_fused_palm, (image_irm.shape[1], image_irm.shape[0]), interpolation=cv2.INTER_CUBIC)
    image_us = image_us[:h, :w]
    image_irm = image_irm[:h, :w]
    image_fused_palm = image_fused_palm[:h, :w]
    # On peut les afficher si désiré
    show_images(image_us, image_irm, image_fused_ddfm_em, image_fused_ddfm_palm, image_fused_palm)
    # Les convertir en Uint8
    image_us = img_as_ubyte(np.clip(image_us, -1, 1))
    image_irm = img_as_ubyte(np.clip(image_irm, -1, 1)) # On s'assure de n'avoir que des éléments entre -1 et 1
    image_fused_ddfm_em = img_as_ubyte(image_fused_ddfm_em)
    image_fused_ddfm_palm = img_as_ubyte(image_fused_ddfm_palm)
    image_fused_palm = img_as_ubyte(np.clip(image_fused_palm, -1, 1))
    print(image_us.dtype, image_irm.dtype, image_fused_ddfm_em.dtype,  image_fused_ddfm_palm.dtype, image_fused_palm.dtype)
    # Calcul des métriques (DDFM - EM)
    en_value_ddfm_em = compute_entropy(image_fused_ddfm_em)
    sd_value_ddfm_em = compute_sd(image_fused_ddfm_em)
    mi_value_ddfm_em = compute_mi(image_us, image_irm, image_fused_ddfm_em)
    vif_value_ddfm_em = compute_vif(image_us, image_irm, image_fused_ddfm_em)
    qabf_value_ddfm_em = compute_edge_metric(image_us, image_irm, image_fused_ddfm_em)
    ssim_value_ddfm_em = compute_ssim(image_us, image_irm, image_fused_ddfm_em)
    # Calcul des métriques (DDFM - PALM)
    en_value_ddfm_palm = compute_entropy(image_fused_ddfm_palm)
    sd_value_ddfm_palm = compute_sd(image_fused_ddfm_palm)
    mi_value_ddfm_palm = compute_mi(image_us, image_irm, image_fused_ddfm_palm)
    vif_value_ddfm_palm = compute_vif(image_us, image_irm, image_fused_ddfm_palm)
    qabf_value_ddfm_palm = compute_edge_metric(image_us, image_irm, image_fused_ddfm_palm)
    ssim_value_ddfm_palm = compute_ssim(image_us, image_irm, image_fused_ddfm_palm)
    # Calcul des métriques (PALM)
    en_value_palm = compute_entropy(image_fused_palm)
    sd_value_palm = compute_sd(image_fused_palm)
    mi_value_palm = compute_mi(image_us, image_irm, image_fused_palm)
    vif_value_palm = compute_vif(image_us, image_irm, image_fused_palm)
    qabf_value_palm = compute_edge_metric(image_us, image_irm, image_fused_palm)
    ssim_value_palm = compute_ssim(image_us, image_irm, image_fused_palm)
    # Création du tableau avec pandas
    data = {
        'Métrique': ['Entropy (EN)', 'Standard Deviation (SD)', 'Mutual Information (MI)', 'Visual Information Fidelity (VIF)', 'QAB/F', 'SSIM'],
        'Valeur DDFM - EM': [en_value_ddfm_em, sd_value_ddfm_em, mi_value_ddfm_em, vif_value_ddfm_em, qabf_value_ddfm_em, ssim_value_ddfm_em],
        'Valeur DDFM - PALM': [en_value_ddfm_palm, sd_value_ddfm_palm, mi_value_ddfm_palm, vif_value_ddfm_palm, qabf_value_ddfm_palm, ssim_value_ddfm_palm],
        'Valeur PALM': [en_value_palm, sd_value_palm, mi_value_palm, vif_value_palm, qabf_value_palm, ssim_value_palm]
    }
    # Création des DataFrames
    df['Dataset'] = in_base  # Ajoute une colonne pour identifier
    df_final = pd.concat([df_final, df], axis=0)
    # Affichage des tableaux
    print("**************************************************************")
    print(f"Comparaisons pour les images du dataset {in_base}")
    print("Affichage des métriques sur la fusion avec les modèles")
    print(df.round(4))
    print("**************************************************************")
    # Définir le nom du fichier de sortie
    output_csv = "fusion_metrics_results.csv"
    # Vérifier si c'est la première itération (pour créer le DataFrame global)
    if i == 0:
        df_final = df  # Initialisation avec le premier tableau
    else:
        df_final = pd.concat([df_final, df], axis=1)  # Concaténation horizontale
    # Sauvegarde du DataFrame final une fois la boucle terminée
    if i + 3 >= len(img_names):  # Vérifier si on est à la dernière itération
        df_final.to_csv(output_csv, index=False)
        print(f"Les résultats ont été enregistrés dans {output_csv}")
    if i == 0:
        i += 2
    else:
        i += 3