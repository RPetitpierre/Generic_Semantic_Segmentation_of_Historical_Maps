import cv2, os, glob, tqdm
from skimage.feature import local_binary_pattern, hog
from scipy.stats import skew, kurtosis
import numpy as np

def getHOGdescriptors(image: np.ndarray, n_bins:int = 24, patch_size:int = 50, hog_cell_size:int = 5):

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    divider = int(patch_size//hog_cell_size)

    fd = hog(gray, orientations=n_bins, pixels_per_cell=(hog_cell_size, hog_cell_size), cells_per_block=(1, 1), 
                        visualize=False, feature_vector=False)

    hog_feat = np.reshape(fd[:,:,0,0], (fd.shape[0], fd.shape[1], n_bins))

    hog_feat = hog_feat[:(hog_feat.shape[0]//divider)*divider, :(hog_feat.shape[1]//divider)*divider]
    hog_feat = np.reshape(hog_feat, (hog_feat.shape[0]//divider, divider, hog_feat.shape[1]//divider, divider, n_bins))

    hog_feat = np.sum(np.sum(hog_feat, axis=1), axis=2)
    hog_feat = np.reshape(hog_feat, (hog_feat.shape[0]*hog_feat.shape[1], n_bins))

    return hog_feat


def getLBPdescriptors(image: np.ndarray, patch_size:int = 50, lbp_radius:int = 2, histogram_bins: int = 12):
    
    assert (histogram_bins <= 25), "The number of bins in histogram cannot exceed 25"

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    ret, thr = cv2.threshold(gray.copy(), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    lbp = (local_binary_pattern(thr, 8*lbp_radius, lbp_radius, 'uniform')).astype('uint8')
    
    texture_feat = []
    for i in range((lbp.shape[0]//patch_size)):
        for j in range((lbp.shape[1]//patch_size)):
            patch = lbp[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            texture_feat.append(np.histogram(patch, bins=12, range=(0, 17))[0]/patch_size**2)
    texture_feat = np.asarray(texture_feat)
    
    return texture_feat


def getColorDescriptors(image: np.ndarray, patch_size:int = 50, histogram_bins: int = 255):
    
    assert (len(image.shape) == 3), "The image should have 3 color channels"
    
    b, g, r = cv2.split(image)

    color_feat = []
    for color in [b, g, r]:
        features = []
        for i in range((image.shape[0]//patch_size)):
            for j in range((image.shape[1]//patch_size)):
                patch = color[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                features.append(np.histogram(patch, bins=histogram_bins, range=(0, 255))[0])
        features = np.asarray(features)
        mean = np.argmax(features, axis=1)
        std = np.std(features, axis=1)
        skewness = skew(features, axis=1)
        kurt = kurtosis(features, axis=1)
        color_feat.append(np.array([mean, std, skewness, kurt]))
        
    color_feat = np.concatenate(color_feat).T
    
    return color_feat

def compactFeat(features):
    ''' Regroup the features for visualisation '''

    features_ = np.concatenate([np.asarray(features)[:, 0:1],
                np.asarray(features)[:, 4:5],
                np.asarray(features)[:, 8:9],
                np.asarray(features)[:, 1:4] + np.asarray(features)[:, 5:8] + np.asarray(features)[:, 9:12],
                np.asarray(features)[:, 12:18] + np.flip(np.asarray(features)[:, 18:24], axis=1),
                np.asarray(features)[:, 24:]], axis=-1)
    
    features_ = np.concatenate([features_[:, :6],
                                features_[:, 6:7] + features_[:, 7:8],
                                features_[:, 7:8] + features_[:, 8:9],
                                features_[:, 8:9] + features_[:, 9:10],
                                features_[:, -5:]], axis=-1)
    
    return features_

