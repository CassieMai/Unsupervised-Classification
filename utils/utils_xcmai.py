import cv2
import numpy as np
from skimage import io
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import torch
import os
from skimage.measure import regionprops


def segment_colorfulness(image, mask):
    	# split the image into its respective RGB components, then mask
	# each of the individual RGB channels so we can compute
	# statistics only for the masked region
	(B, G, R) = cv2.split(image.astype("float"))
	R = np.ma.masked_array(R, mask=mask)
	G = np.ma.masked_array(B, mask=mask)
	B = np.ma.masked_array(B, mask=mask)
	# compute rg = R - G
	rg = np.absolute(R - G)
	# compute yb = 0.5 * (R + G) - B
	yb = np.absolute(0.5 * (R + G) - B)
	# compute the mean and standard deviation of both `rg` and `yb`,
	# then combine them
	stdRoot = np.sqrt((rg.std() ** 2) + (yb.std() ** 2))
	meanRoot = np.sqrt((rg.mean() ** 2) + (yb.mean() ** 2))
	# derive the "colorfulness" metric and return it
	return stdRoot+0.5 + (0.5 * meanRoot)

def vis_superpixels(img, segs, label):
    vis = np.zeros(img.shape[:2], dtype="float")

    # loop over each of the unique superpixels
    for v in np.unique(segs):
        # construct a mask for the segment so we can compute image
        # statistics for *only* the masked region
        mask = np.ones(img.shape[:2])
        mask[segs == v] = 0
        # compute the superpixel colorfulness, then update the
        # visualization array
        C = segment_colorfulness(img, mask)
        vis[segs == v] = C

    # scale the visualization image from an unrestricted floating point
    # to unsigned 8-bit integer array so we can use it with OpenCV and
    # display it to our screen
    vis = rescale_intensity(vis, out_range=(0, 255)).astype("uint8")
    # overlay the superpixel colorfulness visualization on the original image
    alpha = 0.6
    overlay = np.dstack([vis] * 3)
    output = img.copy()
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # cv2.imshow("Input", img)
    # cv2.imshow("Visualization", vis)
    # cv2.imshow("Output", output)

    fig, axes = plt.subplots(1, 2, tight_layout=True,  constrained_layout=True)
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title('image', fontsize=8)
    axes[1].imshow(output)
    axes[1].axis('off')
    axes[1].set_title('class ' + str(label) + ': ' + str(np.max(segs)+1) + ' superpixels', fontsize=8)
    fig.savefig('class' + str(label) + 'patches' + str(np.max(segs)+1) +'.png', facecolor='white', edgecolor='red')
    plt.close()

    