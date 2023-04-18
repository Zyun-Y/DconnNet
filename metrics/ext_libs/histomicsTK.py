"""
This is a module for svs files used for the analysis of histology images
"""

import histomicstk as htk

import numpy as np

import skimage.feature
import skimage.io
import skimage.measure
import skimage.color
from skimage.filters import sobel
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.util.shape import view_as_windows

import scipy as sp
from scipy import ndimage

from sklearn.feature_extraction import image
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



def compute_nuclei_centroids(matrix,min_nucleus_area=40,foreground_threshold=140, minimum_area=20):

    """
        A method for extracting a point cloud from an histopathology image where each point correspond
        to the centroid of a nucleus. This function is substantially copied from
        https://digitalslidearchive.github.io/HistomicsTK/examples/nuclei-segmentation.html
        with an additional step for subdividing clusters of nuclei by using the convexity of the extracted mask.

        Indicative parameters to use (empirical):

        * For IvyGap images - min_nucleus_area=5, foreground_threshold=190, minimum_area=20

        * For TCGA images - min_nucleus_area=40, foreground_threshold=100, minimum_area=20


        :param matrix: matrix containig the loaded image
        :type matrix: np.array

        :returns: list of points with each point encoded as a pair of coordinates [x,y]
    """

    im_reference = matrix[:, :, :3]

    # get mean and stddev of reference image in lab space
    mean_ref, std_ref = htk.preprocessing.color_conversion.lab_mean_std(im_reference)

    # if sum(std_ref) < 0.65:
    #     foreground_threshold = 190

    if std_ref[1] < 0.1:
        foreground_threshold = 120


    # perform reinhard color normalization
    im_nmzd = htk.preprocessing.color_normalization.reinhard(im_reference, mean_ref, std_ref)

    #identify hematoxylin and nuclei----------------
    stainColorMap = {'hematoxylin': [0.65, 0.70, 0.29],'eosin':       [0.07, 0.99, 0.11],'dab':         [0.27, 0.57, 0.78],'null':        [0.0, 0.0, 0.0]}

    # specify stains of input image
    stain_1 = 'hematoxylin'   # nuclei stain
    stain_2 = 'eosin'         # cytoplasm stain
    stain_3 = 'dab'          # set to null of input contains only two stains

    # create stain matrix
    W = np.array([stainColorMap[stain_1],
                  stainColorMap[stain_2],
                  stainColorMap[stain_3]]).T

    # perform standard color deconvolution
    im_stains = htk.preprocessing.color_deconvolution.color_deconvolution(im_nmzd, W).Stains
    im_nuclei_stain = im_stains[:, :, 0]

    #identify image mask----------------


    im_fgnd_mask = sp.ndimage.morphology.binary_fill_holes(
    im_nuclei_stain < foreground_threshold)

    # run adaptive multi-scale LoG filter
    min_radius = 10
    max_radius = 15

    im_log_max, im_sigma_max = htk.filters.shape.cdog(
        im_nuclei_stain, im_fgnd_mask,
        sigma_min=min_radius * np.sqrt(2),
        sigma_max=max_radius * np.sqrt(2)
    )

    # detect and segment nuclei using local maximum clustering
    local_max_search_radius = 10

    #to prevent errors when we are processing patches
    if(not (im_fgnd_mask.any() == True)):
        return []

    im_nuclei_seg_mask, seeds, maxima = htk.segmentation.nuclear.max_clustering(
        im_log_max, im_fgnd_mask, local_max_search_radius)


    im_nuclei_seg_mask = htk.segmentation.label.area_open(
        im_nuclei_seg_mask, min_nucleus_area).astype(np.int)

    distance = ndimage.distance_transform_edt(im_nuclei_seg_mask)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((20, 20)),
                                labels=im_nuclei_seg_mask)
    markers = ndimage.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=im_nuclei_seg_mask)

    objProps = skimage.measure.regionprops(labels)
    #-----

    # #uncomment to show the selected nuclei
    # # Display results
    # plt.figure(figsize=(20, 10))
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(skimage.color.label2rgb(im_nuclei_seg_mask, im_reference, bg_label=0), origin='lower')
    # plt.title('Nuclei segmentation mask overlay', fontsize=5)
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow( im_reference )
    # plt.xlim([0, im_reference.shape[1]])
    # plt.ylim([0, im_reference.shape[0]])
    # plt.title('Nuclei bounding boxes', fontsize=5)

    myPoints = []
    for i in range(len(objProps)):
        area = objProps[i].convex_area
        if area > minimum_area:
            loc_center = ndimage.measurements.center_of_mass(objProps[i].convex_image)
            myPoints.append([objProps[i].bbox[1]+loc_center[1], objProps[i].bbox[0]+loc_center[0]])

    #         #uncomment to show the selected nuclei
    #         c = [objProps[i].centroid[1], objProps[i].centroid[0], 0]
    #         width = objProps[i].bbox[3] - objProps[i].bbox[1] + 1
    #         height = objProps[i].bbox[2] - objProps[i].bbox[0] + 1
    #
    #         cur_bbox = {
    #             "type":        "rectangle",
    #             "center":      c,
    #             "width":       width,
    #             "height":      height,
    #         }
    #
    #         plt.plot(c[0], c[1], 'g+')
    #         mrect = mpatches.Rectangle([c[0] - 0.5 * width, c[1] - 0.5 * height] ,
    #                                    width, height, fill=False, ec='g', linewidth=2)
    #         plt.gca().add_patch(mrect)
    #
    #
    # print('Number of objects = ', len(objProps))
    # print('Number of points = ', len(myPoints))
    return myPoints



def split_and_compute_nuclei_centroids(matrix, patch_size=500, min_nucleus_area=40, foreground_threshold=100, minimum_area=20):

    """
        A method for extracting a point cloud from an histopathology image where each point correspond
        to the centroid of a nucleus. The input matrix is split in patches of size patch_size and processed
        via the compute_nuclei_centroids() function.

        Indicative parameters to use (empirical):

        * For IvyGap images - min_nucleus_area=5, foreground_threshold=190, minimum_area=20

        * For TCGA images - min_nucleus_area=40, foreground_threshold=100, minimum_area=20

        :param matrix: matrix containig the loaded image
        :type matrix: np.array

        :param patch_size: length of the patch used for processing the image
        :type patch_size: Integer

        :returns: list of points with each point encoded as a pair of coordinates [x,y]
    """

    dims = matrix.shape
    #split here the matrices
    all_points = []

    last_rows = dims[0]/patch_size
    last_columns = dims[1]/patch_size

    for i in range(last_rows):
        for j in range(last_columns):

            sub_matrix = matrix[(i*patch_size):(i*patch_size)+patch_size, (j*patch_size):(j*patch_size)+patch_size, :]

            try:
                points = compute_nuclei_centroids(sub_matrix,min_nucleus_area=min_nucleus_area,foreground_threshold=foreground_threshold, minimum_area=minimum_area)
            except:
                points = []

            if(len(points) > 0):
                x,y = zip(*points)

                x = np.array(x)
                y = np.array(y)
                x = x + (j*patch_size)
                y = y + (i*patch_size)

                points = zip(x,y)
                all_points.extend(points)


    sub_matrix = matrix[(last_rows*patch_size):, :, :]
    try:
        points = compute_nuclei_centroids(sub_matrix,min_nucleus_area=min_nucleus_area,foreground_threshold=foreground_threshold, minimum_area=minimum_area)
    except:
        points = []

    if(len(points) > 0):
        x,y = zip(*points)

        x = np.array(x)
        y = np.array(y)
        x = x
        y = y + (last_rows*patch_size)

        points = zip(x,y)
        all_points.extend(points)

    sub_matrix = matrix[:(last_rows*patch_size), ((last_columns)*patch_size):, :]
    try:
        points = compute_nuclei_centroids(sub_matrix,min_nucleus_area=min_nucleus_area,foreground_threshold=foreground_threshold, minimum_area=minimum_area)
    except:
        points = []
        
    if(len(points) > 0):
        x,y = zip(*points)

        x = np.array(x)
        y = np.array(y)
        x = x + (last_columns)*patch_size
        y = y

        points = zip(x,y)
        all_points.extend(points)

    return all_points
