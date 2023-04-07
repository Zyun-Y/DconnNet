import sys
import numpy as np
import matplotlib.image as mpimg
import os
sys.path.append('Betti_Compute/')
import ext_libs.Gudhi as gdh

def betti_number(imagely):
	# imagely_copy = mpimg.imread('output_IMG_1.png')
	imagely = imagely.detach().cpu().clone().numpy()
	width,height = imagely.shape
	imagely[width - 1, :] = 0
	imagely[:, height - 1] = 0
	imagely[0, :] = 0
	imagely[:, 0] = 0
	temp = gdh.compute_persistence_diagram(imagely, i = 1)
	betti_number = len(temp)
	# print (betti_number)
	return betti_number