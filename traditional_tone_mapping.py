"""Python implementation of ReihandDevlin05
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.exposure


#LUCA: luminance has different coeff to bantele toolbox
def Luminance(imag,
            r_r_new = 0.299,
            r_g_new = 0.587,
            r_b_new = 0.114):

	R=imag[:,:,0]
	G=imag[:,:,1]
	B=imag[:,:,2]

	lum = r_r_new*R + r_g_new*G + r_b_new*B 

	return lum


def load_image(fName,
			  is_H5=True,
			  n_bits = 8):

	if is_H5:
		f = h5py.File(fName, 'r')
		with h5py.File(fName, 'r') as record:
			image_fullres =  np.array(record['input_realbit'])
			filename = str(np.array(record['fname']))[2:-1]
	else:
		image_fullres = cv2.imread(fName, cv2.IMREAD_UNCHANGED)*1.0
		scale_factor = 2**(n_bits) - 1
		image_fullres =  image_fullres / scale_factor   
		filename = fName
	return image_fullres, filename


def get_all_image_names(img_dir):
    """Recursevely gets all the image paths, given a folder path

    Parameters
    ----------
    img_dir : str
            Image path

    Returns
    -------
    file_names: list
        a list of images path 
    """
    
    for root, dirs, files in os.walk(img_dir):
        file_names = [root + "/" + dir for dir in files]
        file_names_filter = list(filter(lambda x: x.split(".")[-1] != "db", file_names ))
    return file_names_filter


"""
delta = 1e-6;
img_delta = log(img + delta);

Lav = exp(mean(img_delta(:)));
"""


def ReinhardDevlinTMO(img,
					rd_m=None,
					rd_f=None,
					rd_a=None,
					rd_c=None,
					bNormalization=True):
	"""
	Input:
	-img: input HDR image
	-rd_m: contrast in [0.3, 1.0]
	-rd_f: intensity in [-8.0, 8.0]
	-rd_a: light adaptation in [0.0, 1.0]
	-rd_c: chromatic adaptation in [0.0, 1.0]
	-bNormalization: if 1 it applies the normalization step as in the original paper.

	Output:
	-imgOut: output tone mapped image in linear domain
	"""
	#img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

	try:
		img_dim = img.shape[2]
		check_13Color = (img_dim == 3) or (img_dim == 1)
	except:
		raise ValueError('The image has to be an RGB or luminance image.')

	try:
		checkNegative = len(np.where(img < 0)) == 0
	except:
		raise ValueError('The image has negative values!')


	L = Luminance(img)
	#Lav = np.percentile(L, 50) #np.exp(np.mean(np.log(L + 1e-6)))
	Lav = np.mean(L) 
	Lav = np.log2(Lav + 1e-9)

	if rd_m is None:
		#Lmax = np.percentile(L, 99)
		Lmax = np.max(L)
		Lmax = np.log2(Lmax + 1e-9)

		#Lmin = np.percentile(L, 1)
		Lmin = np.min(L)
		Lmin = np.log2(Lmin + 1e-9)

		k = (Lmax - Lav) / (Lmax - Lmin)

		rd_m = 0.3 + 0.7 * k**1.4
	else:
		rd_m = np.clip(rd_m, 0.3, 1.0)

	if rd_f is None:
		rd_f = 0.0
	else:
		rd_f = np.clip(rd_f, -8.0, 8.0)

	rd_f = np.exp(-rd_f)

	if rd_c is None:
		rd_c = 0.0
	else:
		rd_c = np.clip(rd_c, 0.0, 1.0)

	if rd_a is None:
		rd_a = 1.0
	else:
		rd_a = np.clip(rd_a, 0.0, 1.0)

	imgOut = np.empty(img.shape)

	for i in range(img_dim):
		Cav = np.mean(np.mean(img[:,:,i]))
		C_i = img[:,:,i]

		I_l = rd_c * C_i + (1. - rd_c) * L
		I_g = rd_c * Cav + (1. - rd_c) * Lav
		I_a = rd_a * I_l + (1. - rd_a) * I_g

		imgOut[:,:,i] = C_i / ( C_i + (rd_f * I_a)**rd_m )
	
	if bNormalization:
		Lout = Luminance(imgOut)
		Lminout = np.min(Lout)
		Lmaxout = np.max(Lout)
		delta = Lmaxout - Lminout
		if delta > 0.0:
			#imgOut = np.clip((imgOut - Lmin) / delta, 0.0, 1.0)
			imgOut = np.clip((imgOut - Lminout) / delta, 0.0, 1.0)
			#imgOut = cv2.normalize(imgOut, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	
	#RemoveSpecials
	imgOut = np.nan_to_num(imgOut, 0.0)

	return imgOut



if __name__ == '__main__':

	input_folder = 'D:/fivekdataset_big/input/test'
	
	input_folder = get_all_image_names(input_folder)#[0:3]


	for input_file  in input_folder:
		image_input, filename_input = load_image(input_file, is_H5=False, n_bits=16)
		print("filename_input", filename_input)
		tm_image  = ReinhardDevlinTMO(image_input,  rd_f= -8.0, bNormalization=True)
	
		gamma = 1.6
		tm_image = np.power(tm_image, 1.0 / gamma)
		tm_img_save = np.array(tm_image*(2.0**16 - 1.0)).astype(np.uint16)
		#tm_img_save = cv2.cvtColor(tm_img_save, cv2.CV_16U)  #skimage.exposure.rescale_intensity(tm_img_save, out_range=(0.0,255.0)).astype(np.uint8)
		cv2.imwrite('D:/fivekdataset_big/ReinhardDevlinTMO/test/' + filename_input.split("/")[-1], tm_img_save)
