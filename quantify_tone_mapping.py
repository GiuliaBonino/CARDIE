"""Quantifies effect of image enhancement operator on the luminance distribution
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.optimize import curve_fit


def Luminance(imag,
            r_r_new = 0.299,
            r_g_new = 0.587,
            r_b_new = 0.114):

	R=imag[:,:,0]
	G=imag[:,:,1]
	B=imag[:,:,2]

	lum = r_r_new*R + r_g_new*G + r_b_new*B 

	return lum

def mse_error(y0,y1):
	return np.mean(np.abs(y0-y1))

def func_lin(x, a, b):
    return a + x*b

def func_pow(x, a, b, c):
    return a*x**b + c

def func_pow_red(x, a, b):
    return a*x**b 

def func_log(x, a, b, c):
    return a * np.log(x*b) + c

def func_exp(x, a, b, c):
    return a * np.exp(x*b) + c

def func_poly(x, a, b, c, d):
    return  a + b*x + c*x**2 + d*x**3

def func_non_linear1(x, a, b):
    return  x**a / (x**a + b**a)

def func_non_linear2(x, a, b, c):
    return  x**a / (x**b + c)

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
		scale_factor = 2.0**(n_bits) - 1.0
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


if __name__ == '__main__':

	rescale = (128,128)

	n_bits = 16

	scale_factor =  2**(n_bits) - 1

	input_folder = 'D:/fivekdataset_big/input/train/'

	input_folder = get_all_image_names(input_folder)[0:10]

	output_folder = 'D:/fivekdataset_big/hdrnet/train/'

	output_folder = get_all_image_names(output_folder)[0:10] 

	#tab_file_path = "D:/fivekdataset_big/output/mit5k_fits_expert.csv"
	tab_file_path = 'D:/HDRplus_dataset/train/delete_me.csv'  #"D:/fivekdataset_big/expertc/mit5k_fits_ReinhardDevlinTMO_non_linear1.csv"

	N_sub = 100

	with open(tab_file_path, 'w') as tab_file:
		#tab_file.write("image_name,image_idx,a,b,c,mse_err\n")
		#tab_file.write("image_name,image_idx,a,b,c,d, mse_err\n")
		tab_file.write("image_name,image_idx,a,b,mse_err\n")

		k = 0 
		for input_file, output_file in zip(input_folder, output_folder):


			print("input_file", input_file)

			image_input, filename_input = load_image(input_file,is_H5=False,n_bits=16)
			image_input = cv2.resize(image_input,rescale)
			lum_input = Luminance(image_input).flatten()
			lum_input_indx =  np.argsort(lum_input)
			fit_step = int(len(lum_input) / N_sub) + 1
			lum_input_indx = lum_input_indx[0::fit_step]
			lum_input = lum_input[lum_input_indx]

			image_output, filename_output = load_image(output_file,is_H5=False,n_bits=16)
			image_output = cv2.resize(image_output,rescale)
			lum_output = Luminance(image_output).flatten()
			lum_output = lum_output[lum_input_indx]


			popt, pcov = curve_fit(func_non_linear1, lum_input, lum_output, nan_policy='omit', maxfev=100000)

			
			#plt.plot(lum_input, func_poly(lum_input, *popt), 'g--', label='fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f' % tuple(popt))

			
			mse_err = mse_error(lum_output, func_non_linear1(lum_input, *popt) )
			
			plt.title("HD MSE = " + str(np.round(mse_err,2)))
			plt.legend()
			plt.scatter(lum_input, lum_output)
			plt.plot(lum_input, func_non_linear1(lum_input, *popt), 'g--', label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
			#plt.title(str(filename_input))
			plt.show()
			#sys.exit(0)
			
			

			#plt.hist(lum_input, 100, [0,1.0], alpha=0.5, weights=np.ones(len(lum_input)) / len(lum_input), label="input")
			#plt.hist(lum_output, 100, [0,1.0], alpha=0.5, weights=np.ones(len(lum_output)) / len(lum_output), label="output")
			#plt.legend()
			#plt.show()
			#ssys.exit(0)


			#save_str = filename_input + ',' + str(k) + ',' + str(popt[0]) + ',' + str(popt[1]) + ',' + str(popt[2]) + ','  + str(mse_err) +'\n'
			save_str = filename_input + ',' + str(k) + ',' + str(popt[0]) + ',' + str(popt[1]) + ',' + str(mse_err) +'\n'
			#save_str = filename_input + ',' + str(k) + ',' + str(popt[0]) + ',' + str(popt[1]) + ',' + str(popt[2]) + ',' + str(popt[3]) + ',' + str(mse_err) +'\n'
			tab_file.write(save_str)
			k+=1




