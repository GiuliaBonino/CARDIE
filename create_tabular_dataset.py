"""Computes the image descriptors for an image dataset (used for this paper MIT5K and HDRPLUS)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import h5py
import matplotlib.pyplot as plt


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


def lum_interval_hist(distribution,
                        lum_intervals = [0, .25, 0.5, 0.75, 1.0]):


    X=np.linspace(0,1.0, len(distribution))

    color_hist = []

    for i in range(len(lum_intervals)-1):
        indexes = np.where((X >= lum_intervals[i]) & (X <= lum_intervals[i+1]))
        color_hist.append(sum(distribution[indexes]))
    
    color_hist = np.array(color_hist)

    return color_hist





def color_interval_hist(distribution,
                        color_intervals = [0, np.pi / 3.0, 2.0*np.pi / 3.0, np.pi, 4.0*np.pi / 3.0,  5.0*np.pi / 3.0, 2 * np.pi]):

    """Computes percentage of hue distribution contained in each color interval 


    Parameters
    ----------
    distribution : list of numpy arrays
            hue distributions for input images


    color_intervals: list of floats, optional
            intervals in which the percentages are calculated


    Returns
    -------
    color_hist : list of floats
            percentages of hue distribution contained in each color interval
       
    """

    X=np.linspace(0,2*np.pi, len(distribution))

    delta_angle = np.pi / 12.0 #(color_intervals[1] - color_intervals[0]) / 2.0

    color_hist = []

    for i in range(len(color_intervals)-1):
    	#print(i,color_intervals[i])

    	if i ==0:
    		indexes = np.where((X <= color_intervals[0] + delta_angle) | (X >= color_intervals[-1] - delta_angle))
    	else:
    		indexes = np.where((X <= color_intervals[i] + delta_angle) & (X >= color_intervals[i] - delta_angle))

    	color_hist.append(sum(distribution[indexes]))
    	#indexes = np.where((X >= color_intervals[i]) & (X < color_intervals[i+1]))

    color_hist = np.array(color_hist)

    #color_hist[0] = color_hist[0] + color_hist[-1]

    #color_hist = color_hist[:-1]

    return color_hist


def Luminance(imag,
            r_r_new = 0.299,
            r_g_new = 0.587,
            r_b_new = 0.114):

	R=imag[:,:,0]
	G=imag[:,:,1]
	B=imag[:,:,2]

	lum = r_r_new*R + r_g_new*G + r_b_new*B 

	return lum

def RGB_to_OPP(imag):
    """Transforms image from RGB space to opponent space

    Parameters
    ----------
    imag : 3 channels, numpy array
        input image

    Returns
    -------
    opp_space: 3 channels, numpy array
        transformed image in opponent space
    """

    R=imag[:,:,0]
    G=imag[:,:,1]
    B=imag[:,:,2]

    O1=(R+G+B-1.5)/1.5
    O2=(R-G)
    O3=(R+G-2*B)/2

    opp_space = [O1,O2,O3]
    return opp_space


def OPP_hue(opp_im):
    """Computes Hue angular distribution and intesity map of an image in opponent space, 
        following the convention of angles between 0 and 2pi

    Parameters
    ----------
    opp_im : 3 channels, numpy array
        image in opponent space

    Returns
    -------
    hue: numpy array
        hue distribution

    intensity_map: numpy array
        intesity map
    """

    hue=np.arctan2(opp_im[2],opp_im[1]).flatten() #in radians between -pi and pi

    
    hue=np.array([x if x > 0 else (2*np.pi+x) for x in hue]) #makes everything in 0 and 2pi
    intensity=np.sqrt(opp_im[2]*opp_im[2] + opp_im[1]*opp_im[1]).flatten()
    return hue, intensity

def as_csv(array):

    return ','.join([str(i) for i in array]) 


if __name__ == '__main__':

	#image_folder = 'D:/Duc_VAE_project/expert_VAE/normal'
	#image_folder = 'D:/praveen_whitepaper/all'
	#image_folder = 'D:/praveen_whitepaper/validation_set'
	image_folder = 'D:/fivekdataset_big/input/train'
	#image_folder = 'D:/HDRplus_dataset/train/input'
	#lum_intervals = [0,0.02,0.04,0.06,0.08]
	#lum_intervals = [0.0, 0.0005, 0.001, 0.0015, 0.0020]
	#color_intervals = [0, np.pi / 3.0, np.pi, 5.0*np.pi / 3.0, 2 * np.pi]
	#color_names = ['red', 'yellow', 'green', 'azure', 'blue', 'violet', 'rose']

	tab_file_path = "./Histrogram_test_luminance/mit5k_delete_this.csv"

	image_names=get_all_image_names(image_folder)


	N_sub = 100

	image_subset = np.random.choice(image_names, size=N_sub, replace=False)

	N_bins = 200
	rescale = (128,128)

	#print(image_subset)

	#luminance_subsets = np.zeros(rescale[0]*rescale[1])
	luminance_subsets = np.zeros([rescale[0]*rescale[1]])

	for image_name in image_subset:
		img, filename = load_image(image_name,is_H5=False, n_bits=16)
		#sys.exit(0)
		print("image_name", image_name)
		img = cv2.resize(img,rescale) #to ensure a fair comparison among difference with different resolution
		#img= np.reshape(img,(-1,img.shape[2]))

		#print(len(np.where(img==1)[0]))
		#print(len(np.where(img==0)[0]))

		lum = Luminance(img).flatten()
		luminance_subsets += lum

	luminance_subsets = luminance_subsets / N_sub
	weighted_histo_lum=np.histogram(luminance_subsets,N_bins,[0,255.0])[0]

	lum_percentile = [20,50,80]
	lum_intervals = np.array([np.percentile(luminance_subsets, lum_percentile[0]),
							  np.percentile(luminance_subsets, lum_percentile[1]),
							  np.percentile(luminance_subsets, lum_percentile[2])])
	print("lum_thersholds", lum_intervals)

	plt.hist(luminance_subsets,N_bins,[0,1.0], alpha=0.5, weights=np.ones(len(luminance_subsets)) / len(luminance_subsets))
	plt.vlines(lum_intervals[0], 0, 0.3, colors = 'r')
	plt.vlines(0.16, 0, 0.3, colors = 'black')
	plt.vlines(lum_intervals[2], 0, 0.3, colors = 'g')
	plt.title("Group luminance distribution")
	plt.xlim([0.01,0.4])
	plt.ylim([0.00,0.15])
	plt.title(r'$L(\mathcal{D}_K,p)$')
	plt.text(0.085,0.13, r'$\bar{L}_{\mathrm{low}}^{\mathcal{D}_K}$',size=16, color='r')
	plt.text(0.165,0.13, r'$\bar{L}_{j}$',size=16, color='black')
	plt.text(0.20,0.13, r'$\bar{L}_{\mathrm{high}}^{\mathcal{D}_K}$',size=16, color='g')
	plt.show()



	color_intervals = [0, np.pi / 3.0, 2.0*np.pi / 3.0, 4.0*np.pi / 3.0 , 5.0*np.pi / 3.0 ,  2 * np.pi]

	lum_thersholds = [np.percentile(luminance_subsets, lum_percentile[0]), np.percentile(luminance_subsets, lum_percentile[1]),np.percentile(luminance_subsets, lum_percentile[2])]
	print(lum_thersholds)
	color_thershold = 1.0 / (len(color_intervals)) 
	print(color_thershold)
	


	with open(tab_file_path, 'w') as tab_file:
		tab_file.write("image_name,image_idx,low lum,avg lum,high lum,red,yellow,green,blue,magenta\n")

		for k, image_name in enumerate(image_names):
			img, filename = load_image(image_name,is_H5=False, n_bits=16)
			print("image_name", image_name)

			img = cv2.resize(img,rescale) #to ensure a fair comparison among difference with different resolution

			lum = Luminance(img).flatten()
			weighted_histo_lum=np.histogram(lum,N_bins,[0,255.0])[0]
			lum_median = np.percentile(lum, 50)
			lum_percs = [0,0,0]
			if lum_median <= lum_thersholds[0]:
				lum_percs[0] = 1
			elif (lum_median >= lum_thersholds[2]):
				lum_percs[2] = 1
			else:
				lum_percs[1] = 1
			
			plt.hist(lum,N_bins,[0,255], alpha=0.5, weights=np.ones(len(lum)) / len(lum))
			plt.title("Luminance")
			plt.vlines(lum_median, 0.,0.3, colors = 'r')
			plt.show()

			opp=RGB_to_OPP(img) #transform the image into opponent colorspace
			hue,weigh=OPP_hue(opp) #calculate the hue
			weighted_histo_hue=np.histogram(hue,N_bins,[0,2*np.pi])[0]
			color_distr = color_interval_hist(weighted_histo_hue, color_intervals)
			print("color_distr", color_distr)

			plt.hist(hue,N_bins,[0,2*np.pi], alpha=0.5, weights=np.ones(len(hue)) / len(hue), color='black')
			
			plt.axvspan(0, np.pi / 12.0, ymin=0.0, ymax=0.999, alpha=0.1, color='red')
			plt.axvspan(np.pi / 3.0 - np.pi / 12.0, np.pi / 3.0 + np.pi / 12.0 , ymin=0.0, ymax=0.999, alpha=0.1, color='yellow')
			plt.axvspan(2.0*np.pi / 3.0 - np.pi / 12.0, 2.0*np.pi / 3.0 + np.pi / 12.0 , ymin=0.0, ymax=0.999, alpha=0.1, color='green')
			plt.axvspan(4.0*np.pi / 3.0 - np.pi / 12.0, 4.0*np.pi / 3.0 + np.pi / 12.0 , ymin=0.0, ymax=0.999, alpha=0.1, color='blue')
			plt.axvspan(5.0*np.pi / 3.0 - np.pi / 12.0, 5.0*np.pi / 3.0 + np.pi / 12.0 , ymin=0.0, ymax=0.999, alpha=0.1, color='magenta')
			plt.axvspan(2.0*np.pi - np.pi / 12.0 , 2.0*np.pi, ymin=0.0, ymax=0.999, alpha=0.1, color='red')

		
			plt.title(r'$\mathcal{H}(\theta)$')
			plt.xlim(0.0, 2*np.pi)
			plt.ylim(0.0, 0.18)
			plt.show()
			

			color_percs = np.round(color_distr / np.sum(color_distr),2)
			color_percs = np.where(color_percs <= color_thershold, color_percs, 1.0)
			color_percs = np.where(color_percs > color_thershold, color_percs, 0.0)
			

			save_str = filename + ',' + str(k) + ',' + as_csv(lum_percs) + ',' + as_csv(color_percs) +'\n'

			tab_file.write(save_str)
