# From Dicom File, makes resized numpy files #

import numpy as np
import pydicom
import cv2
import matplotlib.pylab as plt
import os

root_dir = "/media/compu/ssd2t/cxr_all/cxr"
dest_dir = "./data"    
rsize = (224,224)

all_files = []

for (path, dirs, files) in os.walk(root_dir):

    for file in files:
        path_file = path[len(root_dir):] + '/' + file

        if not os.path.exists(dest_dir + "/" + path[len(root_dir):]):
            os.makedirs(dest_dir + "/" + path[len(root_dir):])
        if path_file[-3:] == "dcm":
            all_files.append(path_file)


def main():

    for file in all_files:
        print (file)    
        dco = pydicom.dcmread(root_dir + file)       # dco: dicom_content
        arr = dco.pixel_array
        # print (arr.shape)
        diff = arr.shape[0] - arr.shape[1]

        if diff > 0:    
            if round(diff/2) * 2 != diff:
                arr = np.delete(arr, arr.shape[0]-1, axis = 0)
                arr = np.delete(arr, range(0, int((diff/2) - 0.5)), axis = 0)
                arr = np.delete(arr, range(arr.shape[0] - int((diff/2) - 0.5), arr.shape[0]), axis = 0)
            else:
                arr = np.delete(arr, range(0, int(diff/2)), axis = 0)
                arr = np.delete(arr, range(arr.shape[0] - int(diff/2), arr.shape[0]), axis = 0)

        elif diff < 0:
            diff = (-diff) 
            if round(diff/2) * 2 != diff:
                arr = np.delete(arr, arr.shape[1]-1, axis = 1)
                arr = np.delete(arr, range(0, int((diff/2) - 0.5)), axis = 1)
                arr = np.delete(arr, range(arr.shape[1] - int((diff/2) - 0.5), arr.shape[1]), axis = 1)
            else:
                arr = np.delete(arr, range(0, int(diff/2)), axis = 1)
                arr = np.delete(arr, range(arr.shape[1] - int(diff/2), arr.shape[1]), axis = 1)

        # print ("->", arr.shape)

        resized_arr = cv2.resize(arr, dsize=rsize)

        np.save(dest_dir + "/" + file[1:-4], resized_arr)

if __name__ == '__main__':
    main()