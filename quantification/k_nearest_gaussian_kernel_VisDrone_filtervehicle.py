#vehicle counting groundtruth

import numpy as np
import scipy
import scipy.io as io
from scipy import spatial
from scipy.ndimage.filters import gaussian_filter
import os
import glob
from matplotlib import pyplot as plt
import h5py
import PIL.Image as Image
from matplotlib import cm as CM
from scipy import misc


#partly borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(img,points):
    '''
    This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.

    points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
    img_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.

    return:
    density: the density-map we want. Same shape as input image but only has one channel.

    example:
    points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
    img_shape: (768,1024) 768 is row and 1024 is column.
    '''
    img_shape=[img.shape[0],img.shape[1]]
    print("Shape of current image: ",img_shape,". Totally need generate ",len(points),"gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(points, k=4)

    print ('generate density...')
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1])<img_shape[0] and int(pt[0])<img_shape[1]:
            pt2d[int(pt[1]),int(pt[0])] = 1.
        else:
            continue

        sigma = 3
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print ('done.')
    return density


# test code
if __name__=="__main__":
    # show an example to use function generate_density_map_with_fixed_kernel.
    root = '../ICCV_workshop/VisDrone/new_separate/'

    train = os.path.join(root,'train','images')
    test = os.path.join(root,'test','images')
    val = os.path.join(root,'val','images')

    path_sets = [train,test,val]
    
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    
    for img_path in img_paths:
        print(img_path)
        a = np.loadtxt(img_path.replace('.jpg','.txt').replace('images','annotations'), delimiter=',')
        img= plt.imread(img_path)#768行*1024列
        k = np.zeros((img.shape[0],img.shape[1]))

        row = a.shape[0]
        try:
            col = a.shape[1]

            kn = 0
            for i in range(row):
                if a[i,5]==4 or a[i,5]==5 or a[i,5]==6 or a[i,5]==9:
                    kn = kn + 1

            j = 0
            b = np.zeros(shape=(kn,2))
            for i in range(row):
                if a[i,5]==4 or a[i,5]==5 or a[i,5]==6 or a[i,5]==9:
                    b[j,0] = a[i,0] + (a[i,2] * 0.5)
                    b[j,1] = a[i,1] + (a[i,3] * 0.5)
                    j = j + 1
        except:

            kn = 0
            for i in range(row):
                if a[5]==4 or a[5]==5 or a[5]==6 or a[5]==9:
                    kn = kn + 1

            j = 0
            b = np.zeros(shape=(kn,2))
            for i in range(row):
                if a[5]==4 or a[5]==5 or a[5]==6 or a[5]==9:
                    b[0] = a[0] + (a[2] * 0.5)
                    b[1] = a[1] + (a[3] * 0.5)
                    j = j + 1

        if kn > 10:
            k = gaussian_filter_density(img,b)
            np.save(img_path.replace('.jpg','.npy').replace('images','filter_GT_vehicle'), k)
            misc.imsave(img_path.replace('images','filter_image_vehicle'), img)
