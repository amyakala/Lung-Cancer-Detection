# patch

import matplotlib
matplotlib.use('Agg')

############




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import dicom
import scipy.misc
import numpy as np

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# you need to change paths at inputs

#df_train = pd.read_csv('/data/scratch/daencs690/stage1_labels.csv') # this is for sample data set
df_train = pd.read_csv('/data/scratch/daencs690/stage1_labels.csv') # this is the train it should be good for entire data set


#p = sns.color_palette()
# Some constants
#INPUT_FOLDER = '/data/scratch/daencs690/data_set/sample/' # this is for sample data
INPUT_FOLDER = '/data/scratch/daencs690/data_set/stage1/' # this is stage 1 and chance at line 172
#INPUT_FOLDER = '/data/scratch/daencs690/data_set/stage2/' # this is stage 2


#patients = os.listdir(INPUT_FOLDER)
#patients.sort()


def read_ct_scan(folder_name):
        # Read the slices from the dicom file
        slices = [dicom.read_file(folder_name +'/'+ filename) for filename in os.listdir(folder_name)]

        # Sort the dicom slices in their respective order
        slices.sort(key=lambda x: int(x.InstanceNumber))

        # Get the pixel values for all the slices
        slices = np.stack([s.pixel_array for s in slices])
        slices[slices == -2000] = 0
        return slices


def get_segmented_lungs(im, plot=False):

    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image.
    '''
    binary = im < 604
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone)

    return im

def segment_lung_from_ct_scan(ct_scan):
    return np.asarray([get_segmented_lungs(slice) for slice in ct_scan])



# 3d function needed to be added here

def plot_3d(image,patient ,threshold=-300):

    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]

    verts, faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.axis('off')

    # clasification setion #########################
    #save_path ='/data/scratch/daencs690/data_set/image_set_v2/' #this is for sample
    save_path ='/data/scratch/daencs690/data_set/train_stage1_v2/' # this is stage1
    #save_path ='/data/scratch/daencs690/data_set/train_stage2/' #this is stage 2
    if df_train.ix[df_train.id== patient].values[0,1] == 1 :
        subpath = 'cancer/'
        total_path= save_path+ subpath
    else:
        subpath = 'non-cancer/'
        total_path= save_path+subpath


    if not os.path.exists(total_path):
        os.makedirs(total_path)
    plt.savefig(total_path+patient)
    return 0


    ###############################################




def fun_call(patient):
    import gc
    ct_scan = read_ct_scan(INPUT_FOLDER + patient)
    segmented_ct_scan = segment_lung_from_ct_scan(ct_scan)
    plot_3d(segmented_ct_scan,patient ,604)
    del segmented_ct_scan
    gc.collect()
    return 0


