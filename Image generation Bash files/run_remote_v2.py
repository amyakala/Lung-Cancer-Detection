### loading of files done
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt

import ipyparallel

import sys
input1 = int(sys.argv[1])
input2 = int(sys.argv[2])
clients = ipyparallel.Client()
dview = clients[:]

print('pids: clients created : ',(clients.ids))
print('\nexecution started')
dview['path'] = '/data/scratch/daencs690/code/'

dview.execute('import os')
dview.execute( 'os.chdir(path )')

dview.execute('import parallel_remote_v2')

#current_paths  = dview.map_sync( lambda x: os.getcwd(),range(4))

#print(current_paths)

print(' started parlalisation')
p = sns.color_palette()
# Some constants
#INPUT_FOLDER = '/data/scratch/daencs690/data_set/sample/' # this is sampele
INPUT_FOLDER = '/data/scratch/daencs690/data_set/stage1/' # this is stage1
#INPUT_FOLDER = '/data/scratch/daencs690/data_set/stage2/' # this is stage2 and change in remote

patient =  pd.read_csv('/data/scratch/daencs690/stage1_labels.csv')

patients = os.listdir(INPUT_FOLDER)
patients.sort()
dview.map_sync( lambda x: parallel_remote_v2.fun_call(x) ,patient.id.values[input1:input2])
clients.shutdown(hub=True)
print("all done")
