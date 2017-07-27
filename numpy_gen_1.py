import os
import numpy as np
import pandas as pd
import ipyparallel


clients = ipyparallel.Client()
dview = clients[:]

print('pids: clients created : ',(clients.ids))
print('\nexecution started')

dview['path'] = '/home/amyakala/pp/'

dview.execute('import os')
dview.execute( 'os.chdir(path )')

# we have now set the path for all the parallel process
current_paths  = dview.map_sync( lambda x: os.getcwd(),range(4))
print('\npath of clinets \n',current_paths)

dview.execute('import numpy_gen_2')

print(' started parlalisation')


INPUT_FOLDER = '/data/scratch/daencs690/data_set/stage1/' # this is stage1

patients =  pd.read_csv('/data/scratch/daencs690/anirudh/stage1_labels.csv')
much_data = dview.map_sync( lambda x: vamsi_f2.fun_call(x) ,patients.id.values)

# one need to mention in this file and vamsi_f2 and both should be same and mention them as global variables.
IMG_PX_SIZE = 512
HM_SLICES = 20

np.save('muchdata-{}-{}-{}.npy'.format(IMG_PX_SIZE,IMG_PX_SIZE,HM_SLICES), much_data)

clients.shutdown(hub=True)

print('all done')





