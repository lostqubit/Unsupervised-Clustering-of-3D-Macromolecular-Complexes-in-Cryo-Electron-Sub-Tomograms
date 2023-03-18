import numpy as np
import pandas as pd
import mrcfile as mrc
import warnings
import os

np.random.seed(0)

warnings.simplefilter('ignore')

base_dir = os.path.join(os.getcwd(),'data')

train_dir = os.path.join(base_dir,'train')
test_dir = os.path.join(base_dir,'test')

test_data = {'filename':[],'pdb_id':[]} 
train_data = {'filename':[],'pdb_id':[]}

locations = []
with open('shrec19_cryoet_full_dataset/full_dataset/9/particle_locations_model_9.txt', newline="\n") as f:
	for line in f:
		pdb_id, Z, Y, X, rot_X, rot_Y, rot_Z = line.rstrip('\n').split()
		locations.append((pdb_id, int(Z), int(Y), int(X)))
	f.close()
    
with mrc.open('shrec19_cryoet_full_dataset/full_dataset/9/reconstruction_model_9.mrc', permissive=True) as f:
	exp_data = f.data
	f.close()
 
size = 24
count = 0
np.random.shuffle(locations)
print('[INFO] Building Test dataset ...')
for pdb_id, Z, Y, X in locations:
	
	subtomogram = exp_data[Z+156-int(size/2):Z+156+int(size/2), Y-int(size/2):Y+int(size/2), X-int(size/2):X+int(size/2)]
	if (subtomogram.shape != (size, size, size)):
		continue
	count = count + 1

	test_data['filename'].append(r'{}.mrc'.format(count))
	test_data['pdb_id'].append(pdb_id)

	if not os.path.exists(test_dir):
		os.makedirs(test_dir)
		with mrc.new(os.path.join(test_dir,'1.mrc')) as f:
			f.set_data(subtomogram)
			f.close()

	else:
		with mrc.new(os.path.join(test_dir,r'{}.mrc'.format(count))) as f:
			f.set_data(subtomogram)
			f.close()

test_labels = pd.DataFrame(test_data)
test_encoded = pd.get_dummies(test_labels,columns=['pdb_id'])
test_encoded.to_csv(os.path.join(base_dir,'test_labels.csv'),index=False) 


size = 24
count = 0
print('[INFO] Building Train dataset ...')
for i in range(9):
	locations = []

	with open(r'shrec19_cryoet_full_dataset/full_dataset/{}/particle_locations_model_{}.txt'.format(i,i), newline="\n") as f:
		for line in f:
			pdb_id, Z, Y, X, rot_X, rot_Y, rot_Z = line.rstrip('\n').split()
			locations.append((pdb_id, int(Z), int(Y), int(X)))
		f.close()

	with mrc.open(r'shrec19_cryoet_full_dataset/full_dataset/{}/reconstruction_model_{}.mrc'.format(i,i), permissive=True) as f:
		exp_data = f.data
	f.close()

	np.random.shuffle(locations)
	for pdb_id, Z, Y, X in locations:

		subtomogram = exp_data[Z+156-int(size/2):Z+156+int(size/2), Y-int(size/2):Y+int(size/2), X-int(size/2):X+int(size/2)]
		if (subtomogram.shape != (size, size, size)):
			continue

		count = count + 1

		train_data['filename'].append(r'{}.mrc'.format(count))
		train_data['pdb_id'].append(pdb_id)
	
		if not os.path.exists(train_dir):
			os.makedirs(train_dir)
			with mrc.new(os.path.join(train_dir,'1.mrc')) as f:
				f.set_data(subtomogram)
				f.close()

		else:
			with mrc.new(os.path.join(train_dir,r'{}.mrc'.format(count))) as f:
				f.set_data(subtomogram)
				f.close()


train_labels = pd.DataFrame(train_data)
train_encoded = pd.get_dummies(train_labels,columns=['pdb_id'])
train_encoded.to_csv(os.path.join(base_dir,'train_labels.csv'), index = False)