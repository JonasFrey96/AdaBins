from infer import *

import os
import sys 
os.chdir(os.path.join(os.getenv('HOME'), 'ASL'))
sys.path.insert(0, os.path.join(os.getenv('HOME'), 'ASL'))
sys.path.append(os.path.join(os.path.join(os.getenv('HOME'), 'ASL') + '/src'))

import yaml
def file_path(string):
  if os.path.isfile(string):
    return string
  else:
    raise NotADirectoryError(string)

def load_yaml(path):
  with open(path) as file:  
    res = yaml.load(file, Loader=yaml.FullLoader) 
  return res

import coloredlogs
coloredlogs.install()
import time
import argparse

from pathlib import Path
import gc

# Frameworks
import torch
import numpy as np
import imageio
# Costume Modules

from datasets import get_dataset
from torchvision import transforms
from torchvision import transforms as tf


DEVICE = 'cuda:1'

if __name__ == "__main__":
	parser = argparse.ArgumentParser() 
	parser.add_argument('--eval', type=file_path, default="/home/jonfrey/ASL/cfg/eval/eval.yml",
											help='Yaml containing dataloader config')
	args = parser.parse_args()
	env_cfg_path = os.path.join('cfg/env', os.environ['ENV_WORKSTATION_NAME']+ '.yml')
	env_cfg = load_yaml(env_cfg_path)	
	eval_cfg = load_yaml(args.eval)

	# SETUP MODEL
	os.chdir(os.path.join(os.getenv('HOME'), 'AdaBins'))
	inferHelper = InferenceHelper(  dataset='nyu', device='cuda:1' )
	os.chdir(os.path.join(os.getenv('HOME'), 'ASL'))

	# SETUP DATALOADER
	dataset_test = get_dataset(
	**eval_cfg['dataset'],
	env = env_cfg,
	output_trafo = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	)
	dataloader_test = torch.utils.data.DataLoader(dataset_test,
	shuffle = False,
	num_workers = 0,
	pin_memory = eval_cfg['loader']['pin_memory'],
	batch_size = 4, 
	drop_last = True)

	# CREATE RESULT FOLDER
	base = os.path.join(env_cfg['base'], eval_cfg['name'], eval_cfg['dataset']['name'])

	globale_idx_to_image_path = dataset_test.image_pths

	tra = tf.Resize((480,640))
	tra_up = tf.Resize(eval_cfg['dataset']['output_size'])
	
	st = time.time()

	# START EVALUATION
	for j, batch in enumerate( dataloader_test ):
		
		images = batch[0].to(DEVICE)
		target = batch[1].to(DEVICE)
		ori_img = batch[2].to(DEVICE)
		replayed = batch[3].to(DEVICE)
		BS = images.shape[0]
		global_idx = batch[4] 
		
		centers, pred = inferHelper.predict( tra(images) )
		
		# pred = tra_up(torch.from_numpy(pred)).numpy()
		
		for b in range(BS):
				img_path = globale_idx_to_image_path[global_idx[b]]
				p = os.path.join(base,
						img_path.split('/')[-3],
						'depth_estimate',
						img_path.split('/')[-1][:-4]+'.png')
				Path(p).parent.mkdir(parents=True, exist_ok=True)
				store = (pred[b,0,:,:]) * 1000
				store = store.astype(np.uint16)
				imageio.imwrite(p, store)
		print(j, "/" , len(dataloader_test), p)
		print("Estimate Total: ", (time.time()-st)/(1+j)*(len(dataloader_test)),'s' )
		print("Estimate Left: ", (time.time()-st)/(1+j)*(len(dataloader_test)-(1+j) ),'s' )