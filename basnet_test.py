import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
import cv2

from model import BASNet

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

def save_output(image_name,pred,d_dir):

	predict = pred
	predict = predict.squeeze()
	predict_np = predict.cpu().data.numpy()

	im = Image.fromarray(predict_np*255).convert('RGB')
	img_name = image_name.split("/")[-1]
	image = io.imread(image_name)
	imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

	pb_np = np.array(imo)

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]
	#im.save(d_dir+imidx+'2.png')
	imo_opencv=np.array(imo)
	imo_opencv=cv2.cvtColor(imo_opencv, cv2.COLOR_BGR2GRAY)
	img_real=cv2.imread(image_name)
	img_real=cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB)
	imo_opencv[imo_opencv!=0]=1
	white_mask=(imo_opencv==0).astype(np.uint8)
	white_mask[white_mask==1]=255
	img=np.array([imo_opencv.T,imo_opencv.T,imo_opencv.T]).T
	white_mask=np.array([white_mask.T,white_mask.T,white_mask.T]).T
	img_n=np.multiply(img_real,img)
	img_f=img_n+white_mask
	img_f=cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB)
	cv2.imwrite('{}{}.png'.format(d_dir,imidx),img_f)

    
    

	#imo.save('{}{}.png'.format(d_dir,imidx))
    


#if __name__ == '__main__':
def segment(test_dir,output_dir,model_weights):
    

	# --------- 1. get image path and name ---------
	
	#image_dir = '/home/juno/SLBR_test_images/rst/only_image/'S
	#prediction_dir = '/home/juno/extrtacted_objects/Ex_af_wr_bf_sr/'
	#model_dir = model_weights
    
	
	img_name_list = glob.glob(test_dir + '*.jpg')
	#print(img_name_list)
	

	test_salobj_dataset = SalObjDataset(img_name_list = img_name_list, lbl_name_list = [],transform=transforms.Compose([RescaleT(256),ToTensorLab(flag=0)]))
	test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1,shuffle=False,num_workers=1)

	# --------- 3. model define ---------
	print("...load BASNet...")
	net = BASNet(3,1)
	state_dict=torch.hub.load_state_dict_from_url(model_weights,map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
	net.load_state_dict(state_dict)
	if torch.cuda.is_available():
		net.cuda()
	net.eval()
	#i=0
    
	
	# --------- 4. inference for each image ---------
	for i_test, data_test in enumerate(test_salobj_dataloader):
	
		#print("inferencing:",img_name_list[i_test].split("/")[-1])
	
		inputs_test = data_test['image']
		inputs_test = inputs_test.type(torch.FloatTensor)
	
		if torch.cuda.is_available():
			inputs_test = Variable(inputs_test.cuda())
		else:
			inputs_test = Variable(inputs_test)
	
		d1,d2,d3,d4,d5,d6,d7,d8 = net(inputs_test)
	
		# normalization
		pred = d1[:,0,:,:]
		pred = normPRED(pred)
	
		# save results to test_results folder
		save_output(img_name_list[i_test],pred,output_dir)
		#i+=1
	
		del d1,d2,d3,d4,d5,d6,d7,d8
