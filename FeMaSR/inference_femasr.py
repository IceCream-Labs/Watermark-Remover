import argparse
import cv2
import glob
import os
from tqdm import tqdm
import torch
from yaml import load

from basicsr.utils import img2tensor, tensor2img, imwrite 
from basicsr.archs.femasr_arch import FeMaSRNet 
from basicsr.utils.download_util import load_file_from_url 

pretrain_model_url = {
    'x4': 'https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX4_model_g.pth',
    'x2': 'https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX2_model_g.pth',
}


def sr(img,out_scale=4,max_size=600):
    """Inference demo for FeMaSR 
    """
    '''parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument('-w', '--weight', type=str, default=None, help='path for model weights')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-s', '--out_scale', type=int, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('--max_size', type=int, default=600, help='Max image size for whole image inference, otherwise use  tiled_test')'''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    #if args.weight is None:
    weight_path = load_file_from_url(pretrain_model_url[f'x{out_scale}'])
   # else:
      #  weight_path = args.w
    
    # set up the model
    sr_model = FeMaSRNet(codebook_params=[[32, 1024, 512]], LQ_stage=True, scale_factor=out_scale).to(device)
    sr_model.load_state_dict(torch.load(weight_path)['params'], strict=False)
    sr_model.eval()
    img_tensor = img2tensor(img).to(device) / 255.
    img_tensor = img_tensor.unsqueeze(0)

    max_size = max_size ** 2 
    h, w = img_tensor.shape[2:]
    if h * w < max_size: 
        output = sr_model.test(img_tensor)
    else:
        output = sr_model.test_tile(img_tensor)
    output_img = tensor2img(output)

    #save_path = os.path.join(args.output, f'{img_name}')
    #imwrite(output_img, save_path)
    #pbar.update(1)
    #pbar.close()
    return output_img


if __name__ == '__main__':
    sr()
