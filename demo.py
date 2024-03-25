from model import DeepEraser
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import warnings
import argparse

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="arguments for mp3 to video generation")

    parser.add_argument(
        "--rec_model_path", type=str, default='./deeperaser.pth',
        help="The path to necessary version of deeperaser.pth"
    )
    
    parser.add_argument(
        "--img_path", type=str, default='./input_imgs/input.jpg',
        help="Path to image for text removal"
    )

    parser.add_argument(
        "--mask_path", type=str, default='./input_imgs/mask.jpg',
        help="Path to precomputed mask image used to specify text removal locations"
    )

    parser.add_argument(
        "--save_path", type=str, default='./output_imgs/output.jpg',
        help="Path to save de-texted image"
    )

    user_args = parser.parse_args()
    return user_args


def reload_rec_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:0')
        # print(len(pretrained_dict.keys()))
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        # print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model


def rec(rec_model_path, img_path, mask_path, save_path):

    net = DeepEraser().cuda()
    reload_rec_model(net, rec_model_path)
    net.eval()

    img = np.array(Image.open(img_path))[:, :, :3]   
    mask = np.array(Image.open(mask_path))[:, :]

    im = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()
    mask = torch.from_numpy(mask / 255.0).unsqueeze(0).float()
    
    with torch.no_grad():
    
        pred_img = net(im.unsqueeze(0).cuda(), mask.unsqueeze(0).cuda())
        pred_img[-1] = torch.clamp(pred_img[-1], 0, 1)
        out = (pred_img[-1][0]*255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        cv2.imwrite(save_path, out[:,:,::-1])
            

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def main():
    user_args = parse_args()
    rec(user_args.rec_model_path, user_args.img_path, user_args.mask_path, user_args.save_path)


if __name__ == "__main__":
    main()
