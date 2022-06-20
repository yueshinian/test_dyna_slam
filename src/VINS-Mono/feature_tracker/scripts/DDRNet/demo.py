# -*- coding: UTF-8 -*-
from multiprocessing import parent_process
import os
import argparse
from statistics import mode
import torch
import time
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
import cv2

from torchvision import transforms
from PIL import Image
from segmentation.DDRNet_23_slim import get_seg_model

parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--model', type=str, default='fast_scnn',
                    help='model name (default: fast_scnn)')
parser.add_argument('--dataset', type=str, default='citys',
                    help='dataset name (default: citys)')
parser.add_argument('--weights-folder', default='./weights/best_val_smaller.pth',
                    help='Directory for saving checkpoint models')
parser.add_argument('--input-pic', type=str,
                    default='./datasets/citys/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png',
                    help='path to the input picture')
parser.add_argument('--outdir', default='./test_result', type=str,
                    help='path to save the predict result')

parser.add_argument('--cpu', dest='cpu', action='store_true')
parser.set_defaults(cpu=False)
parser.set_defaults(pretrained=True)

args = parser.parse_args()

cityspallete = [
    128, 64, 128,#地面
    244, 35, 232,
    70, 70, 70,
    102, 102, 156,
    190, 153, 153,
    153, 153, 153,
    250, 170, 30,
    220, 220, 0,
    107, 142, 35,
    152, 251, 152,
    0, 130, 180,
    220, 20, 60,#person
    255, 0, 0,
    0, 0, 142,
    0, 0, 70,
    0, 60, 100,
    0, 80, 100,
    0, 0, 230,
    119, 11, 32,
]

person = [
    0, 0, 0,#地面
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    220, 20, 60,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
]

def demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output folder
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    #weights folder
    if args.pretrained:
        if not os.path.isfile(args.weights_folder):
            return
    # image transform
    transform = transforms.Compose([
            transforms.Resize((512, 1024)),#尺寸hw
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])#将图像转换为tensor向量，同时归一化
    
    #get and load model
    model = get_seg_model(pretrained=False).to(device)
    pretrained_dict = torch.load(args.weights_folder)
    if 'state_dict' in pretrained_dict:
        print("state_dict in pre_dict")
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    print('Finished loading model!')

    #test
    model.eval() #关闭dropout和BN
    input = torch.randn(1, 3, 1024, 2048).cuda()
    with torch.no_grad():#不求导
        outputs = model(input)
    for i in range(5):
        imagePath = './png/test' + str(i) + '.png'
        image = Image.open(imagePath).convert('RGB')#去掉深度通道
        size_t = image.size
        size = [0,1]
        size[0], size[1] = size_t[1], size_t[0]
        #print(size)
        image = transform(image).unsqueeze(0).to(device)#[C H W] to [N C H W] 
        start = time.time()
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        with torch.no_grad():#不求导
            outputs = model(image)
        #outputs =  F.interpolate(outputs[0], size=(1024, 2048), mode='bilinear')
        outputs[0] = F.interpolate(outputs[0], size[-2:],mode='bilinear', align_corners=True)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        end = time.time()
        print("Time used is: %.6f ms"%((end - start)*1000))
        print(type(outputs))
        print(outputs[0].shape)
        
        #tensor to image
        pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()
        out_img = Image.fromarray(pred.astype('uint8'))#numpy to image
        if(i==10):
            weight, height = out_img.size
            pred = pred.astype('int8')
            print(pred.dtype)
            np.savetxt('2.txt',pred)
        out_img.putpalette(person) #着色板
        outname = './test_result/res' + str(i) + '.png'
        out_img.save( outname)
    torch.cuda.empty_cache()

class seg():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((512, 1024)),#尺寸hw
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        #get and load model
        self.model = get_seg_model(pretrained=False).to(self.device)
        pretrained_dict = torch.load(args.weights_folder, map_location= self.device)
        if 'state_dict' in pretrained_dict:
            print("state_dict in pre_dict")
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = self.model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
            if k[6:] in model_dict.keys()}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        self.model.to(self.device)
        print('Finished loading model!')

        #test
        self.model.eval() #关闭dropout和BN
        input = torch.randn(1, 3, 1024, 2048).cuda()
        with torch.no_grad():#不求导
            outputs = self.model(input)
    def getSeg(self, inputs):
        image = inputs.convert('RGB')#去掉深度通道
        size_t = image.size
        size = [0,1]
        size[0], size[1] = size_t[1], size_t[0]
        image = self.transform(image).unsqueeze(0).to(self.device)#[C H W] to [N C H W] 
        with torch.no_grad():#不求导
            outputs = self.model(image)
        outputs[0] = F.interpolate(outputs[0], size[-2:],mode='bilinear', align_corners=True)
        #tensor to image
        pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()
        out_img = Image.fromarray(pred.astype('uint8'))#numpy to image
        out_img.putpalette(person) #着色板
        return np.array(out_img),out_img
    def __del__(self):
        torch.cuda.empty_cache()

if __name__ == '__main__':
    demo()
    imagePath = './png/test0.png'
    image = Image.open(imagePath)
    s = seg()
    npimg,outimg = s.getSeg(image)
    img = cv2.cvtColor( npimg*255,cv2.COLOR_RGB2BGR)
    cv2.imshow("OpenCV",img)
    cv2.waitKey(0)
    input("Press Enter to continue...")
    # cv2.waitKey()
