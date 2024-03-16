import os
import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import time
import os.path
from torch.autograd import Variable
from models import ECNDNet
from utils import *



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="ECNDNet_Test")

parser.add_argument("--num_of_layers", type=int, default=17)
parser.add_argument("--logdir", type=str, default="model")
parser.add_argument("--test_data", type=str, default='input_image')
opt = parser.parse_args()


#直方图均衡化
def equalHist(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    equalized_v = cv2.equalizeHist(v)
    equalized_hsv_image = cv2.merge([h, s, equalized_v])
    equalized_image = cv2.cvtColor(equalized_hsv_image, cv2.COLOR_HSV2BGR)
    return equalized_image

#归一化
def normalize(data):
    return data/255.

'''
这是我们封装的函数
'''
def Denoise_Function(Hist, m_Blur, sharp, kernal):

    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    save_dir =  'Output' + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    net = ECNDNet(channels=1, num_of_layers=opt.num_of_layers)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'model_180.pth')))
    model.eval()
    files_source = glob.glob(os.path.join(opt.test_data, '*.png'))
    files_source.sort()

    for f in files_source:
        Img = cv2.imread(f)
        if m_Blur:
            Img = cv2.medianBlur(Img, kernal)
        if Hist:
            Img = equalHist(Img)
        Img = normalize(np.float32(Img[:,:,0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)
        torch.manual_seed(0)
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=15/255.)
        INoisy = ISource + noise
        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
        with torch.no_grad():
            out1 = model(INoisy)
            out1 = model(out1)
            out1 = model(out1)
            out1 = model(out1)
            out1 = model(out1)
            out1 = out1[0,0,:,:]
            out1 = out1.cpu().numpy()
            if sharp:
                out1 = cv2.filter2D(out1, -1, sharpen_kernel)
        a = os.path.basename(f)
        cv2.imwrite(os.path.join(save_dir, a), out1*255)

if __name__ == "__main__":

    Denoise_Function(Hist=True, m_Blur=True, sharp=False, kernal=3)
