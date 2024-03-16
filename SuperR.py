import os.path
import torch
import numpy as np
import utils_image as util
from RFDN import RFDN
import cv2
import argparse
import glob


parser = argparse.ArgumentParser(description="RFDN_Test")
parser.add_argument("--logdir", type=str, default="model")
parser.add_argument("--test_data", type=str, default='input_image')
opt = parser.parse_args()



def superR(sharp):

    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    save_dir = 'Output' + '/'
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    model_path = os.path.join(opt.logdir, 'RFDN_AIM.pth')
    model = RFDN()
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    L_folder = glob.glob(os.path.join(opt.test_data, '*.png'))
    L_folder.sort()

    for img in L_folder:
        img_L = util.imread_uint(img, n_channels=3)
        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(device)
        img_E = model(img_L)
        torch.cuda.synchronize()
        img_E = util.tensor2uint(img_E)
        if sharp:
            img_E = cv2.filter2D(img_E, -1, sharpen_kernel)
        a = os.path.basename(img)
        cv2.imwrite(os.path.join(save_dir, a), img_E)

