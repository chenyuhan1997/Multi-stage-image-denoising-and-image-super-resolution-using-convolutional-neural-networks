import os
import os.path
from torch.autograd import Variable
from models import ECNDNet
from utils import *
import utils_image as util
from RFDN import RFDN
import cv2
import argparse
import glob


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
def Denoise_SuperR(Hist, m_Blur, sharp, kernal, superR):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    save_dir =  'Output' + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    net1 = ECNDNet(channels=1, num_of_layers=opt.num_of_layers)
    model_path = os.path.join(opt.logdir, 'RFDN_AIM.pth')
    model2 = RFDN()
    device_ids = [0]
    model1 = nn.DataParallel(net1, device_ids=device_ids).cuda()
    model1.load_state_dict(torch.load(os.path.join(opt.logdir, 'model_180.pth')))
    model2.load_state_dict(torch.load(model_path), strict=True)
    model1.eval()
    model2.eval()


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
            out1 = model1(INoisy)
            out1 = model1(out1)
            out1 = model1(out1)
            out1 = model1(out1)
            out1 = model1(out1)
            out1 = out1[0,0,:,:]
            out1 = (out1*255).cpu().numpy()
            if superR:
                img = cv2.cvtColor(out1, cv2.COLOR_GRAY2RGB)
                img_L = util.uint2tensor4(img)
                img_E = model2(img_L)
                torch.cuda.synchronize()
                img_E = util.tensor2uint(img_E)

                if sharp:
                    img_E = cv2.filter2D(img_E, -1, sharpen_kernel)

        a = os.path.basename(f)
        cv2.imwrite(os.path.join(save_dir, a), cv2.cvtColor(img_E, cv2.COLOR_BGR2GRAY))

if __name__ == "__main__":

    Denoise_SuperR(Hist=True, m_Blur=True, sharp=True, kernal=3, superR=True)