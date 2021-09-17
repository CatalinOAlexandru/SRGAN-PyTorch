# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Dataset paths:
# /home/calexand/datasets/Histology/test --- 27 image
# /home/calexand/datasets/histo_split_4/cropTarget -- 120 images

print('Starting Testing...')

import logging
import os
import sys
from argparse import ArgumentParser
import matplotlib.pyplot as plt 
import pandas as pd
import cv2
import time
import numpy as np

import torch
from PIL import Image
from skimage.color import rgb2ycbcr
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

# from srgan_pytorch.model import Generator
from srgan_pytorch.utils import create_folder

import lpips

import warnings
warnings.simplefilter("ignore", UserWarning)

# It is a convenient method for simple scripts to configure the log package at one time.
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

parserTest = ArgumentParser()
parserTest.add_argument("--pretrained", dest="pretrained", action="store_true", help="Use pre-trained model.")
parserTest.add_argument("--model-path", default=None, type=str, help="Path to latest checkpoint for model.")
parserTest.add_argument("--model-type", default="model2", type=str, help="Which model file to use. Default: model2")
# parserTest.add_argument("--test-path-lr", default="data/Set14/LRbicx4", type=str, help="Path to test images")
parserTest.add_argument("--test-path-hr", default="/home/calexand/datasets/histo_split_4/cropTarget", type=str, help="Path to test images")
parserTest.add_argument("--scale-factor", default=4, type=int, help="Scale Factor for image")
parserTest.add_argument("--max-images", default=None, type=int, help="Only run testing on N amount of images instead of the entire test folder.")
parserTest.add_argument("--num-resBlocks", default=16, type=int, help="Number of Residual Blocks blocks used in the model. Default: 16 as in the paper.")
parserTest.add_argument("--name", default="Default", type=str, help="Name for test folder")
parserTest.add_argument("--cuda", dest="cuda", action="store_true", help="Enables cuda.")
args = parserTest.parse_args()

if args.model_type == 'model2' or args.model_type == 'model3' or args.model_type == 'model32':
    from srgan_pytorch.model2 import Generator
elif args.model_type == 'model1':
    from srgan_pytorch.model1 import Generator
else:
    print('Model selected is not available. Use "model1", "model2", "model3" or "model32".')
    sys.exit()

# Set whether to use CUDA.
device = torch.device("cuda:0" if args.cuda else "cpu")
print('Using device:',device)


def sr8(model, hr_filename, sr_filename):
    with torch.no_grad():

        img = cv2.imread(hr_filename)

        downDim = (int(img.shape[0]/8),int(img.shape[1]/8)) 
        downImg = cv2.resize(img,downDim,interpolation = cv2.INTER_CUBIC)
        downImg = cv2.cvtColor(downImg, cv2.COLOR_BGR2RGB)

        downImgTensor = ToTensor()(downImg).unsqueeze(0).to(device)

        upImg = model(downImgTensor)

        save_image(upImg.detach(), sr_filename, normalize=True)


def sr2(model, hr_filename, sr_filename):
    with torch.no_grad():

        img = cv2.imread(hr_filename)

        downDim = (int(img.shape[0]/8),int(img.shape[1]/8)) 
        downImg = cv2.resize(img,downDim,interpolation = cv2.INTER_CUBIC)
        downImg = cv2.cvtColor(downImg, cv2.COLOR_BGR2RGB)

        downImgTensor = ToTensor()(downImg).unsqueeze(0).to(device)

        upImg1 = model(downImgTensor)
        upImg2 = model(upImg1)
        upImg3 = model(upImg2)
        save_image(upImg3.detach(), sr_filename, normalize=True)


def sr42(model2, model4, hr_filename, sr_filename):
    with torch.no_grad():

        img = cv2.imread(hr_filename)

        downDim = (int(img.shape[0]/8),int(img.shape[1]/8)) 
        downImg = cv2.resize(img,downDim,interpolation = cv2.INTER_CUBIC)
        downImg = cv2.cvtColor(downImg, cv2.COLOR_BGR2RGB)

        downImgTensor = ToTensor()(downImg).unsqueeze(0).to(device)

        upImg1 = model4(downImgTensor)
        upImg2 = model2(upImg1)
        save_image(upImg2.detach(), sr_filename, normalize=True)


def sr24(model2, model4, hr_filename, sr_filename):
    with torch.no_grad():

        img = cv2.imread(hr_filename)

        downDim = (int(img.shape[0]/8),int(img.shape[1]/8)) 
        downImg = cv2.resize(img,downDim,interpolation = cv2.INTER_CUBIC)
        downImg = cv2.cvtColor(downImg, cv2.COLOR_BGR2RGB)

        downImgTensor = ToTensor()(downImg).unsqueeze(0).to(device)

        upImg1 = model2(downImgTensor)
        upImg2 = model4(upImg1)
        save_image(upImg2.detach(), sr_filename, normalize=True)


# normalize image between xmin and xmax
def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom


def perpSim(sr_filename, hr_filename,loss_fn_alex):

    sr_image = lpips.load_image(sr_filename) # RGB image from [-1,1]
    hr_image = lpips.load_image(hr_filename)

    srSize = sr_image.shape[0]
    hrSize = hr_image.shape[0]
    if(srSize > hrSize):
        diff = int((srSize - hrSize)/2)
        sr_image = sr_image[diff:-diff, diff:-diff, ...]
    elif(hrSize > srSize):
        diff = int((hrSize - srSize)/2)
        hr_image = hr_image[diff:-diff, diff:-diff, ...]
    else:
        pass

    sr_image = lpips.im2tensor(sr_image) 
    hr_image = lpips.im2tensor(hr_image)

    d1 = loss_fn_alex(sr_image, hr_image)

    return d1.item()


def iqa(sr_filename, hr_filename):
    r""" Image quality evaluation function.

    Args:
        sr_filename (str): Super resolution image address.
        hr_filename (srt): High resolution image address.

    Returns:
        PSNR value(float), SSIM value(float).
    """
    sr_image = imread(sr_filename)
    hr_image = imread(hr_filename)

    srSize = sr_image.shape[0]
    hrSize = hr_image.shape[0]

    # Delete 4 pixels around the image to facilitate PSNR calculation.
    if(srSize == hrSize):
        sr_image = sr_image[4:-4, 4:-4, ...]
        hr_image = hr_image[4:-4, 4:-4, ...]
    elif(srSize > hrSize):
        diff = int((srSize - hrSize)/2)
        sr_image = sr_image[diff:-diff, diff:-diff, ...]

        sr_image = sr_image[4:-4, 4:-4, ...]
        hr_image = hr_image[4:-4, 4:-4, ...]
    else:
        diff = int((hrSize - srSize)/2) 
        hr_image = hr_image[diff:-diff, diff:-diff, ...]

        sr_image = sr_image[4:-4, 4:-4, ...]
        hr_image = hr_image[4:-4, 4:-4, ...]

    # Calculate the Y channel of the image. Use the Y channel to calculate PSNR
    # and SSIM instead of using RGB three channels.
    sr_image = sr_image / 255.0
    hr_image = hr_image / 255.0
    sr_image = rgb2ycbcr(sr_image)[:, :, 0:1]
    hr_image = rgb2ycbcr(hr_image)[:, :, 0:1]
    # Because rgb2ycbcr() outputs a floating point type and the range is [0, 255],
    # it needs to be renormalized to [0, 1].
    sr_image = sr_image / 255.0
    hr_image = hr_image / 255.0

    psnr = peak_signal_noise_ratio(sr_image, hr_image)
    ssim = structural_similarity(sr_image,
                                 hr_image,
                                 win_size=11,
                                 gaussian_weights=True,
                                 multichannel=True,
                                 data_range=1.0,
                                 K1=0.01,
                                 K2=0.03,
                                 sigma=1.5)

    return psnr, ssim


def saveZoomImages(sr_filename,hr_filename,filename,psnr,ssim,d1):
    # lr_image = imread(lr_filename)
    # sr_image = imread(sr_filename)
    # hr_image = imread(hr_filename)

    sr_image = cv2.cvtColor(cv2.imread(sr_filename), cv2.COLOR_BGR2RGB)
    hr_image = cv2.cvtColor(cv2.imread(hr_filename), cv2.COLOR_BGR2RGB)
    dims = (int(hr_image.shape[0]/args.scale_factor),int(hr_image.shape[1]/args.scale_factor))
    lr_image = cv2.resize(hr_image,dims,interpolation = cv2.INTER_CUBIC)

    cropSize = 50 # 50 in each direction, therefore 100x100 zoom image
    sf = args.scale_factor # scale factor

    fig, axs = plt.subplots(2, 3)
    name = 'PSNR: ' + str(round(psnr,2)) + \
           ' | SSIM: ' + str(round(ssim,2)) + \
           ' | PerceptSim: '+str(round(d1,2))
    fig.suptitle(name)
    lrsize = int(lr_image.shape[0]/2)
    srsize = int(sr_image.shape[0]/2)

    axs[0,0].imshow(lr_image[lrsize-cropSize:lrsize+cropSize, lrsize-cropSize:lrsize+cropSize, :])
    axs[0,0].set_title('Low Res')
    axs[0,0].axis('off')
    axs[0,1].imshow(sr_image[srsize-(cropSize*sf):srsize+(cropSize*sf), srsize-(cropSize*sf):srsize+(cropSize*sf), :])
    axs[0,1].set_title('Super Res')
    axs[0,1].axis('off')
    axs[0,2].imshow(hr_image[srsize-(cropSize*sf):srsize+(cropSize*sf), srsize-(cropSize*sf):srsize+(cropSize*sf), :])
    axs[0,2].set_title('Ground Truth')
    axs[0,2].axis('off')

    axs[1,0].imshow(lr_image)
    axs[1,0].axis('off')
    axs[1,1].imshow(sr_image)
    axs[1,1].axis('off')
    axs[1,2].imshow(hr_image)
    axs[1,2].axis('off')
    plt.tight_layout()
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    namePath = os.path.join("tests", args.name, 'figs',filename)
    fig.savefig(namePath,dpi=300)
    plt.close()



def main():
    # Load model and weights.

    model2 = Generator(2, 16).to(device).eval()
    model4 = Generator(4, 16).to(device).eval()
    model8 = Generator(8, 16).to(device).eval()

    model2.load_state_dict(torch.load('/home/calexand/calexand/SRGAN-PyTorch/weights/BreTest1_x2_128/G-last.pth'))
    model4.load_state_dict(torch.load('/home/calexand/calexand/SRGAN-PyTorch/weights/BreTest2_x4_128/G-last.pth'))
    model8.load_state_dict(torch.load('/home/calexand/calexand/SRGAN-PyTorch/weights/BreTest3_x8_128/G-last.pth'))


    # loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores

    # Get test image file index.
    filenames = os.listdir('/home/calexand/datasets/histo_split_4/cropTarget')

    # for statistics
    # resultScores = {'psnr': [], 'ssim': [], 'perpSimi': []}
    print('HR images are 2200 and will get downscaled by a factor of 8 to 275.')

    for index in range(len(filenames)):
        if (args.max_images != None) and args.max_images == index:
            break

        logger.info(f"Image: {index+1} / {len(filenames)}")

        sr_filename = os.path.join("tests", args.name, 'full_image/x8_only.png')
        hr_filename = os.path.join('/home/calexand/datasets/histo_split_4/cropTarget', filenames[index])

        sr8(model8, hr_filename, sr_filename)

        psnr, ssim = iqa(sr_filename, hr_filename)

        print('LR > x8 > SR | PSNR:',round(psnr,2))



        sr_filename = os.path.join("tests", args.name, 'full_image/x2_3times.png')

        sr2(model2, hr_filename, sr_filename)

        psnr, ssim = iqa(sr_filename, hr_filename)

        print('LR > x2 > x2 > x2 > SR | PSNR:',round(psnr,2))



        sr_filename = os.path.join("tests", args.name, 'full_image/x2_x4.png')

        sr24(model2,model4, hr_filename, sr_filename)

        psnr, ssim = iqa(sr_filename, hr_filename)

        print('LR > x2 > x4 > SR | PSNR:',round(psnr,2))




        sr_filename = os.path.join("tests", args.name, 'full_image/x4_x2.png')

        sr42(model2,model4, hr_filename, sr_filename)

        psnr, ssim = iqa(sr_filename, hr_filename)

        print('LR > x4 > x2 > SR | PSNR:',round(psnr,2))



        # resultScores['psnr'].append(psnr)
        # resultScores['ssim'].append(ssim)
        # resultScores['perpSimi'].append(perpSimi)

        if index == 4:
            break
            # saveZoomImages(sr_filename,hr_filename,filenames[index],psnr,ssim,perpSimi)


    # Calculate the average index value of the image quality of the test dataset.
    # avg_psnr = sum(resultScores['psnr']) / len(resultScores['psnr'])
    # avg_ssim = sum(resultScores['ssim']) / len(resultScores['ssim'])
    # avg_percepSim = sum(resultScores['perpSimi']) / len(resultScores['perpSimi'])

    # logger.info(f"Mean Average PSNR: {str(round(avg_psnr,2))}")
    # logger.info(f"Mean Average SSIM: {str(round(avg_ssim,4))}")
    # logger.info(f"Mean Average Perceptual Similarity: {str(round(avg_percepSim,4))}")
    # logger.info(f"PSNR Best / Worst: {str(round(np.max(resultScores['psnr']),2))} / {str(round(np.min(resultScores['psnr']),2))}")
    # logger.info(f"SSIM Best / Worst: {str(round(np.max(resultScores['ssim']),2))} / {str(round(np.min(resultScores['ssim']),2))}")
    # logger.info(f"PercepSim Best/Worst: {str(round(np.min(resultScores['perpSimi']),3))} / {str(round(np.max(resultScores['perpSimi']),3))}")

    # out_path = 'stats/testing/'
    # data_frame_score = pd.DataFrame(
    #     data={'PSNR Score': resultScores['psnr'], 'SSIM Score': resultScores['ssim'], 'Perceptual Similarity Score': resultScores['perpSimi']},
    #     index=filenames[:len(resultScores['psnr'])])
    # data_frame_score.to_csv(out_path + str(args.name) + '_Testing_Scores.csv', index_label='File Name')


if __name__ == "__main__":
    create_folder("tests")
    create_folder(os.path.join("tests", args.name))
    create_folder(os.path.join("tests", args.name, "figs"))
    create_folder(os.path.join("tests", args.name, "full_image"))
    create_folder("stats")
    create_folder("stats/testing")

    logger.info("TrainEngine:")
    logger.info("\tAPI version .......... 0.4.1")
    logger.info("\tBuild ................ 2021.07.09")
    logger.info("\tModified by ... Catalin Alexandru")
    logger.info("\tOn ................... 2021.08.08")

    main()

    logger.info("All testing has been completed successfully.\n")
