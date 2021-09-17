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
parserTest.add_argument("--name", default="DEF", type=str, help="Name for test folder")
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


def sr(model, hr_filename, sr_filename):
    r""" Turn low resolution into super resolution.

    Args:
        model (torch.nn.Module): SR model.
        lr_filename (str): Low resolution image address.
        sr_filename (srt): Super resolution image address.
    """
    with torch.no_grad():
        start = time.time()

        img = cv2.imread(hr_filename)

        downDim = (int(img.shape[0]/args.scale_factor),int(img.shape[1]/args.scale_factor)) 
        downImg = cv2.resize(img,downDim,interpolation = cv2.INTER_CUBIC)
        downImg = cv2.cvtColor(downImg, cv2.COLOR_BGR2RGB)

        downImgTensor = ToTensor()(downImg).unsqueeze(0).to(device)

        upImg = model(downImgTensor)

        # Old testing code
        # lr = Image.open(lr_filename).convert("RGB")
        # lr_tensor = ToTensor()(lr).unsqueeze(0).to(device)
        # upImg = model(lr_tensor)

        end = time.time()
        logger.info(f"It took {round(end - start,2)} seconds to process 1 SR image.")
        save_image(upImg.detach(), sr_filename, normalize=True)


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
    if args.model_type == 'model1':
        model = Generator(args.scale_factor).to(device).eval()
    else:
        model = Generator(args.scale_factor, args.num_resBlocks).to(device).eval()

    if args.model_path != None:
        logger.info(f"Loading weights from `{args.model_path}`.")
        model.load_state_dict(torch.load(args.model_path))
    else:
        crateModelPath = os.path.join("weights", args.name, 'G-last.pth')
        logger.info(f"Loading weights from `{crateModelPath}`.")
        model.load_state_dict(torch.load(crateModelPath))

    loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores

    # Get test image file index.
    filenames = os.listdir(os.path.join(args.test_path_hr))

    # for statistics
    resultScores = {'psnr': [], 'ssim': [], 'perpSimi': []}


    for index in range(len(filenames)):
        if (args.max_images != None) and args.max_images == index:
            break

        # print("Image {}/{}".format(index+1,len(filenames)))
        logger.info(f"Image: {index+1} / {len(filenames)}")

        # lr_filename = os.path.join(args.test_path_lr, filenames[index])
        sr_filename = os.path.join("tests", args.name, 'full_image', filenames[index])
        hr_filename = os.path.join(args.test_path_hr, filenames[index])

        # Process low-resolution images into super-resolution images.
        sr(model, hr_filename, sr_filename)

        # Test the image quality difference between the super-resolution image
        # and the original high-resolution image.
        psnr, ssim = iqa(sr_filename, hr_filename)
        perpSimi = perpSim(sr_filename, hr_filename,loss_fn_alex)

        resultScores['psnr'].append(psnr)
        resultScores['ssim'].append(ssim)
        resultScores['perpSimi'].append(perpSimi)

        if index < 5:
            saveZoomImages(sr_filename,hr_filename,filenames[index],psnr,ssim,perpSimi)


    # Calculate the average index value of the image quality of the test dataset.
    avg_psnr = sum(resultScores['psnr']) / len(resultScores['psnr'])
    avg_ssim = sum(resultScores['ssim']) / len(resultScores['ssim'])
    avg_percepSim = sum(resultScores['perpSimi']) / len(resultScores['perpSimi'])

    logger.info(f"Mean Average PSNR: {str(round(avg_psnr,2))}")
    logger.info(f"Mean Average SSIM: {str(round(avg_ssim,4))}")
    logger.info(f"Mean Average Perceptual Similarity: {str(round(avg_percepSim,4))}")
    logger.info(f"PSNR Best / Worst: {str(round(np.max(resultScores['psnr']),2))} / {str(round(np.min(resultScores['psnr']),2))}")
    logger.info(f"SSIM Best / Worst: {str(round(np.max(resultScores['ssim']),2))} / {str(round(np.min(resultScores['ssim']),2))}")
    logger.info(f"PercepSim Best/Worst: {str(round(np.min(resultScores['perpSimi']),3))} / {str(round(np.max(resultScores['perpSimi']),3))}")

    out_path = 'stats/testing/'
    data_frame_score = pd.DataFrame(
        data={'PSNR Score': resultScores['psnr'], 'SSIM Score': resultScores['ssim'], 'Perceptual Similarity Score': resultScores['perpSimi']},
        index=filenames[:len(resultScores['psnr'])])
    data_frame_score.to_csv(out_path + str(args.name) + '_Testing_Scores.csv', index_label='File Name')


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
