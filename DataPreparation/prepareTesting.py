import cv2
import os
import argparse
import shutil

parser = argparse.ArgumentParser(description='Crop images into 2x2')
parser.add_argument('--path', default='histo_split_4/test', type=str, help='path to folder')
parser.add_argument('--save_path', default='histo_split_4/testProcessed', type=str, help='path to folder')
parser.add_argument('--scale_factor', default=8, type=int, help='scale factor to downscale')

if __name__ == '__main__':
    opt = parser.parse_args()
    pathDataset = opt.path
    pathSave = opt.save_path
    sf = opt.scale_factor

    # loading images from folder
    files = os.listdir(pathDataset)
    if '.DS_Store' in files:
        files.remove('.DS_Store')

    index = 1
    for file in files:
        print('{} Downsampling: {}'.format(index,file))
        index += 1

        pathImage = str(pathDataset + '/'+ file)
        img = cv2.imread(pathImage)
        # print('input image shape:',img.shape)
        w,h = img.shape[:2]

        dim = (round(w/sf),round(h/sf)) 
        downImg = cv2.resize(img,dim,sf,sf,interpolation = cv2.INTER_CUBIC)
        # print('output image shape:',downImg.shape)
        
        if not os.path.exists(pathSave):
            os.makedirs(pathSave)
        # lowPath = str(pathSave + '/data')
        # if not os.path.exists(lowPath):
        #     os.makedirs(lowPath)
        # targetPath = str(pathSave + '/target')
        # if not os.path.exists(targetPath):
        #     os.makedirs(targetPath)

        cv2.imwrite(str(pathSave+'/'+file),downImg)
        # shutil.copyfile(pathImage,str(targetPath+'/'+file))

        # if index > 10:
        #     break


