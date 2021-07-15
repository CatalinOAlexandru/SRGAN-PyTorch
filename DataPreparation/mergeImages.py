import cv2
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Crop images into 2x2')
parser.add_argument('--path', default='histo_split_4/val', type=str, help='path to folder')
parser.add_argument('--save_path', default='merged', type=str, help='path to folder')

if __name__ == '__main__':
    opt = parser.parse_args()
    pathDataset = opt.path
    pathSave = opt.save_path

    # loading images from folder
    files = os.listdir(pathDataset)
    if '.DS_Store' in files:
        files.remove('.DS_Store')

    files.sort()
    print('Merging...')

    for i in range(0,len(files),4):

        img1 = cv2.imread(str(pathDataset + '/'+ files[i]))
        img2 = cv2.imread(str(pathDataset + '/'+ files[i+1]))
        img3 = cv2.imread(str(pathDataset + '/'+ files[i+2]))
        img4 = cv2.imread(str(pathDataset + '/'+ files[i+3]))

        w,h = img1.shape[:2]

        top = np.concatenate((img1, img2), axis=1)
        bottom = np.concatenate((img3, img4), axis=1)
        whole = np.concatenate((top, bottom), axis=0)

        newFile = files[i].split("-")
        cv2.imwrite(str(pathSave+'/'+newFile[0]+'-merged.jpg'),whole)


        