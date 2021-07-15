import cv2
import os
import argparse

parser = argparse.ArgumentParser(description='Crop images into 2x2')
parser.add_argument('--path', default='Cropped_training_images', type=str, help='path to folder')
parser.add_argument('--save_path', default='histo_split_4', type=str, help='path to folder')

if __name__ == '__main__':
    opt = parser.parse_args()
    pathDataset = opt.path
    pathSave = opt.save_path

    # loading images from folder
    files = os.listdir(pathDataset)
    if '.DS_Store' in files:
        files.remove('.DS_Store')
        
    index = 1
    for file in files:
        print('{}. File name: {}'.format(index,file))
        img = cv2.imread(str(pathDataset + '/'+ file))
        w,h = img.shape[:2]

        # cropping images into 4
        crop_img1 = img[0:int(w/2), 0:int(h/2)]
        crop_img2 = img[0:int(w/2), int(h/2):h]
        crop_img3 = img[int(w/2):w, 0:int(h/2)]
        crop_img4 = img[int(w/2):w, int(h/2):h]

        # cv2.imshow("cropped1", crop_img1)
        # cv2.imshow("cropped2", crop_img2)
        # cv2.imshow("cropped3", crop_img3)
        # cv2.imshow("cropped4", crop_img4)
        # cv2.waitKey(0)

        # splitting images into folders 
        firstCut = round(len(files)*(80/100)) # 80%
        if index <= firstCut:
            newPath = pathSave + '/train' 
        elif index <= (firstCut + round(len(files)*(10/100))): # next 10%
            newPath = pathSave + '/val' 
        else:
            newPath = pathSave + '/test' # last 10%

        # saving the images with - and a part number to easily 
        # reconstruct later
        newFile = file.split(".")
        cv2.imwrite(str(newPath+'/'+newFile[0]+'-p1.jpg'),crop_img1)
        cv2.imwrite(str(newPath+'/'+newFile[0]+'-p2.jpg'),crop_img2)
        cv2.imwrite(str(newPath+'/'+newFile[0]+'-p3.jpg'),crop_img3)
        cv2.imwrite(str(newPath+'/'+newFile[0]+'-p4.jpg'),crop_img4)

        index += 1
        # break


