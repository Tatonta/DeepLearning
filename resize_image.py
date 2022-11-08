import cv2
import os

path = 'C:\\Users\\Tatonta\\Desktop\\Deep Learning\\temp\\sc5_versione_ridotta'
resized_path = 'C:\\Users\\Tatonta\\Desktop\\Deep Learning\\temp\\sc5_resized'

im_size = 256

images = []
print(os.listdir(path))
for dir in os.listdir(path):
    os.mkdir(resized_path+"\\"+dir)
    for file in os.listdir(os.path.join(path,dir)):
        img = cv2.imread(os.path.join(path,dir+"\\"+file))
        img = cv2.resize(img, (im_size, im_size))
        images.append(img)
        cv2.imwrite(os.path.join(resized_path,dir+"\\"+file) + "_resized.jpg", img)