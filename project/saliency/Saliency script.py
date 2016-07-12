import cv2
import pySaliencyMap
import glob
import os

def load_images_from_folder(folder):
    images = []
    glob_images=glob.glob(folder+'\\*.jpg')
    for filename in glob_images:
            img = cv2.imread(filename)
            images.append(img)
    return images
    
def main():
  #initialize
  sm = pySaliencyMap.pySaliencyMap(640, 480)
# read
  path_to_folder=os.path.dirname(os.path.abspath(__file__))
  images=load_images_from_folder(path_to_folder+'\\imgs\\train\\c0')
  p=0
  for image in images:     
    salient_region = sm.SMGetSalientRegion(image)
    salient_region_image=cv2.cvtColor(salient_region, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path_to_folder+'\\salient\\c0\\'+str(p)+'.jpg',salient_region_image)
    p+=1

if __name__ == '__main__':
  main()
