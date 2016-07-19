1. Set DATA_DIR in utils/load.py to point to directory with data:
   Data dir should contain the following:
   - test folder with test images
   - train folder with train images
   - driver_imgs_list.csv

	DATA_DIR = 'D:/hlcv-project/imgs'


2. In nets/net.py change path variable in the following function definition.
	Path points to location of vgg16 pretrained model (can be downloaded from https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)

	def vgg_std16_model(img_rows, img_cols, color=True, path='data/vgg16_weights.h5'):