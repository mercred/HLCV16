# How to use the code?
1. Download data from Kaggle, https://www.kaggle.com/c/state-farm-distracted-driver-detection/data
2. Go to current folder (e.g. project). Folder 'project' should contain starter code, folder 'imgs' that has train and test subfolder with images.

# TODO
1. Speed up data loading. Maybe save it as pickle dump for future uses.
Checkout this link https://www.kaggle.com/inoryy/state-farm-distracted-driver-detection/fast-image-pre-process-in-parallel/discussion
2. Add code to do cross validation based on drivers. There are 26 drivers in total.
3. Optimize memory usage: load and preprocess in batches?

# Project road map
1. Setup baseline CNN model
2. Improve on baseline using data jittering, augmentation, mean img subtraction, etc.
3. Use pretrained network to do transfer learning
4. Try RCNN, R*CNN, or some other model.

# Important dates
1. 11.07 Interim presentation (progress report / problems encountered / feedback)
2. 25.07 Final presentation
3. 28.07 Written report submission

# Resources
1. Kaggle competition web site: https://www.kaggle.com/c/state-farm-distracted-driver-detection
