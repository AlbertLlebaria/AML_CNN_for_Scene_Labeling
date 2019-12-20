import utils

utils.split_sift_dataset('siftData', 'sift_dataset')

utils.matToTxtLabels('sift_dataset/train/labels/')
utils.matToTxtLabels('sift_dataset/test/labels/')
