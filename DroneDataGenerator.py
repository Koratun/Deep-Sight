import tensorflow as tf
import numpy as np
import glob
import imageio
import random
import os


class DroneDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, datadir, batch_size=32, shuffle=True):
        # Initialization
        self.datdir = datadir
        # Get a list of runs available in the data directory
        self.run_dirs = glob.glob(os.path.join(datadir, "Run*"))
        # Get the number of images in a run
        self.run_size = len(glob.glob(os.path.join(self.run_dirs[0], "photos", '*.png')))
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # read Y positions
        yData = []
        for run in self.run_dirs:
            with open(os.path.join(run, 'yPos.txt'), 'r') as f:
                yData += [float(s) for s in f.readlines()]
        self.yData = np.array(yData)
        #print(self.yData.shape)

        self.photofiles = glob.glob(os.path.join(datadir, "Run*", 'photos', '*.png'))
        #print(len(self.photofiles))
        # get image size
        image0 = np.array(imageio.imread(self.photofiles[0]))
        self.xsize, self.ysize, _ = image0.shape
        #print(self.xsize)
        #print(self.ysize)
        
        # Gets the number of image pairs in the dataset (coincidentally will be the legnth of pickable indexes)
        self.set_len = (self.run_size-1)*len(self.run_dirs)

        # Generate possible indexes for any sample
        self.pickable_indexes = []
        for i in range(len(self.run_dirs)):
            self.pickable_indexes += range(i*self.run_size, (i+1)*self.run_size-1)

        #print(len(self.pickable_indexes))
        #print(self.set_len)
        self.indexes = self.pickable_indexes

        if self.shuffle == True:
            self.indexes = random.sample(self.pickable_indexes, k=self.set_len)
        
    def __len__(self):
        return self.set_len//self.batch_size

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        ypos = np.zeros((self.batch_size,), dtype='float32')
        image_concat_pair_batch = np.zeros((self.batch_size, 2, self.xsize, self.ysize, 3), dtype='float32')
        
        for count, ind in enumerate(indexes):
            # get images
            curr_image = imageio.imread(self.photofiles[ind])
            next_image = imageio.imread(self.photofiles[ind+1])
            # get positions
            curr_ypos = self.yData[ind]
            new_ypos = self.yData[ind+1]
            # diff
            ypos[count] = new_ypos-curr_ypos
            # expand axis
            curr_image = np.array(curr_image)[np.newaxis, :, :, :-1]
            next_image = np.array(next_image)[np.newaxis, :, :, :-1]
            # concatenat image pairs
            image_concat = np.concatenate((curr_image, next_image), axis=0)/255.
            # store
            image_concat_pair_batch[count] = image_concat
        
        return tf.convert_to_tensor(image_concat_pair_batch, dtype=tf.dtypes.float32), tf.convert_to_tensor(ypos, dtype=tf.dtypes.float32)

    def on_epoch_end(self):
        if self.shuffle == True:
            self.indexes = random.sample(self.pickable_indexes, k=self.set_len)