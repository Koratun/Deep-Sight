import tensorflow as tf
import numpy as np
import glob
import imageio
import random
import os


datadir = "Data"

class DroneDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size=32, shuffle=True):
        # Initialization
        self.datdir = datadir
        # Get a list of runs available in the data directory
        self.run_dirs = glob.glob(os.path.join(datadir, "Run*"))
        # Get the number of images in a run
        self.run_size = len(glob.glob(os.path.join(self.run_dirs[0], "photos", '*.png')))
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # read positional data
        xData = []
        for run in self.run_dirs:
            with open(os.path.join(run, 'xPos.txt'), 'r') as f:
                xData += [float(s) for s in f.readlines()]
        self.xData = np.array(xData)

        yData = []
        for run in self.run_dirs:
            with open(os.path.join(run, 'yPos.txt'), 'r') as f:
                yData += [float(s) for s in f.readlines()]
        self.yData = np.array(yData)

        zData = []
        for run in self.run_dirs:
            with open(os.path.join(run, 'zPos.txt'), 'r') as f:
                zData += [float(s) for s in f.readlines()]
        self.zData = np.array(zData)

        # Read rotational data and calculate their sines and cosines
        rData = []
        for run in self.run_dirs:
            with open(os.path.join(run, 'rotation.txt'), 'r') as f:
                rData += [[[np.cos(float(r)), np.sin(float(r))] for r in s.split(', ')] for s in f.readlines()]
        self.rData = np.array(rData)
        # Shape will be:
        # [[[cos(x), sin(x)], [cos(y), sin(y)], [cos(z), sin(z)]], [and so on...]]
        # [image, xyz, cos/sine]

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
        
        pos = np.zeros((self.batch_size, 3), dtype='float32')
        image_concat_pair_batch = np.zeros((self.batch_size, 2, self.xsize, self.ysize, 3), dtype='float32')
        rotation_difference = np.zeros((self.batch_size, 3, 2), dtype='float32')

        for count, ind in enumerate(indexes):
            # get images
            curr_image = imageio.imread(self.photofiles[ind])
            next_image = imageio.imread(self.photofiles[ind+1])
            # get positions
            curr_pos = np.array([self.xData[ind], self.yData[ind], self.zData[ind]])
            new_pos = np.array([self.xData[ind+1], self.yData[ind+1], self.zData[ind+1]])
            # diff
            pos[count] = new_pos-curr_pos
            # get rotations
            rotation_difference[count] = self.rData[ind+1] - self.rData[ind]
            # expand axis
            curr_image = np.array(curr_image)[np.newaxis, :, :, :3]
            next_image = np.array(next_image)[np.newaxis, :, :, :3]
            # concatenat image pairs
            image_concat = np.concatenate((curr_image, next_image), axis=0)/255.
            # store
            image_concat_pair_batch[count] = image_concat
        
        return tf.convert_to_tensor(image_concat_pair_batch, dtype=tf.dtypes.float32), tf.convert_to_tensor(rotation_difference, dtype=tf.dtypes.float32), tf.convert_to_tensor(pos, dtype=tf.dtypes.float32)

    def on_epoch_end(self):
        if self.shuffle == True:
            self.indexes = random.sample(self.pickable_indexes, k=self.set_len)