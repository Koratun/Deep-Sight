import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import numpy as np
import imageio
import os

# Deprecated function. tf Datasets are better.
def preprocess_data():
    runs = os.listdir("Data")
    image_pairs = []
    dY = []
    for run in runs:
        print("Processing ", run)
        #Get all Y data for the run
        with open("Data/"+run+"/yPos.txt", 'r') as yPosFile:
            yData = [float(s) for s in yPosFile.readlines()]

        #Get all photo data for the run
        allPhotos = []
        photoFiles = os.listdir("Data/"+run+"/photos")
        for photoFile in photoFiles:
            allPhotos.append(imageio.imread("Data/"+run+"/photos/"+photoFile))

        #Combine photos and y position data in sets of twos
        for i in range(0, len(allPhotos)-1):
            image_pairs.append([allPhotos[i], allPhotos[i+1]])
            dY.append(yData[i+1]-yData[i])

    np_image = np.array(image_pairs)

    #Split image pairs for loading into the neural network
    #Dimensions are: Batch, image pair, x, y, RGBA
    #Then squeeze the image pair dimension out.
    return [np.squeeze(np_image[:, :1, :, :, :]), np.squeeze(np_image[:, -1:, :, :, :])], np.array(dY)




def load_data(image_files):
    image_file, image_file2 = bytes.decode(image_files.numpy()[0]), bytes.decode(image_files.numpy()[1])
    # Extract number of png file
    run_folder = image_file[:image_file.rfind('\\')][:-6]
    pic_number = int(image_file[image_file.rfind('\\')+1:image_file.find('.')])

    # Grab the y positions for the indicated pictures, then find their difference
    with open(run_folder+"\\yPos.txt", 'r') as yPosFile:
        for _ in range(pic_number):
            yPosFile.readline()
        oldY = float(yPosFile.readline())
        dY = float(yPosFile.readline()) - oldY

    # Load in the images from their file names, and strip the Alpha value from the RGBA values. It's always 255, so we don't need that extra data.
    image = imageio.imread(image_file)
    # Scale the RGB data down to between 0-1 so that the model has an easier time creating weights.
    return image[:, :, :-1]/255, imageio.imread(image_file2)[:, :, :-1]/255, dY


# Takes the list of output from the load_data function (which must be wrapped in tf.py_function)
# and outputs the data in the nested structure necessary for training, which the map function can process.
# Unfortunately, the py_function cannot output nested data structures, so we have to do a little wrapping here.
def load_data_wrapper(image_files):
    image, image2, dY = tf.py_function(load_data, [image_files], [tf.float32, tf.float32, tf.float32])
    return ([image, image2], dY)


# Takes dataset like [0, 1, 2, 3, 4]
# and converts it to: [[0,1],[1,2],[2,3],[3,4]]
def prep_dataset(dtst):
    # First repeat individual elements, then print those repeated elements after each other
    dtst = dtst.interleave(lambda x: tf.data.Dataset.from_tensors(x).repeat(2), cycle_length=2, block_length=2)
    # Skip the first element so that numbers are paired with the next greatest in the sequence with the batch function. 
    return dtst.skip(1).batch(2, drop_remainder=True) #.take_while(lambda x: tf.squeeze(tf.greater(tf.shape(x), 1)))


def tf_load_data():
    runs = os.listdir("Data")
    image_datasets = None

    for run in runs:
        image_dataset = tf.data.Dataset.list_files("Data/"+run+"/photos/?.png", shuffle=False).apply(prep_dataset)
        image_dataset = image_dataset.map(load_data_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if image_datasets == None:
            image_datasets = image_dataset
        else:
            image_datasets = image_datasets.concatenate(image_dataset)

    #print(image_datasets)

    image_datasets = image_datasets.shuffle(buffer_size=int(599*25/32)).batch(32)

    # for data in image_datasets.take(1):
    #     print(data)

    return image_datasets


def main():
    # Create model

    # Start with smaller model that processes the two images in the same way.
    single_image_input = keras.Input(shape=(128,128,3))

    image = layers.Conv2D(64, (3,3))(single_image_input)
    image = layers.LeakyReLU()(image)
    image = layers.BatchNormalization()(image)
    # Run through MaxPool2D to help the algorithm identify features in different areas of the image.
    # Has the effect of downsampling and cutting the dimensions in half.
    image = layers.MaxPool2D()(image)

    image = layers.Conv2D(128, (3, 3))(image)
    image = layers.LeakyReLU()(image)
    image = layers.BatchNormalization()(image)
    image = layers.Dropout(.3)(image)

    image_model = keras.Model(single_image_input, image)
    
    # Create larger model
    first_image, second_image = keras.Input(shape=(128,128,3)), keras.Input(shape=(128,128,3))

    image_outputs = [image_model(first_image), image_model(second_image)]
    model = layers.Concatenate()(image_outputs)

    model = layers.Flatten()(model)

    model = layers.Dense(128)(model)
    model = layers.LeakyReLU()(model)
    model = layers.BatchNormalization()(model)
    model = layers.Dropout(.3)(model)

    # Output is change in y-position of drone
    out_layer = layers.Dense(1, activation='linear')(model)

    final_model = keras.Model([first_image, second_image], out_layer)
    final_model.compile(loss="mse", optimizer=optimizers.Adam(lr=0.0003, beta_1=0.7))

    image_model.summary()

    final_model.summary()


    #Preprocess data
    print("Loading and processing data...")
    train_data = tf_load_data()

    #Train model
    final_model.fit(train_data, epochs=5)



if __name__ == "__main__":
    main()
    #tf_load_data()