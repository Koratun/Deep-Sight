import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import DroneDataGenerator as gen
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# # Deprecated function. tf Datasets are better.
# def preprocess_data():
#     runs = os.listdir("Data")
#     image_pairs = []
#     dY = []
#     for run in runs:
#         print("Processing ", run)
#         #Get all Y data for the run
#         with open("Data/"+run+"/yPos.txt", 'r') as yPosFile:
#             yData = [float(s) for s in yPosFile.readlines()]

#         #Get all photo data for the run
#         allPhotos = []
#         photoFiles = os.listdir("Data/"+run+"/photos")
#         for photoFile in photoFiles:
#             allPhotos.append(imageio.imread("Data/"+run+"/photos/"+photoFile))

#         #Combine photos and y position data in sets of twos
#         for i in range(0, len(allPhotos)-1):
#             image_pairs.append([allPhotos[i], allPhotos[i+1]])
#             dY.append(yData[i+1]-yData[i])

#     np_image = np.array(image_pairs)

#     #Split image pairs for loading into the neural network
#     #Dimensions are: Batch, image pair, x, y, RGBA
#     #Then squeeze the image pair dimension out.
#     return [np.squeeze(np_image[:, :1, :, :, :]), np.squeeze(np_image[:, -1:, :, :, :])], np.array(dY)




# def load_data(image_files):
#     image_file, image_file2 = bytes.decode(image_files.numpy()[0]), bytes.decode(image_files.numpy()[1])
#     # Extract number of png file
#     run_folder = image_file[:image_file.rfind('\\')][:-6]
#     pic_number = int(image_file[image_file.rfind('\\')+1:image_file.find('.')])

#     # Grab the y positions for the indicated pictures, then find their difference
#     with open(run_folder+"\\yPos.txt", 'r') as yPosFile:
#         for _ in range(pic_number):
#             yPosFile.readline()
#         oldY = float(yPosFile.readline())
#         dY = float(yPosFile.readline()) - oldY

#     # Load in the images from their file names, and strip the Alpha value from the RGBA values. It's always 255, so we don't need that extra data.
#     image = imageio.imread(image_file)
#     # Scale the RGB data down to between 0-1 so that the model has an easier time creating weights.
#     return image[:, :, :-1]/255, imageio.imread(image_file2)[:, :, :-1]/255, dY


# # Takes the list of output from the load_data function (which must be wrapped in tf.py_function)
# # and outputs the data in the nested structure necessary for training, which the map function can process.
# # Unfortunately, the py_function cannot output nested data structures, so we have to do a little wrapping here.
# def load_data_wrapper(image_files):
#     image, image2, dY = tf.py_function(load_data, [image_files], [tf.float32, tf.float32, tf.float32])
#     return ([image, image2], dY)


# # Takes dataset like [0, 1, 2, 3, 4]
# # and converts it to: [[0,1],[1,2],[2,3],[3,4]]
# def prep_dataset(dtst):
#     # First repeat individual elements, then print those repeated elements after each other
#     dtst = dtst.interleave(lambda x: tf.data.Dataset.from_tensors(x).repeat(2), cycle_length=2, block_length=2)
#     # Skip the first element so that numbers are paired with the next greatest in the sequence with the batch function. 
#     return dtst.skip(1).batch(2, drop_remainder=True) #.take_while(lambda x: tf.squeeze(tf.greater(tf.shape(x), 1)))


# def tf_load_data():
#     runs = os.listdir("Data")
#     image_datasets = None

#     for run in runs:
#         image_dataset = tf.data.Dataset.list_files("Data/"+run+"/photos/?.png", shuffle=False).apply(prep_dataset)
#         image_dataset = image_dataset.map(load_data_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#         if image_datasets == None:
#             image_datasets = image_dataset
#         else:
#             image_datasets = image_datasets.concatenate(image_dataset)

#     #print(image_datasets)

#     image_datasets = image_datasets.shuffle(buffer_size=int(599*25/32)).batch(32)

#     # for data in image_datasets.take(1):
#     #     print(data)

#     return image_datasets


def main():
    # Create model

    # Start with smaller model that processes the two images in the same way.
    single_image_input = keras.Input(shape=(128,128,3))

    
    image = layers.Conv2D(32, (3,3))(single_image_input)
    image = layers.LeakyReLU()(image)
    image = layers.BatchNormalization()(image)
    # Run through MaxPool2D to help the algorithm identify features in different areas of the image.
    # Has the effect of downsampling and cutting the dimensions in half.
    image = layers.MaxPool2D()(image)

    image = layers.Conv2D(64, (3, 3))(image)
    image = layers.LeakyReLU()(image)
    image = layers.BatchNormalization()(image)
    image = layers.Dropout(.3)(image)

    # Reduce feature map to one channel in the last dimension.
    image = layers.Conv2D(1, (3, 3))(image)
    image = layers.LeakyReLU()(image)

    image_model = keras.Model(single_image_input, image)
    
    # Create larger model
    image_inputs = keras.Input(shape=(2,128,128,3))

    first_image, second_image = tf.split(image_inputs, num_or_size_splits=2, axis=1)
    first_image, second_image = tf.squeeze(first_image), tf.squeeze(second_image)

    image_outputs = [image_model(first_image), image_model(second_image)]
    model = layers.Concatenate()(image_outputs)

    # Use the two image-shaped feature maps in a convolutional layer to create a new feature map.
    # This should detect the movement of features between the two images.
    model = layers.Conv2D(64, (3, 3))(model)
    model = layers.LeakyReLU()(model)
    model = layers.BatchNormalization()(model)
    model = layers.Dropout(.3)(model)

    model = layers.Flatten()(model)

    # Input the rotational data and expand it to the same shape as the feature map.
    rotational_input = keras.Input(shape=(3,2))
    # Flatten the input
    rotation_portion = layers.Flatten()(rotational_input)

    # Run through dense layer
    rotation_portion = layers.Dense(64)(rotation_portion)
    rotation_portion = layers.LeakyReLU()(rotation_portion)
    rotation_portion = layers.BatchNormalization()(rotation_portion)
    rotation_portion = layers.Dropout(.2)(rotation_portion)

    model = layers.Concatenate()([model, rotation_portion])

    model = layers.Dense(128)(model)
    model = layers.LeakyReLU()(model)
    model = layers.BatchNormalization()(model)
    model = layers.Dropout(.3)(model)

    # Output is change in position of drone
    out_layer = layers.Dense(3, activation='linear')(model)

    final_model = keras.Model([image_inputs, rotational_input], out_layer)
    final_model.compile(loss="mse", optimizer=optimizers.Adam(lr=0.0003, beta_1=0.7))

    image_model.summary()

    final_model.summary()

    return final_model


    


# Plots the history of the training algorithm
def plot_losses(history):
    sns.set()  # Switch to the Seaborn look
    plt.plot(history.history['loss'], label='Training set',
             color='blue', linestyle='-')
    plt.xlabel("Epochs", fontsize=30)
    plt.ylabel("Loss", fontsize=30)
    plt.xlim(0, len(history.history['loss']))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=30)
    plt.show()


# Tests the model on a single input from the train generator
def test_model(model, train_gen: tf.keras.utils.Sequence):
    # Grab the first image from the train generator
    image_data, rot, posData = train_gen[0]
    # Run the model on the image
    prediction = model.predict([image_data, rot])
    # Get the average error between the prediction and the actual position
    error = np.average(np.abs(prediction - posData.numpy()))
    print("Average error over 32 samples:", error)
    # Print the predicted data and actual data for one sample
    print("Predicted:", prediction[0])
    print("Actual:", posData.numpy()[0])


if __name__ == "__main__":
    model = main()
    #model = keras.models.load_model("D:\\Documents\\Programming\\Night Tours\\DroneMovement")

    #Preprocess data
    print("Loading and processing data...")
    gen.datadir = "Data"
    train_data_generator = gen.DroneDataGenerator(batch_size=32)

    # Convert from keras sequence to tf.data.Dataset
    tfgen = tf.data.Dataset.from_generator(lambda: range(len(train_data_generator)), tf.uint8)
    tfgen = tfgen.shuffle(buffer_size=len(train_data_generator), reshuffle_each_iteration=True)
    tfgen = tfgen.map(lambda i: tf.py_function(train_data_generator.__getitem__, inp=[i], Tout=(
        tf.float32, tf.float32, tf.float32)), 
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Reshape the data to the proper shape
    tfgen = tfgen.map(lambda image, rot, pos: ((image, rot), pos), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # tfgen = tf.data.Dataset.from_generator(gen.DroneDataGenerator, args=[32], output_signature=(
    #     (tf.TensorSpec(shape=(None, 2, 128, 128, 3), dtype=tf.float32),
    #     tf.TensorSpec(shape=(None, 3, 2), dtype=tf.float32)),
    #     tf.TensorSpec(shape=(None, 3), dtype=tf.float32)
    # ))
    
    # # autotune the number of samples that are prefetched
    tfgen = tfgen.prefetch(tf.data.experimental.AUTOTUNE)
    tfgen = tfgen.repeat()

    # print the data from one sample from the generator
    # print(tfgen.take(1))

    callbacks = [keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1, write_graph=True, write_images=True, profile_batch=(20, 40))]

    #Train model
    history = model.fit(tfgen, steps_per_epoch=len(train_data_generator), epochs=2, callbacks=callbacks)
    
    test_model(model, train_data_generator)

    #model.save("D:\\Documents\\Programming\\Night Tours\\DroneMovement")

    # Not helpful when you only have 1 epoch
    #plot_losses(history)