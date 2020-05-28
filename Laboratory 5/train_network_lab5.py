# first neural network with keras tutorial
from numpy import loadtxt
from tensorflow import keras
from tensorflow.keras import metrics
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D,MaxPooling2D,Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd


def retrain_network(img_width, img_height):
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    val_generator = \
        val_datagen.flow_from_directory('data/validation_set'
            , target_size=(img_width, img_height), batch_size=32, class_mode='categorical')

    train_generator = \
        train_datagen.flow_from_directory('data/training_set'
            , target_size=(img_width, img_height), batch_size=32, class_mode='categorical')

    # Configure the CNN (Convolutional Neural Network).
    classifier = Sequential()

    # Convolution - extracting appropriate features from the input image.
    classifier.add(Conv2D(16, (3, 3), input_shape=(img_width, img_height, 3),activation='relu',padding='valid'))
    # Pooling: reduces dimensionality of the feature maps but keeps the most important information.
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # add dropout
    #model.add(Dropout(args.dropout))

    # Flattening layer to arrange 3D volumes into a 1D vector.
    classifier.add(Flatten())# DO NOT CHANGE!

    # Fully connected layers: ensures connections to all activations in the previous layer.
    classifier.add(Dense(units=64, activation='relu'))
    classifier.add(Dense(units=4,activation='softmax')) # DO NOT CHANGE!

    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(classifier.summary())
    history=classifier.fit(train_generator, epochs=100,
                             validation_data=val_generator,
                             validation_steps=30)
    
    # list all data in history
    print(history.history.keys())
    import matplotlib.pyplot as plt
    # import matplotlib.image as mpimg
    plt.plot(history.history['loss'][:])
    plt.plot(history.history['val_loss'][:])
    plt.legend(['train', 'val'], loc='upper left')

    plt.ylabel('MSE loss')
    plt.xlabel('epoch')
    plt.title('Loss function MSE')

    image_path="training_class1.png"
    plt.savefig(image_path)
    plt.clf()

    classifier.save("classifier.h5")
    return classifier


def load_saved_model(model_name):
    classifier=load_model(model_name)  
    print(classifier.summary())
    return classifier

def test_trained_network(classifier,img_width, img_height):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    val_datagen = ImageDataGenerator(rescale=1. / 255)


    val_generator = \
        val_datagen.flow_from_directory('data/validation_set'
            , target_size=(img_width, img_height), batch_size=32, class_mode='categorical')


    test_generator = \
        test_datagen.flow_from_directory('data/test'
            , target_size=(img_width, img_height), batch_size=32, class_mode='categorical')

    test_per_class=20

    test_generator = test_datagen.flow_from_directory(
        directory='data/test',
        target_size=(img_width, img_height),
        color_mode="rgb",
        batch_size=test_per_class,
        class_mode="categorical",
        shuffle=False)

    STEP_SIZE_VAL=val_generator.n//val_generator.batch_size
    scores_val=classifier.evaluate_generator(generator=val_generator,
    steps=STEP_SIZE_VAL)
    print("validation accuracy = ", scores_val[1])

    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
    scores_test=classifier.evaluate_generator(generator=test_generator,
    steps=STEP_SIZE_TEST)
    print("test accuracy = ", scores_test[1])


def predict_class_file(classifier,img_str,img_width, img_height):
    
    # PREDICT THE CLASS OF ONE IMAGE
    img = image.load_img(img_str, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = classifier.predict_classes(images, batch_size=1)
    classes_predic = classifier.predict(images, batch_size=1)
    class_names=("carton","glass","metal","plastic")
    print (classes[0],class_names[classes[0]])
    return classes[0],class_names[classes[0]]

def main():
    img_width=64
    img_height=64
    # ------------------ YOUR CODE HERE-------------------------------------------------#

#   UNCOMMENT WHEN YOU WAN TO LOAD A MODEL INSTEAD OF TRAINING A NEW ONE
    # classifier=load_saved_model("pretrained_classifier.h5") 
#   UNCOMMENT WHEN YOU WANT TO TRAIN A MODEL INSTEAD OF LOADING
    classifier=retrain_network(img_width, img_height) 
    
    # -------------------YOUR CODE ENDS HEEW---------------------------------------------#
    test_trained_network(classifier,img_width, img_height)
    class_names=("carton","glass","metal","plastic")
    class_code,class_name=predict_class_file(classifier,"current_image.png",img_width, img_height)


if __name__ == '__main__':
    main()  