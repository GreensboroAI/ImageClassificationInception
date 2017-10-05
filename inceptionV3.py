from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np # linear algebra

#Changing image width and height to be 32 to see if I can run with this first in a managable amount of time
img_width  = 128 #was 128
img_height = 128 #was 128
batch_size = 4
num_classes = 5

train_data_dir = 'C:\\Users\\DanJas\\Desktop\\KerasFoodItentifier\\input\\train'
#validation_data_dir = 'D:\\CDiscount\\output\\validation'
test_data_dir = 'C:\\Users\\DanJas\\Desktop\\KerasFoodItentifier\\input\\test'

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer with num_classes
predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest',
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True) #shuffle was false

class_dictionary = train_generator.class_indices

train_steps_per_epoch = np.math.ceil(train_generator.samples / train_generator.batch_size)

# train the model on the new data for a few epochs
model.fit_generator(train_generator,
                    steps_per_epoch=train_steps_per_epoch,
                    epochs=2)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

model.fit_generator(train_generator,
                    steps_per_epoch=train_steps_per_epoch,
                    epochs=100)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size = batch_size,
    class_mode=None,
    shuffle=False)

test_order = test_generator.filenames

test_steps_per_epoch = np.math.ceil(test_generator.samples / test_generator.batch_size)

predictions = model.predict_generator(generator=test_generator, steps=test_steps_per_epoch, max_queue_size=2)

argmax = np.argmax(predictions, axis=1)

predictions_classes = []

print(test_order)
print(argmax)
print(len(argmax))
print(class_dictionary)
for i in argmax:
    predictions_classes.append(list(class_dictionary.keys())[list(class_dictionary.values()).index(i)])
print(predictions_classes)
print(len(predictions_classes))
