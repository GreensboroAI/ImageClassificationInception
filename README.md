# ImageClassificationInception
Using the InceptionV3 model retrain top layers on our data

# If you are new to machine learning this is a great one to start with
We will use a pretrained CNN model with 'imagenet' weights to quickly build an image classification system.

# What packages you need
Keras, Numpy

# How you need to structure your data
Since we are using the Keras flow_from_directory feature to make life easier on us you simply need to have your pictures structured in the following format within your project folder

/input
  -/train
    -/*class1*
      -*bunch of images*
    -/*class2*
      -*bunch of images*
    -/*etc..*
  -/test
  
Simply have a folder that says 'input' then create two folders in that one that says 'train' and one that says 'test'. In the train folder create one folder for each class you want to predict so say for example 'dog' and 'cat'. Fill those two training folders with dog and cat images respectively. For the test folder you just fill it with images of dogs and cats.

# What variables you need to change

Now all you need to do is change a few variables in the script

change 'num_classes' to whatever number of classes you have. For the 'dog' and 'cat' example it would be 2

You do not NEED to change anything else but you can change the 'img_width' and 'img_height' to whatever you want, possible smaller size if you want to train faster. Also you can change 'batch_size' to whatever works best with your GPU/CPU
