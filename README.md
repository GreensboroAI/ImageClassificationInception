# ImageClassificationInception
Using the InceptionV3 model retrain top layers on our data

# If you are new to machine learning this is a great one to start with
We will use a pretrained CNN model with 'imagenet' weights to quickly build an image classification system.

# What packages you need
Python with Keras, Numpy

# How you need to structure your data
Since we are using the Keras flow_from_directory feature to make life easier on us you simply need to have your pictures structured in the following format within your project folder
  
Simply have a folder that says 'input' then create two folders in that one that say 'train' and 'test'. In the train folder create one folder for each class you want to predict so say for example 'dog' and 'cat'. Fill those two training folders with dog and cat images respectively. For the test folder you just fill it with images of dogs and cats.

To provide a quick explanation the flow_from_directory feature allows us to have Keras take a few images at a time from the given directory and 'flow' them in to the Inception model. This means that we do not have to save all the images at one time in to memory and allows us to work with much larger datasets on smaller computers than would otherwise be possible. So for example we set batch_size to 2 and this means that Keras takes 2 images at a time from the train directory and passes them to the Inception model. Once it has done those two it removes them from memory and passes 2 more to the model, so on and so on until it has processed them all.

# What variables you need to change

Now all you need to do is change a few variables in the script

change 'num_classes' to whatever number of classes you have. For the 'dog' and 'cat' example it would be 2

You do not NEED to change anything else but you can change the 'img_width' and 'img_height' to whatever you want, possible smaller size if you want to train faster. Also you can change 'batch_size' to whatever works best with your GPU/CPU
