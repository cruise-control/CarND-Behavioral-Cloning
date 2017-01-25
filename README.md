# Overview
This submission uses a custom designed and trained continuous output CNN model which is heavily based on the reference paper provided "End to End Learning fro Self-Driving Cars" with some reference to the solution by comma.ai located at https://github.com/commaai/research. Successive rounds of training, testing, deduction and refining were done to come up with a working solution. 

## Challenges
* Getting useful data
* Getting good data
* Keyboard Bang-Bang steering angles
* Not knowing if the non-beta version was correct to use
* Potential problems with the speed of my development machine during simulation
* The model is a large block box this presented unique challenges
* The image generator also adds another element of "what is this actually doing" to the process

# Data
Ultimately the data that I used to train the network was provided by Udacity. I found it challenging
to get any useful data when I ran the simulator myself using the keyboard. The data that I got was so bad in fact
that when I augmented the Udacity data with my own, it caused the model to fail on a couple of specific corners 
on the track and also made the model steer the car in an unstable oscillatory way.
Unfortunatly I only realized this by refining my design with my augmented data and then 
finally removing my data, retraining and it worked!

## Training & Validation Data
Udacity provided training data was utilized. The center images and a *portion* of the Left and Right camera images were used for training. This portion is *1/40th* of the number of center image samples used per Epoch. When using more than this portion, the simulation become unstable or unreliable due to a 'kneejerk' reaction at the edges of the road. The Udacity training set by itself was insufficient and more data was required. Ultimately I ran out of memory on both my development machine and on the AWS so I implemented a Keras ImageGenerater to augment the training data. Per Epoch I loaded 15000 images from the original data set. This was divided into a 1/3 for validation and the remainder for training. Of the 10000 raw images for training, this was augmented to 20000 images from the generator.

## Generator
The following generator parameters were utilized:
```python
    train_datagen = ImageDataGenerator(
        rotation_range=3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.0,
        zoom_range=0.0,
        fill_mode='nearest',
        horizontal_flip=False,
        vertical_flip=False)
```
The logic behind the images is as follows:

* There may be a small amount of image rotation depending on the speed of the vehicle, cornering and suspension etc... 
* Adding width and height shift was approporiate as it enables the model to learn from a range of positions offset from the center image (and the Left and Right images). 
* There is no shear or zoom, these would not aid in training for the simulation. The camera is fixed and there should be no shear (at least in the simulation)
* There is no vertical flip as this would never present itself in any simulation. 
* There is no horizontal_flip. 
  - The primary reason for this is that I could not automatically modify the corresponding Y value (which would need to be flipped also) as a left turn would then become a right turn. If this is possible to do in the generator, I could not figure it out.

# Model Description
After analysing the problem I decided to implement a continous model where (ideally) the output of the model is the steering angle for the vehicle without needing further post-processing. A discrete model with binning of the steering angles was also considered but ultimately I thought that cars are not steered in chunks of angles so it would need post-processing or be an unrealistic solution.

The image is shrunk to 1/2 of its original size. The primary purpose for this is to reduce the computation requirements and it allowed (with lots of patience) training on my development machine for some runs. This provided a satisfactory solution. Future work (if I had the time) would be to experiment and see if the input images could be further reduced in size yet still yield a acceptable model.

## The Final Model 
1. The RGB images are reduced in size and fed into the model
2. Convolution2D > MaxPooling2D > ELU 
3. Convolution2D > MaxPooling2D > ELU 
4. Convolution2D > MaxPooling2D > ELU
5. Convolution2D > ELU
6. Convolution2D > ELU > Flatten > Dropout
7. Dense > TanH
8. Dense > Dropout > TanH
9. Dense 
10. Dense > TanH

* Total params: 385,883
* Trainable params: 385,883

A mixture of ELU and TanH activation functions were used. With only ELU the output angle was going beyond the allowed range during simulation. This resulted in wider swings of the vehicle couple with overcorrecting which eventualy led to it driving off of the road. Using TanH for the fully connected layers gave an output in the expected range of -1 to +1. This scheme was chosen after successive rounds of experimentation to determine the mixture of activation functions and dropout levels in the fully connected layer.

The optimizer used was the Adam optimizer with the mean squared error as the error metric. The default learning rate was used.

# Training Approach
1. Get a random set of data from the Udacity Data Set
  - Split into test and validation sets
  - Train the model with this data see how long it takes to converge.
2. Test the result
3. Make a hypothesis as to why it did not work / what failed
4. Modify one of the following:
  - The model
  - The generator parameters
  - The number of images to train with
  - the raw data itself
5. Go back to 1 until a working solution is available

## Train, Validation, Test Split
In my first submission, the Udacity data was specificially only used for **Training** and **Validation**. The reason for this was because the actual 'Test' is the running in the simulator and a Human's determination if this model is suitable. A subset of Test data could be kept back and used for Testing of the actual trained model, however, I have observed that a low error value can have little to do with how well the car behaves in the simulator. After thinking on this, a small sample of images is used to ascertain a reference value for how the model performed. This sample set is the first 1000 images in the total data set. It is not a random sample but it is only passingly used to ascertain the health of the model.

The process probably took close to 70 or 80 iterations until a working model was found.
## Some problems encountered during training
### Car oscillates between the road edges
Too much data from the Left and Right Cameras without a high enough bias on the steering angles for those images. I think the Left and Right images flooded the network enough so that the trained model could only ever steer between the sides of the road and not keep the center line. **Solution** reduced the Left and Right images and increased the bias on the steering angle. Also increased dropout
### Car immediately drives off the road
This happened when using bad data, too much data from the Left and Right Cameras and a (probably) overfitted network. **Solution** Reduced the bad data, reduced the Left and Right camera images and added more dropout. Also started training with many more images.
### Car gets confused on some corners
This was due to the approach angle. With some corners the car would approach too wide or too tight and eventually would drive off the road and follow the outside edge of the road on the grass until it crashed. **Solution** Increase the bias on the Left and Right Steering Angle and ensure that there was enough data from the Left and Right Cameras to keep the car in the center of the road but not too much so that it began oscillating.
### Car not responding very well
Sometimes the car would appear to be 'laggy' in its operation. I tried several things and ultimately decided to reduce the throttle from 0.2 to 0.1 in drive.py. This resulted in a much smoother simulation and was a major step toward the working solution. I believe that my development machine was not able to respond quick enough for the simulator to operate correctly.

