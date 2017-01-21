# Overview
This submission uses a custom designed and trained continuous output CNN model which is heavily based on the reference paper provided "End to End Learning fro Self-Driving Cars". Successive rounds of training, testing, deduction and refining were done to come up with a working solution. 

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

# Training Approach
Feed in some images, see how long to converge. Test the result. Make a hypothesis as to why it did not work, modify either the model, the generator parameters, the number of images to train with or the raw data itself. Train the model again, see what happens. Change 1 parameter, observe, rinse, repeat unil the final model is created. The process probably took close to 70 or 80 iterations until a working model was found.
