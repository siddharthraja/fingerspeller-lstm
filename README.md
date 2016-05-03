# ASL Finger Spelling Recognition using LSTM

This project aims to recognize finger spelling from hand poses presented as RGB images.

model.lua provides a sample run of the model on a pre-selected sequence of training and testing images chosen for the purpose of this demonstration. With everything else in place, this file can be run as simply as "th model.lua"

## Image Localization
We use Google Incpetion architecture to achieve this. The main library used here is TensorFlow

## Depth Estimation
We use a residual network for this. This is implemented in TensorFlow again.

## Demo Run
#### th model.lua
This will do a sample run for a chunked data set consisting of a sequence length of 5. Make sure to unzip the hand_images.zip file before running this command.
