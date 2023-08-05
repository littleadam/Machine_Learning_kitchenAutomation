
The purpose of the project is to involve AI to reduce the effort of humans in the kitchen
This code assumes that you have a pre-processed dataset of images of vegetables, along with their labels, stored in the vegetable_training_data.npy and vegetable_training_labels.npy files, respectively. It also assumes that you have a pre-trained object detection model stored in the path/to/saved_model directory. The code uses the first camera available on the system (index 0) to capture the image. If you have multiple cameras, you can specify a different index to use a different camera. Additionally, the code assumes that you have added the appropriate preprocessing steps for the object detection model you are using.

The tool suggests cooking recepies and runs a full cooking schedule with a phone And headset
