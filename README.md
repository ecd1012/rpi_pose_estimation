# **Pose Esimation on the Raspberry Pi 4**

Results
=======
<img src="images/pose.gif" width="500" height="500">

Description of Repository
=========================
This repository contains code and instructions to configure the necessary software for running pose estimation on the Raspberry Pi 4!

Details of Software and Neural Network Model for Object Detection:
* Language: Python
* Framework: TensorFlow Lite
* Network: PoseNet with MobileNet-V1


The motivation for the Project
========================
The goal of this project was to how well pose estimation could perform on the Raspberry Pi.
Google provides code to run pose estimation on Android and IOS devices - but I wanted to write
python code to interface with and test the model on the Pi.

Additional Resources
===================
* **YouTube Turorial For This Repository**: 
* **Blog Pose on Posenet**: https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5
* **Pose estimation with TensorFlow Lite**:https://www.tensorflow.org/lite/models/pose_estimation/overview

Testing Configuration
=============================

Core
* Raspberry Pi 4 GB
* Raspberry Pi 5MP Camera (rev 1.3)

Setting Up Software
====================
1.) Please see my other post here: https://github.com/ecd1012/rpi_road_object_detection
And follow Setting Up Software Steps: 1-9 before proceding

2.)At this point you should have all the dependencies installed and your virtual environment activated

3.) Grab the sample TensorFlow Lite Posenet model from Google
```
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite
```


Running Pose Estimation
=================
15.) After all your hardware and software is configured correctly run the following command:
```
python3 TFLite_pose.py --modeldir notebooks/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite --output_path asdf
```
Where the --output_path you specify is where you want images saved.



