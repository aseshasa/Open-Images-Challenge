# Open-Images-Challenge
Object detection challenge on open images dataset

Open Images Challenge is an object detection challenge on a subset of the open images dataset consisting of 500 classes. The dataset for the competition uses 1.7M training images, 41K validation images. The challenge is evaluated using 100K test images. The total dataset is 0.6-0.7 TB. 

I used the Faster RCNN pretrained model to do object detection. I wrote code based on the object detection tutorial given with Tensorflow's Object Detection API. The code runs inference on the challenge2018 set. I used google cloud compute instances to process the data because of the huge quantities involved.

This requires Tensorflow's Object Detection API to run. It scores 0.36121 public score and 0.33093 private score on kaggle evaluation server. This achieves 43rd on kaggle's public and private leaderboards of the open images challenge. It takes 21 hours to run on a 4 core 15 GB RAM Nvidia Tesla P100 GPU instance. Other implementations on the same hardware take 27 hours.

Download model at http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_14_10_2017.tar.gz
Modify lines 19-34 for file paths. Run frcnn1.py to run inference on 99999 images in the OID challenge2018 set. 

Misc files are just files I used for experiments.

Things to do in future: Train YOLO on oid. Change Training Methods. Use subsets of the training set to train faster. Try modifying the learning rate.
