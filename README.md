## Vehicle-Detection and Tracking
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

by Mas Chano (Feb 2017 Cohort)

### Project Information
This repository contains my implementation of the Vehicle Detection and Tracking project for the Udacity Self-Driving Car Engineer Nanodegree (original Udacity project repo can be found [here](https://github.com/udacity/CarND-Vehicle-Detection)). 

### Project Goals

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### Dataset used for Training
A dataset provided by Udacity was used for the training of the classifier. Links to the labeled dataset can be obtained through the [Udacity project repository](https://github.com/udacity/CarND-Vehicle-Detection). The example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself. To run `writeup.ipynb`, the dataset must be extracted to the appropriate directory as defined in the notebook. 

### Description of Files

The repository contains the following files/directories:

| File / Directory  | Description |
| ---               | ---   |
| test_sequence/    | Sample images used for visualization of algorithm  |
| test_sequence2/   | Sample images used for visualization of algorithm  |
| test_videos/      | Original highway driving video file.  |
| [output_videos/](https://github.com/mchano/CarND-P05-Vehicle-Detection/tree/master/output_videos)  | Directory containing video showing tracked vehicles  |
| [writeup.ipynb](https://github.com/mchano/CarND-P05-Vehicle-Detection/blob/master/writeup.ipynb) | Jupyter notebook that contains the vehicle detection algorithm and summary of results. |
| [writeup.md](https://github.com/mchano/CarND-P05-Vehicle-Detection/blob/master/writeup.md) | Markdown containing results of Jupyter Notebook `writeup.ipynb` |
