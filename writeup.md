## Self-Driving Car Engineer Nanodegree
### Project: Vehicle Detection

Author: Mas Chano <br />
Date: 2017/05/26
***
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/1car_example.png
[image2]: ./output_images/2non_car_example.png
[image3]: ./output_images/3hog_images.png
[image4]: ./output_images/4sliding_windows.png
[image5]: ./output_images/5pipeline.png
[video1]: ./test_output.mp4

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code this project can be found in the IPython notebook `writeup_clean.ipynb`. For HOG feature extraction, please see heading **1. Histogram of Oriented Gradients**. All other headings referenced in this document can also be found in the notebook.

In **1.1**, I examine the Car and Non-Car dataset (GTI and KITTI). In particular, the GTI dataset contains time series data, which means it contains batches of continuous time-series images. As an example, time-series data for both Car and Non-Car are provided below.

![alt text][image1]
![alt text][image2]

In **1.2**, I read in the data and split it into `training` and `test` sets. Particular care is taken to ensure that we minimize mixing time-series batches from the GTI dataset across the `training` and `test` set. This is done by taking advantage of the naming order of the images in the GTI dataset.

In hindsight, it may have been too early to split the dataset in to `training` and `test` sets due to the fact that fitting and scaling have to be performed on all data in the following steps. However, proper action was taken to make sure that the correct fitting and scaling was applied to the split data.

In **1.3**, I explore the various parameters available to tweak the feature extraction. Examples of a HOG feature extraction on the Y channel after a RGB to YCbCr conversion can be observed below. Here the following settings were used: `orient=10`, `pix_per_cell = 8`, and `cell_per_block=2`.

![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

In the end, I ended up with the following.

```python
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 10  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (24, 24) # Spatial binning dimensions
hist_bins = 24    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```

These settings were driven by trial and error. YCrCb was chosen because the Y Channel seemed effective compared to other channels in other color_spaces. I believe this is due to the fact that luma component was rather consistent despite cars appearing in a multitude of colors. This made it a good feature to extract. All channels in YCrCb were used due to the increased accuracy provided.

Increasing the orientation of the HOG detection beyond 9 or 10 didn't increase accuracy. Similarly decreasing spatial size and histogram binning to something like 16 x 16 and 16, respectively, caused accuracy to drop a little bit.

In general, an attempt was made to keep the resulting feature vector from exceeding 7000-8000 in length while keeping accuracy of the classifier in the 98% to 99% range. Increasing the feature vector length adversely affects the classifier training time but more importantly, adversely affects feature extraction time. Increased feature extraction time results in increased processing time when used in the detection pipeline.

With the settings described above, accuracy was generally in the 98.5-98.9% range.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the HOG features, spatial features, and histogram features as can be seen in **1.4** - **1.6**. The feature vector length ended up being 7680, and is mostly made up of the HOG features. Training took approximately 40 seconds using roughly 14000 training data.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window was implemented as written under heading **2. Sliding Window Search**. The functions used are based on the Hog Sub-sampling method provided in the lectures. The following image visualizes the windows that were swept across the image.

![alt text][image4]

Here, five scales were used: 1, 1.25, 1.5, 2, 2.75. For scales 1, 1.25, and 1.5, multiple sweeps were performed as can be seen by the overlapping rows in the image. Overlap in the x and y were set at 75%. Values were selected to ensure both far and close vehicles were detected. Especially the scales 1 and 1.25 were needed to better detect the white car as it moved further away.

Currently, the number of windows being evaluated is quite high and significantly slows down the final pipeline. Reducing the number of windows causes the pipeline to miss cars because it becomes less robust to variations in vehicle position.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The following are examples of the pipeline implemented on static but contiguous images. The code can be found in **2.3**. In order to optimize the performance of the pipeline, the sliding window region was limited to a narrow portion of the frame.

For the 5 scales, the following upper and lower y-boundaries were used.
```python
scales = [  1, 1.25, 1.5,   2, 2.75]
ystarts =[400,  395, 390, 385,  380]
ystops = [520,  520, 540, 545,  600]
```

![alt text][image5]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/test_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In order to eliminate false positives, the position of all detections were tracked in a class `BBoxes_Tracker` found in **2.1** (towards the bottom of the cell) and **2.3** (where it's implemented in the pipeline). This class held on to the last 16 frames worth of positive detection "boxes". From these historical detection "boxes", a heatmap was generated and thresholded. If within the last 16 frames, 26 or more detections (postive or negative) were made, the detection was deemed a vehicle.

The `scipy.ndimage.measurements.label()` was used to group connected / overlapping individual detections in the heatmap as a single entity. This entity is assumed to be the detected vehicle. The boundary of the entity is then overlayed on the frame as can be seen in the video or the previous screenshot.

Due to the large number of windows used in the pipeline, the threshold value and historical tracking of detections is quite deep. This is because there is a fairly high number of false positives despite an accuracy of over 98.5%.


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach was rather brute force with lots of trial and error. In particular, it was a struggle to identify how finely to slide the windows across the image because of variations in car location and distance (size). Also, certain cars (white) were harder to identify. Making the system more sensitive to the white car caused the pipeline to pick up white cars moving in the opposite lane.

The current setting represent a balance that minimizes missed detections and false positives. As can be seen, there are still frames where the white car will "disappear" when it goes too far away. To address this, it is possible that more training data on white cars at varying distances could solve this issue. Another possibility could be adding another scaled sweep with a smaller window.

The pipeline also struggles with some shadows. This is evident at the 41 second mark when the pipeline mistakes the shadow cast by the tree to be a car. It didn't help that a white truck simultaneously appeared in the same region. A way to avoid this could be to narrow the region of interest like we did in the lane finding projects to avoid searching on the extremities. However, doing so would reduce the use of this pipeline in non-highway situations and possibly reduce its robustness at the periphery. If we look at the end of the video, the pipeline is able to captures the speeding SUV coming from the right immediately.

Other limitations of this pipeline could be the processing time. It takes roughly 17 minutes to process the 50 second video on my 2013 MacBook Pro with 2.3 GHz Core i7. While we won't see this laptop used in a production vehicle, I think it's clear that further tweaks, parallelizing the code and/or use of more powerful or specialized hardware (Realtime / FPGA / GPU) would be necessary to do this in realtime.
