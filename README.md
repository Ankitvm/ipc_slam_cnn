# Title: 

# **Indoor Place Categorization for Visual SLAM**

## Team members

### Ankit Manerikar (*amanerik@purdue.edu*), 
#####  Robot Vision Lab, *Purdue University*
# 
*(This project has been implemented as a part of the coursework for the course BME 595 Deep Learning (Fall 2017) taught at Purdue University).*

## Summary

The problem of visual place categorization (VPC), i.e., the visual identification of an input scene as belonging to a particular semantic pre-trained category is a well-established one and has been tackled through the development of a large number of datasets as well as the design of CNNs using these datasets that give a high value of accuracy. For example, the Places205-AlexNet [1] trained on the MIT Places Scene Recognition Database [2] allows for a test accuracy of more than 95 % for the case of scene recognition. However, extending this CNN-based place categorization to Visual SLAM systems can result in a few implementational constraints. This project aims at developing a two-tier technique that provides an extension of CNNs trained for classification of 2D Visual Place Categorization to that of paritioning maps constructed from SLAM using semantic place labels.

## Implementation

Visual SLAM systems operate upon an input stream of images (either monocular, stereo or RGB-D) to construct a 3D Map of the environment while performing localization of the robot simultaneously. Since the sensory input to such a system is an image, it is only convenient to use CNNs trained for VPC to classify these input image frames using pre-trained labels as they are streamed for the purpose of SLAM. The main challenge is therefore to utilize these spatio-temporally generated images labels as feature vectors for parititioning the entire map into different places. 

This can be done using a two-tier technique [3] employing two classifiers back-to-back as follows:

- The first classifier is a VPC CNN trained on a Place Recognition dataset that operates on the 2D input images and assigns place labels to current pose of the robot, thus generating a spatio-temporal grid of class labels corresponding to the path traversed by the robot.
- The second classifier takes as input this grid of place labels and performs a semantic segmentation of the grid to partition the same into different places.

This implementation has been carried out for each of the classifiers as follows:

##### Classifier I - VPC CNN: 
This is a CNN trained on the Places365 dataset[4] for *indoor scene categories only* and  operates on the raw input stream of images. An pre-trained 18-layer ResNet CNN (trained on Places365) has been used for the purpose - this is restrained for implemeneting the VPC.

##### Classifier II - 2D Agglomerative Clustering Classifier:
This explores the spatio correlation between the scene labels to cluster them together and gives a refined labeled indoor map.

## Operational Details:
### System Pre-requisites:

To run this code, your machine should support:

- Python 2.7
- ROS Kinetic (or higher distro)
- Torch
- Cuda (optional)

### Installation Procedure 
(For Linux OS only)
 
In order to run the code, clone the ROS packages into your ROS catkin workspace using the following commands (To learn how to create a catkin workspace, check the following the [tutorial](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment#Create_a_ROS_Workspace) ):
 
```
$  cd ~/catkin_ws/src
$  git clone https://github.com/Ankitvm/ipc_slam_cnn.git
$  cd ..
$  catkin_ws
```
 
Once the package has been built, the demo code can be run using the following launch file:

```
$  source ~/catkin_ws/devel/setup.bash
$  roslaunch offline_slam start_offline_slam.launch
```
 
By default, the program performs the simulation on the REMP dataset maintained by Robot Vision Lab, Purdue University (found [here](https://engineering.purdue.edu/RVL/Research/PlaceRecognition/REMP_DataSet.tar)) and uses the trained Indoor Places365 CNN for scene labeling. The operation can be modifed by using the following options:

```
$ roslaunch offline_slam start_offline_slam.launch   dataset:=<path_to_dataset>   nn:=places365  
```

Options:
*dataset*    -  
the path to the dataset to be used; the offline simulation requires the Stereo Images, Robot Trajectory and Point Clouds to be included in the dataset - to check for operation, one can test the simulation on the other datasets available at [Robot Vision Lab, Purdue University](https://engineering.purdue.edu/RVL/Research/PlaceRecognition/index.html).

*nn*       -
The CNN to be used for scene labeling; three options are avaialble for selecting the CNN:
'places365'  - pre-trained places365 resnet18 CNN
'indoor_net' - re-trained resnet-18 CNN  for indoor place categorization (default)
'two-tier'   - the two-tier classifer with CNN and clustering classifier

Optionally, you can re-train the ResNet-18 CNN for indoor place recognition using the following command:

```
$   roscd offline_slam
$   python src/ResNetTrainer.py --data <path to places365 dataset>
```

To use this command, the places 365 dataset can be downloaded from [here](http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar)

Training the CNN results generates a new torch CNN package *indoor_resnet18_places365.pth.tar* which is saved in the */src* folder of the ROS package. In addition, plots for training curves are also generated which can be found in */Figures* folder within the ROS package directory.

Example plots for training and simulations can be found in [Results and figures](https://github.com/Ankitvm/ipc_slam_cnn/tree/master/Results%20and%20Figures).

## Results:

##### Unlabeled Indoor Map (REMP Dataset):
![](https://github.com/Ankitvm/ipc_slam_cnn/blob/master/Results%20and%20Figures/Indoor%20Maps/original_pt_cloud.png)
##### Labeled Indoor Map with *places365* CNN (REMP Dataset):
![](https://github.com/Ankitvm/ipc_slam_cnn/blob/master/Results%20and%20Figures/Indoor%20Maps/resnet_18_slam_result.png)
##### Labeled Indoor Map with *indoor_net* CNN (REMP Dataset):
![](https://github.com/Ankitvm/ipc_slam_cnn/blob/master/Results%20and%20Figures/Indoor%20Maps/indoor_net_18.png)
##### Labeled Indoor Map with *two-tier* CNN (REMP Dataset):
![](https://github.com/Ankitvm/ipc_slam_cnn/blob/master/Results%20and%20Figures/Indoor%20Maps/knn_classifier_slam.png)


## References

[1]   Zhou, Bolei, et al. "Learning deep features for scene recognition using places database." Advances in neural 	information processing systems. 2014.

[2]   Wang, Limin, et al. "Places205-vggnet models for scene recognition." arXiv preprint arXiv:1508.01667 (2015).

[3]   Sünderhauf, Niko, et al. "Place categorization and semantic mapping on a mobile robot." Robotics and Automation (ICRA), 2016 IEEE International Conference on. IEEE, 2016.

[4] REMP Dataset, RVL - Purdue: https://engineering.purdue.edu/RVL/Research/PlaceRecognition/REMP_DataSet.tar

[5] Places365 Dataset/ CNN Repository: https://github.com/CSAILVision/places365.git

[6] Khalil Ahmad Yousef, Johnny Park, and Avinash C. Kak, "Place Recognition and Self-Localization in Interior Hallways by Indoor Mobile Robots: A Signature-Based Cascaded Filtering Framework," IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Chicago, September 14-18, 2014

[7] Khalil M. Ahmad Yousef, Johnny Park, Avinash C. Kak, "An Approach-Path Independent Framework for Place Recognition and Mobile Robot Localization in Interior Hallways," 2013 IEEE International Conference on Robotics and Automation, Karlsruhe, May 6-10, 2013

[8] Hyukseong Kwon, Khalil M. Ahmad Yousef, Avinash C. Kak, "Building 3D visual maps of interior space with a new hierarchical sensor fusion architecture," Robotics and Autonomous Systems, ISSN 0921-8890, 10.1016/j.robot.2013.04.016.

