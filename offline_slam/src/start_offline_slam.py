#!/usr/bin/env python

#-------------------------------------------------------------------------------
"""start_offline_slam.py: Reads vision/LiDAR data from the dataset and establishes
                          the related topics - the main ROS node to run in order to
                          execute code for VPC with SLAM"""

__author__      = "Ankit Manerikar"
__copyright__   = "Copyright (C) 2017, BME_595 - Deep Learning Project"
__date__        = "5th December, 2017"
__credits__     = ["Ankit Manerikar"]
__license__     = "Public Domain"
__version__     = "1.0"
__maintainer__  = "Ankit Manerikar"
__email__       = "amanerik@purdue.edu"
__status__      = "Prototype"
#-------------------------------------------------------------------------------

# Basic Python Modules 
import numpy as np
import time
import math
import re,os,sys, argparse
import pickle
import matplotlib.pyplot as plt
import torch
import cv2

from shutil                 import move,copy
from mpl_toolkits.mplot3d   import Axes3D

# ROS Python Modules
import rospy,rospkg
import tf
from cv_bridge import CvBridge, CvBridgeError

from std_msgs.msg      import String, Header
from nav_msgs.msg      import Odometry
from sensor_msgs.msg   import PointCloud2, PointField, Image
from sensor_msgs.msg   import PointCloud
from geometry_msgs.msg import Point, Pose, Quaternion, Twist
import sensor_msgs.point_cloud2 as pc2

from Places365NetIndoor import Places365ResNet

class OfflineSLAMClass(object):
    """-------------------------------------------------------------------------
    Desc.:      Class for carrying out SLAM along with Visual Place Categorization
                (VPC) - the code extracts Images, SICK point Clouds and Robotr pose
                estimates from the RVL datasets - to execute code, run the following
                command:
                $ rosrun offline_slam start_offline_slam.py --<arg> <val>

                --dataset   path to dataset
                --nn        model for VPC - 'places365', 'indoor_net', 'two-tier'

                For complete visualization, run the launch file:
                $ roslaunch offline_slam start_offline_slam.launch --<arg>:=<val> 

                (The launch file contains the same arguments as start_offline_slam.py
                )
                
    Attr.:      
    Methods:    extract_range_data()        - extracts offline range data
                extract_image_data()        - extracts offline image data
                extract_robot_pose()        - extracts offline pose data
                generate_frame_data()       - generate tf framesfor rviz visualization
                start_slam_data_publisher() - starts publishing offline data to ROS topics
    -------------------------------------------------------------------------"""

# Data attributes for the OfflineSLAMInit class --------------------------------    

    parser = argparse.ArgumentParser(description='Dataset Loader')
    file_loc = dict()
    ResNetP365 = Places365ResNet()

# Methods for the OfflineSLAMInit class ----------------------------------------

    def __init__(self):
        """---------------------------------------------------------------------
        Desc.:      Constructor for Offline SLAM 
        Args:       -
        Returns:    -
        ---------------------------------------------------------------------"""

        print '===================================================================='
        print 'SLAM Node for Publishing Data Offline'
        print '====================================================================\n'

        print 'Program Start Time : ', time.strftime('%d-%m-%Y %H:%M:%S', time.localtime())

        print "Initializing SLAM Node ..."
        
        self.parser.add_argument('--dataset',
                                 metavar='DIR',
                                 default='/home/ankit-rvl/Datassets/REMP_DataSet/',
                                 help='path to Dataset for SLAM')
        self.parser.add_argument('--__name:=',
                                 metavar='DIR',
                                 default='offline_slam_node',
                                 help='Name of the ROS Node')
        self.parser.add_argument('--__log:=',
                                 metavar='DIR',
                                 default='./check.log',
                                 help='Log file for ROS Node')        
        self.parser.add_argument('--nn',
                                 type=str,
                                 default='places365',
                                 help='Choose the net to use: places365, indoor_net or two-tier')
        self.arg = self.parser.parse_args()

        rospack = rospkg.RosPack()
        curr_dir = rospack.get_path('offline_slam')
        data_dir =curr_dir +'/Datasets/'

        if not os.path.exists(self.arg.dataset):
            os.system("tar xf REMP_DataSet.tar")
            os.system("mkdir "+data_dir)
            os.system("sudo mv REMP_DataSet " + data_dir)            
            self.arg.dataset = data_dir+"/REMP_DataSet/"            
            
        self.file_loc['range']  = self.arg.dataset+'/LaserScans/'
        self.file_loc['images'] = self.arg.dataset+'/StereoImages/'
        self.file_loc['pose']   = self.arg.dataset+'/RobotEncoder/'
        self.file_loc['traj']   = self.arg.dataset+'/RobotTrajectory.txt'

        self.file_loc['indoor_net'] = self.arg.dataset+'/ResNetLabels.txt'
        self.file_loc['two_tier']   = self.arg.dataset+'/KNNLabels.txt'

##        if os.path.exists(self.file_loc['indoor_net']):
##            os.remove(self.file_loc['indoor_net'])
##        if os.path.exists(self.file_loc['two_tier']):
##            os.remove(self.file_loc['two_tier'])
            
        self.data_cnt = len(os.listdir(self.file_loc['range']))
        self.frame_tx = tf.TransformBroadcaster()
        
        print "Loading Robot Trajectory ..."

        with open(self.file_loc['traj'],'r') as traj_file:
            traj_rdata = traj_file.read().splitlines()

            self.traj_data = []

            for rline in traj_rdata:
                line_data = rline.replace('\n','').split(' ')
                line = [float(line_data[0]),
                        float(line_data[1]),
                        float(line_data[2])]
                self.traj_data.append(line)
        if self.arg.dataset == 'two-tier':
            with open(self.file_loc['two_tier'], 'r') as knn_file:
                knn_data = knn_file.read().splitlines()

                self.knn_data = []
                for val in knn_data:
                    val = val.replace('\n','')
                    self.knn_data.append(int(val))
                knn_file.close()

        self.knn_ctr = 0
        
        print "\nDataset: ", self.arg.dataset
        print "Range Data Folder: ", self.file_loc['range']
        print "Image Data Folder: ", self.file_loc['images']
        print "Robot Pose Folder: ", self.file_loc['pose']
        print "Dataset Count: ", self.data_cnt
        print "---------------------------------------\n"
        
#------------------------------------------------------------------------------

    def extract_range_data(self, file_no):
        """---------------------------------------------------------------------
        Desc.:      Extract Range Data from the Dataset (in RangeData folder) and
                    create a ROS PointCloud2 message
        Args:       file_no - file for nth reading
        Returns:    -
        ---------------------------------------------------------------------"""        
        curr_file = self.file_loc['range'] + \
                    str('DetectedPoints{:02d}.txt'.format(file_no))

  #      print "Loading Range Data ..."
        with open(curr_file, 'r') as laser_scan_file:
            laser_scan_data = laser_scan_file.read().splitlines()
            point_cloud = []
            
            for currline in laser_scan_data[1:]:
                curr_data = currline.split(' ')
                curr_point = np.array([float(curr_data[0])/1000,
                                       float(curr_data[1])/1000,
                                       0.177,
                                       self.scene_id])
                point_cloud.append(curr_point)

            HEADER = Header(frame_id='sick_point_cloud')

            FIELDS = [
                PointField(name='x',   offset=0,
                           datatype=PointField.FLOAT32, count=1),
                PointField(name='y',   offset=4,
                           datatype=PointField.FLOAT32, count=1),
                PointField(name='z',   offset=8,
                           datatype=PointField.FLOAT32, count=1),
                PointField(name='int', offset=12,
                           datatype=PointField.UINT32, count=1),]

            POINTS = point_cloud
                        
            self.SickPtCloud         = pc2.create_cloud(HEADER,
                                                        FIELDS,
                                                        POINTS)

 #           print "Number of Points Loaded: ", len(point_cloud)
#------------------------------------------------------------------------------

    def extract_image_data(self, file_no):
        """---------------------------------------------------------------------
        Desc.:      Extract Images from the Dataset (in StereoImages folder) and
                    create two ROS Image topics
        Args:       file_no - file for nth reading
        Returns:    -
        ---------------------------------------------------------------------"""
#        print "Loading Images ..."
        l_img_file   = self.file_loc['images'] + str('imgL{:02d}.jpg'.format(file_no))
        r_img_file   = self.file_loc['images'] + str('imgR{:02d}.jpg'.format(file_no))

        try:
            if   self.arg.nn == 'places365':
                l_prob, l_id, l_label = self.ResNetP365.forward_loaded(l_img_file)
                r_prob, r_id, r_label = self.ResNetP365.forward_loaded(r_img_file)
            elif self.arg.nn == 'indoor_net':
                l_prob, l_id, l_label = self.ResNetP365.forward_indoor(l_img_file)
                r_prob, r_id, r_label = self.ResNetP365.forward_indoor(r_img_file)
            elif self.arg.nn == 'two-tier':
                l_prob = 1
                r_prob = 0
                l_label = ' '
                l_id = self.knn_data[self.knn_ctr]
                print l_id
                self.knn_ctr +=1

            if l_prob >= r_prob:
                self.scene_label = l_label
                self.scene_id    = l_id
            else:
                self.scene_label = r_label
                self.scene_id    = r_id
                
        except:
            print "Bad Image Data!"

        with open(self.file_loc['indoor_net'], 'a') as resnet_file:
            line = self.scene_label + str('\t%i\n'%self.scene_id)
            resnet_file.write(line)
            resnet_file.close()

        l_image = cv2.imread(l_img_file,1)
        r_image = cv2.imread(r_img_file,1)
        
        cv2.putText(l_image,self.scene_label,(10,400), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),10,cv2.LINE_AA)
        cv2.putText(r_image,self.scene_label,(10,400), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),10,cv2.LINE_AA)
        
        bridge = CvBridge()
        self.left_image   = bridge.cv2_to_imgmsg(l_image , "bgr8")
        self.right_image  = bridge.cv2_to_imgmsg(r_image , "bgr8")
#------------------------------------------------------------------------------        
        
    def extract_robot_pose(self,file_no):
        """---------------------------------------------------------------------
        Desc.:      Extract Robot Pose from the Dataset (in RobotEncoder folder)
                    and create a ROS Odometry message
        Args:       file_no - file for nth reading
        Returns:    -
        ---------------------------------------------------------------------"""
#        print "Loading Pose ..."
        curr_file = self.file_loc['pose'] + \
                    str('RobotPosition{:02d}.txt'.format(file_no))        

##        with open(curr_file, 'r') as pose_file:
##            curr_data = pose_file.read().replace('\n','').split(' ')
##            self.x  = float(curr_data[0])/1000
##            self.y  = float(curr_data[1])/1000
##            self.th = float(curr_data[2])

        self.x  = self.traj_data[file_no][0]/1000            
        self.y  = self.traj_data[file_no][1]/1000            
        self.th = self.traj_data[file_no][2]*(np.pi/180)            

        odom_quat = tf.transformations.quaternion_from_euler(0, 0, self.th)
        self.odom = Odometry()
        self.odom.header.stamp = self.current_time
        self.odom.header.frame_id = "odom"
        self.odom.pose.pose = Pose(Point(self.x, self.y, 0.), Quaternion(*odom_quat))        
#------------------------------------------------------------------------------

    def generate_frame_data(self):
        """---------------------------------------------------------------------
        Desc.:      Extract and Publish Dataset Data
        Args:       -
        Returns:    -
        ---------------------------------------------------------------------"""
##        print "\nGenerating frame data ..."
        
        self.frame_tx.sendTransform((self.x, self.y,0),
                     tf.transformations.quaternion_from_euler(0, 0,self.th),
                     rospy.Time.now(),
                     "base_link",
                     "map")
##        
##        self.frame_tx.sendTransform((0,-0.225, 0),
##                     tf.transformations.quaternion_from_euler(0, 0, np.pi/2),
##                     rospy.Time.now(),
##                     "base_link",
##                     "sick_point_cloud")
##
##        self.frame_tx.sendTransform((0, 0, 0.177),
##                     tf.transformations.quaternion_from_euler(0, 0, 0),
##                     rospy.Time.now(),
##                     "base_link",
##                     "odom")       
        
#------------------------------------------------------------------------------
        
    def start_slam_data_publisher(self):
        """---------------------------------------------------------------------
        Desc.:      Extract and Publish Dataset Data
        Args:       -
        Returns:    -
        ---------------------------------------------------------------------"""
        
        print "\nPublishing Data Offline from the Dataset", self.arg.dataset
        
        self.point_cloud_publisher = rospy.Publisher('sick_point_cloud',
                                                     PointCloud2,
                                                     queue_size=10)
        self.image_1_publisher     = rospy.Publisher('left_image' ,
                                                     Image,
                                                     queue_size=10)
        self.image_2_publisher     = rospy.Publisher('right_image',
                                                     Image,
                                                     queue_size=10)
        self.odom_publisher        = rospy.Publisher('odom',
                                                     Odometry,
                                                     queue_size=10)
        
        self.odom_broadcaster      = tf.TransformBroadcaster()
        
        rospy.init_node('offline_slam_node', anonymous=True)
        rate = rospy.Rate(10) 

        for file_no in range(self.data_cnt) :
            if not rospy.is_shutdown():
##                print "Reading No: ", file_no
                self.current_time = rospy.Time.now()

                self.extract_image_data(file_no)
                self.extract_range_data(file_no)
                self.extract_robot_pose(file_no)
                
                self.point_cloud_publisher.publish(self.SickPtCloud)
                self.image_1_publisher.publish(self.left_image)
                self.image_2_publisher.publish(self.right_image)
                self.odom_publisher.publish(self.odom)
                self.generate_frame_data()
                rate.sleep()
#------------------------------------------------------------------------------
## Class Ends            
#==============================================================================
        
if __name__ == "__main__":
    OfflineSLAMNode = OfflineSLAMClass()
    OfflineSLAMNode.start_slam_data_publisher()
