#!/usr/bin/env python

#-------------------------------------------------------------------------------
"""TwoTierClusterer.py: Carries out the secondary classifier for the two-tier classifier
                    """

__author__      = "Ankit Manerikar"
__copyright__   = "Copyright (C) 2017, BME_595-Deep Learning Project"
__date__        = "30th November, 2017"
__credits__     = ["Ankit Manerikar"]
__license__     = "Public Domain"
__version__     = "1.0"
__maintainer__  = "Ankit Manerikar"
__email__       = "amanerik@purdue.edu"
__status__      = "Prototype"
#-------------------------------------------------------------------------------

import numpy as np
import time
import math
import re,os,sys, argparse
import pickle
import matplotlib.pyplot as plt
import torch
import cv2
import sklearn.cluster as cluster

class TwoTierClusterer(object):
    
    parser = argparse.ArgumentParser(description='Dataset Loader')

    def __init__(self):
        self.parser.add_argument('--dataset',
                                 metavar='DIR',
                                 default='../Datasets/RVL_DataSet/',
                                 help='path to Dataset for SLAM')

        self.arg = self.parser.parse_args()

        self.indoor_label_file = os.path.join(self.arg.dataset, 'ResNetLabels.txt')
        self.knn_label_file    = os.path.join(self.arg.dataset, 'KNNLabels.txt')
        self.robot_pose_file   = os.path.join(self.arg.dataset, 'RobotTrajectory.txt')

        self.label_no =[]
        
        with open(self.indoor_label_file, 'r') as in_file:
            resnet_data = in_file.read().splitlines()

            for line in resnet_data:
                ldata = line.replace('\n','').split('\t')
                self.label_no.append(int(ldata[1]))
                
        with open(self.robot_pose_file, 'r') as in_file:
            pose_data = in_file.read().splitlines()

            self.feature_set = []
            for line in pose_data:
                ldata = line.replace('\n','').split(' ')
                self.feature_set.append([  float(ldata[0]),
                                           float(ldata[1])  ]  )           
                
        dbscan_cls = cluster.AgglomerativeClustering(n_clusters=5)

        self.knn_labels = dbscan_cls.fit_predict(np.asarray(self.feature_set),
                                        self.label_no)

        with open(self.knn_label_file, 'w') as knn_file:

            for label in self.knn_labels:
                line = str('%i\n'%label)
                knn_file.write(line)
                
            knn_file.close()

if __name__ == '__main__':
    A = TwoTierClusterer()
