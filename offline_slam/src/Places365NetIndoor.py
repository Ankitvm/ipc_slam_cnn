#!/usr/bin/env python

#-------------------------------------------------------------------------------
"""Places365Net.py: Creates a Place Recognition ResNet based on the Places365
                    dataset"""

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
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
from ResNetTrainer import Places365ResNetTrainer
import rospkg

class Places365ResNet(object):
    rospack = rospkg.RosPack()
    curr_dir = rospack.get_path('offline_slam')

    arch_name = 'resnet18'
    model_file = os.path.join(curr_dir,'src/whole_resnet18_places365.pth.tar')
    indoor_model_file = os.path.join(curr_dir,'src/indoor_resnet18_places365.pth.tar')
    file_name = os.path.join(curr_dir,'src/categories_places365.txt')
    indoor_file_name = os.path.join(curr_dir,'src/indoor_categories_places365.txt')

#-------------------------------------------------------------------------------
    def __init__(self):
        if not os.access(self.model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/whole_%s_places365.pth.tar' % self.arch_name
            os.system('wget ' + weight_url)

        useGPU = 0
        if useGPU == 1:
            self.model = torch.load(self.model_file)
        else:
            self.model = torch.load(self.model_file,
                                    map_location=lambda storage,
                                    loc: storage) 

#        self.model.eval()

        if not os.path.exists('./'+self.indoor_model_file):
            print "Indoor Places ResNet classifier not found ..."
            print "Training classifier from Places365 ResNet"
            Trainer = Places365ResNetTrainer()
            Trainer.train_nn()
            Trainer.save_plots()

        self.indoor_model = torch.load(self.indoor_model_file)

        self.centre_crop = trn.Compose([ trn.Scale(256),
                                    trn.CenterCrop(224),
                                    trn.ToTensor(),
                                    trn.Normalize([0.485, 0.456, 0.406],
                                                  [0.229, 0.224, 0.225])
                                  ])

        if not os.access(self.file_name, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            os.system('wget ' + synset_url)

        if not os.path.exists('./'+self.indoor_model_file):
            print "Indoor Places ResNet classifier not found ..."
            print "Training classifier from Places365 ResNet"
        
        self.classes = list()
        self.indoor_classes = list()

        with open(self.file_name) as class_file:
            for line in class_file:
                self.classes.append(line.strip().split(' ')[0][3:])
        self.classes    = tuple(self.classes)

        with open(self.indoor_file_name) as class_file:
            for line in class_file:
                self.indoor_classes.append(line.strip().split(' ')[0][3:])
        self.indoor_classes    = tuple(self.indoor_classes)

#-------------------------------------------------------------------------------
    
    def test_loaded(self):
        img_name = str('13.jpg')
        self.img_url = 'http://places.csail.mit.edu/demo/' + img_name
        os.system('wget ' + self.img_url)
        img = Image.open(img_name)
        input_image = V(self.centre_crop(img).unsqueeze(0),
                      volatile=True)
        a,b,c = self.forward_loaded(img_name)
        return a,b,c

#-------------------------------------------------------------------------------

    def test_indoor(self):
        img_name = str('13.jpg')
        self.img_url = 'http://places.csail.mit.edu/demo/' + img_name
        os.system('wget ' + self.img_url)
        img = Image.open(img_name)
        input_image = V(self.centre_crop(img).unsqueeze(0),
                      volatile=True)
        a,b,c = self.forward_indoor(img_name)
        return a,b,c
#-------------------------------------------------------------------------------

    def get_resnet_input(self, image_name):
        img = Image.open(image_name)
        return V(self.centre_crop(img).unsqueeze(0),
                      volatile=True)        
#-------------------------------------------------------------------------------

    def normalize_loaded_label(self, val):
        return np.uint32(val*(255.0/364))

    def normalize_indoor_label(self, val):
        return np.uint32(val*(255.0/50))

#-------------------------------------------------------------------------------
    
    def forward_loaded(self,image_file):
        input_image = self.get_resnet_input(image_file)
        logit = self.model.forward(input_image)
        h_x = F.softmax(logit).data.squeeze()
        probs, idx = h_x.sort(0, True)

        print('{:.3f} -> {} {}'.format(probs[0], idx[0],self.classes[idx[0]]))        
        return probs[0], self.normalize_loaded_label(idx[0]), self.classes[idx[0]]

    def forward_indoor(self,image_file):
        input_image = self.get_resnet_input(image_file)
        logit = self.indoor_model.forward(input_image)
        h_x = F.softmax(logit).data.squeeze()
        probs, idx = h_x.sort(0, True)

        print('{:.3f} -> {} {}'.format(probs[0], idx[0],self.indoor_classes[idx[0]]))
        return probs[0], self.normalize_indoor_label(idx[0]), self.indoor_classes[idx[0]]

#-------------------------------------------------------------------------------
## Class Ends
#===============================================================================

if __name__=="__main__":

    A = Places365ResNet()
    prob, label_no, label_name = A.test_indoor()
    print "Label No:\t", label_no
    print "Label Name:\t",label_name
#===============================================================================
