#!/usr/bin/env python

# This code is modified from the pytorch example code:
# https://github.com/pytorch/examples/blob/master/imagenet/main.py
# and the following places-365 code:
# https://github.com/CSAILVision/places365.git
# Bolei Zhou

#-------------------------------------------------------------------------------
"""Places365ResNetTrainer.py: Trains a Place Recognition ResNet based on the
                              Places365 dataset"""

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

# Python utility modules
import argparse
import os
import shutil
import time
import pickle
import matplotlib.pyplot as plt

# Pytorch modules
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# Class for Generating and Updating Training parameters -----------------------

class AverageMeter(object):
    """-----------------------------------------------------------------------
    Desc.:      Computes and stores the average and current value of training
                parameters
    Attributes: -
    Methods:    reset()
    			update()
    -----------------------------------------------------------------------"""
    
    def __init__(self):
        """-------------------------------------------------------------------
        Desc.:      Constructor
        Args:       -
        Returns:    -
        -------------------------------------------------------------------"""

        self.reset()

    def reset(self):
        """-------------------------------------------------------------------
        Desc.:      Reset Values
        Args:       -
        Returns:    -
        -------------------------------------------------------------------"""

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """-------------------------------------------------------------------
        Desc.:      Update Values 
        Args:       -
        Returns:    -
        -------------------------------------------------------------------"""

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
#=============================================================================

# Class for Training a ResNet using the Places365 dataset --------------------

class Places365ResNetTrainer(object):
    """-----------------------------------------------------------------------
    Desc.:      Class for Training the Places365 ResNet for Indoor Scene
                Recognition. The pre-trained ResNet is loaded and the last fully
                connected layer is re-trained for the concise Places dataset
                containing only indoor scenes.
                
    Attributes: -
    Methods:    
    -----------------------------------------------------------------------"""

    parser = argparse.ArgumentParser(description='PyTorch Places365 Training')        
    model_file =            'whole_resnet18_places365.pth.tar'
    indoor_model_file =     'indoor_resnet18_places365.pth.tar'
    file_name =             'categories_places365.txt'
    indoor_file_name =      'indoor_categories_places365.txt'

    def __init__(self):
        """-------------------------------------------------------------------
        Desc.:      Computes and stores the average and current value
        Args:       -
        Returns:    generate_indoor_dataset()
        			create_parser()
        			train_nn()
        			train()
        			vaidate()
        			adjust_learning_rate()
        			save_plots()
        			get_accuracy()
        -------------------------------------------------------------------"""

        self.create_parser()
        self.args = self.parser.parse_args()

        if not os.access(self.model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/whole_resnet18_places365.pth.tar'
            os.system('wget ' + weight_url)

        if not os.access(self.file_name, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            os.system('wget ' + synset_url)

        self.model = torch.load(self.model_file)
        self.model.fc = nn.Linear(512, 50)
        self.generate_indoor_dataset()
        print "Model Loaded"
        
        # Dataset Loaders ----------------------------
        traindir = os.path.join(self.args.data, 'train_indoor')
        valdir = os.path.join(self.args.data, 'val_indoor')
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(traindir, transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.workers, pin_memory=True)

        self.val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=self.args.batch_size, shuffle=False,
            num_workers=self.args.workers, pin_memory=True)
        #---------------------------------------------
        print "Dataset Loaded"
        
        # define Criterion and Optimizer--------------
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD(self.model.fc.parameters(),
                                         lr = self.args.lr,
                                         weight_decay=self.args.weight_decay)
        #---------------------------------------------
        print "Initialized"
#-------------------------------------------------------------------------------------

    def generate_indoor_dataset(self):
        self.category_list = [
        'arch',                 'art_gallery',          'art_school',       'auditorium',          'auto_showroom',
        'balcony-interior',     'bathroom',             'bedroom',          'bowling_alley',        'campus',
        'classroom',            'computer_room',        'conference_center','conference_room',      'corridor',
        'courthouse',           'department_store',     'dining_hall',      'dining_room',          'discotheque',
        'elevator-door',        'elevator_lobby',       'elevator_shaft',   'flea_market-indoor',   'food_court',
        'garage-indoor',        'hangar-indoor',        'home_office',      'ice_skating_rink-indoor', 'lawn',
        'lecture_room',         'library-indoor',       'living_room',      'lobby',                'museum-indoor',
        'office_building',      'office_cubicles',      'pub-indoor',       'reception',            'recreation_room',
        'restaurant_kitchen',   'server_room',          'shopping_mall-indoor', 'stage-indoor',     'staircase',
        'supermarket',          'television_room',      'ticket_booth',     'train_interior',       'utility_room'
        ]
##        self.category_list = [        'arch',                 'art_gallery']

        loaded_train_dir   = os.path.join(self.args.data, 'train')
        indoor_train_dir   = os.path.join(self.args.data, 'train_indoor')
        loaded_val_dir     = os.path.join(self.args.data, 'val')
        indoor_val_dir     = os.path.join(self.args.data, 'val')

        if not os.path.exists(indoor_train_dir):
            os.makedirs(indoor_train_dir)

        if not os.path.exists(indoor_val_dir):
            os.makedirs(indoor_val_dir)

        label_ctr = 0
        for label_name in self.category_list:
            src_file = os.path.join(loaded_train_dir,label_name)
            dst_file = os.path.join(indoor_train_dir,label_name)
            shutil.copytree(src_file,dst_file)

            src_file = os.path.join(loaded_val_dir,label_name)
            dst_file = os.path.join(indoor_val_dir,label_name)
            shutil.copytree(src_file,dst_file)

            with open(self.indoor_file_name,'a') as cat_file:
                line = '/'+list(label_name)[0]+'/'+label_name+str(' %i'%label_ctr) +'\n'
                cat_file.write(line)
                cat_file.close()

            label_ctr += 1
#-------------------------------------------------------------------------------------

    def create_parser(self):
        """-------------------------------------------------------------------
        Desc.:      Computes and stores the average and current value
        Args:       -
        Returns:    -
        -------------------------------------------------------------------"""

        self.parser.add_argument('--data',
                                 metavar='DIR',
                                 default="./places365_standard/",
                                 help='path to dataset')
        self.parser.add_argument('-j', '--workers',
                                 default=2,
                                 type=int,
                                 metavar='N',
                                 help='number of data loading workers (default: 4)')
        self.parser.add_argument('--epochs',
                                 default=1,
                                 type=int,
                                 metavar='N',
                                 help='number of total epochs to run')
        self.parser.add_argument('--start-epoch',
                                 default=0,
                                 type=int,
                                 metavar='N',
                                 help='manual epoch number (useful on restarts)')
        self.parser.add_argument('-b', '--batch-size',
                                 default=10,
                                 type=int,
                                 metavar='N',
                                 help='mini-batch size (default: 50)')
        self.parser.add_argument('--lr', '--learning-rate',
                                 default=0.01,
                                 type=float,
                                 metavar='LR',
                                 help='initial learning rate')
        self.parser.add_argument('--momentum',
                                 default=0.9,
                                 type=float,
                                 metavar='M',
                                 help='momentum')
        self.parser.add_argument('--weight-decay', '--wd',
                                 default=1e-4,
                                 type=float,
                                 metavar='W',
                                 help='weight decay (default: 1e-4)')
        self.parser.add_argument('--print-freq', '-p',
                                 default=10,
                                 type=int,
                                 metavar='N',
                                 help='print frequency (default: 10)')
        self.parser.add_argument('--resume',
                                 default='',
                                 type=str,
                                 metavar='PATH',
                                 help='path to latest checkpoint (default: none)')
        self.parser.add_argument('--num_classes',
                                 default=50,
                                 type=int,
                                 help='num of class in the model')
        self.parser.add_argument('--dataset',
                                 default='places365',
                                 help='which dataset to train')
#-------------------------------------------------------------------------------------

    def train_nn(self):
        """-------------------------------------------------------------------
        Desc.:      trains the NN for each epoch
        Args:       -
        Returns:    -
        -------------------------------------------------------------------"""

        print("Training ResNet on Places365 dataset ...")

        for epoch in range(self.args.epochs):
            print("Epoch No:\t%i"%epoch)
            self.adjust_learning_rate(epoch)
            self.train(epoch)
#-------------------------------------------------------------------------------------
        
    def train(self, epoch):
        """-------------------------------------------------------------------
        Desc.:      Per-epoch training phase of the NN
        Args:       -
        Returns:    -
        -------------------------------------------------------------------"""
        t_loss_val   = []
        t_loss_avg   = []
        t_prec1_val   = []
        t_prec1_avg   = []
        t_prec5_val   = []
        t_prec5_avg   = []

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        self.model.train()

        end = time.time()
        for i, (input, target) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input_var   = torch.autograd.Variable(input)
            target_var  = torch.autograd.Variable(target)

            # compute output
            output  = self.model(input_var)
            loss    = self.criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = self.get_accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(self.train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

            t_loss_val.append(losses.val)
            t_loss_avg.append(losses.avg)
            t_prec1_val.append(top1.val)
            t_prec1_avg.append(top1.avg)
            t_prec5_val.append(top5.val)
            t_prec5_avg.append(top5.avg)

        self.tparam_dict = dict()
        self.tparam_dict['train_losses_avg'] = t_loss_avg  
        self.tparam_dict['train_losses_val'] = t_loss_val  
        self.tparam_dict['train_prec1_avg']  = t_prec1_avg  
        self.tparam_dict['train_prec1_val']  = t_prec1_val  
        self.tparam_dict['train_prec5_avg']  = t_prec5_avg  
        self.tparam_dict['train_prec5_val']  = t_prec5_val  

##        param_dict['val_losses_avg']  = v_loss_avg  
##        param_dict['val_losses_val']  = v_loss_val  
##        param_dict['val_prec1_avg']   = v_prec1_avg  
##        param_dict['val_prec1_val']   = v_prec1_val  
##        param_dict['val_prec5_avg']   = v_prec5_avg  
##        param_dict['val_prec5_val']   = v_prec5_val  

        self.t_param_file = 'train_param_dict.pyc'

        with open(self.t_param_file,'w') as curr_file:
            pickle.dump(self.tparam_dict,curr_file)
            curr_file.close()

        print "Training Parameters Saved ..."
        torch.save(self.model, self.indoor_model_file)
        
#-------------------------------------------------------------------------------------
    def validate(self):
        v_loss_val   = []
        v_loss_avg   = []
        v_prec1_val   = []
        v_prec1_avg   = []
        v_prec5_val   = []
        v_prec5_avg   = []
        
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        for i, (input, target) in enumerate(self.val_loader):
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = self.get_accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(self.val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

            v_loss_val.append(losses.val)
            v_loss_avg.append(losses.avg)
            v_prec1_val.append(top1.val)
            v_prec1_avg.append(top1.avg)
            v_prec5_val.append(top5.val)
            v_prec5_avg.append(top5.avg)

        self.vparam_dict = dict()
        self.vparam_dict['val_losses_avg']  = v_loss_avg  
        self.vparam_dict['val_losses_val']  = v_loss_val  
        self.vparam_dict['val_prec1_avg']   = v_prec1_avg  
        self.vparam_dict['val_prec1_val']   = v_prec1_val  
        self.vparam_dict['val_prec5_avg']   = v_prec5_avg  
        self.vparam_dict['val_prec5_val']   = v_prec5_val 

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        self.v_param_file = 'val_param_dict.pyc'

        with open(self.v_param_file,'w') as curr_file:
            pickle.dump(self.vparam_dict,curr_file)
            curr_file.close()
        
        return top5.avg


    def adjust_learning_rate(self, epoch):
        """-------------------------------------------------------------------
        Desc.:      Computes and stores the average and current value
        Args:       -
        Returns:    -
        -------------------------------------------------------------------"""

        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.args.lr * (0.1 ** (epoch // 30))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
#-------------------------------------------------------------------------------------

    def get_accuracy(self, output, target, topk=(1,)):
        """-------------------------------------------------------------------
        Desc.:      Computes and stores the average and current value
        Args:       -
        Returns:    -
        -------------------------------------------------------------------"""

        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
#-------------------------------------------------------------------------------------
            
    def save_plots(self):
        if os.path.exists('./Figures/'):
            shutil.rmtree('./Figures/')
            os.makedir("./Figures/")

        plt.figure(1)
        plt.title("Places Indoor_ResNet: Number of Batches v/s Losses")
        plt.xlabel("Number of Batches")
        plt.ylabel("Losses")
        plt.plot(self.tparam_dict["train_losses_avg"], 'b', label=" Batchwise Loss")
        plt.plot(self.tparam_dict["train_losses_val"], 'r', linestyle = ':', linewidth = 0.1, label="Average Loss")
        plt.grid()
        plt.legend()
        plt.savefig("./Figures/Training_losses.png")

        plt.figure(2)
        plt.title("Places Indoor_ResNet: Number of Batches v/s Accuracy")
        plt.xlabel("Number of Batches")
        plt.ylabel("Accuracy")
        plt.plot(self.tparam_dict["train_prec1_avg"], 'b', label=" Batchwise Accuracy")
        plt.plot(self.tparam_dict["train_prec1_val"], 'r', linestyle = ':', linewidth = 0.1, label="Average Accuracy")
        plt.grid()
        plt.legend()
        plt.savefig("./Figures/Training_Top1_Accuracy.png")

        plt.figure(3)
        plt.title("Places Indoor_ResNet: Number of Batches v/s Accuracy")
        plt.xlabel("Number of Batches")
        plt.ylabel("Accuracy")
        plt.plot(self.tparam_dict["train_prec5_avg"], 'b', label=" Batchwise Accuracy")
        plt.plot(self.tparam_dict["train_prec5_val"], 'r', linestyle = ':', linewidth = 0.1, label="Average Accuracy")
        plt.grid()
        plt.legend()
        plt.savefig("./Figures/Training_Top5_Accuracy.png")
        

#-------------------------------------------------------------------------------------
## Classes End
#=====================================================================================

if __name__ == '__main__':
    Trainer = Places365ResNetTrainer()
    Trainer.train_nn()
    print "Validation Accuracy:", Trainer.validate()


