
from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from data_loader import SpeechDataLoader
import numpy as np

from model import LeNet, VGG, CNNRNN


import model as model
from train import train, test, val
import os


# Training settings
parser = argparse.ArgumentParser(
    description='ConvNets for Speech Commands Recognition')
parser.add_argument('--train_path', default='/train',
                    help='path to the train data folder')
parser.add_argument('--test_path', default='/test',
                    help='path to the test data folder')
parser.add_argument('--valid_path', default='/valid',
                    help='path to the valid data folder')
parser.add_argument('--batch_size', type=int, default=100,
                    metavar='N', help='training and valid batch size')
parser.add_argument('--test_batch_size', type=int, default=100,
                    metavar='N', help='batch size for testing')
parser.add_argument('--arc', default='LeNet',
                    help='network architecture: LeNet, VGG11, VGG13, VGG16, VGG19, ResNet18, ResNet34, CNNRNN')
parser.add_argument('--input_format', default='STFT',
                    help='Input format: STFT, MEL100, MEL32, MEL40, MEL128, MEL64')
parser.add_argument('--epochs', type=int, default=100,
                    metavar='N', help='number of epochs to train')
parser.add_argument('--max_len', type=int, default=101,
                    help='max length of spectrogram')
parser.add_argument('--lr', type=float, default=0.001,
                    metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    metavar='M', help='SGD momentum, for SGD only')
parser.add_argument('--optimizer', default='adam',
                    help='optimization method: sgd | adam')
parser.add_argument('--cuda', default=True, help='enable CUDA')
parser.add_argument('--seed', type=int, default=1234,
                    metavar='S', help='random seed')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='num of batches to wait until logging train status')

parser.add_argument('--patience', type=int, default=5, metavar='N',
                    help='how many epochs of no loss improvement should we wait before stop training')
parser.add_argument('--loss_func', default='NLL',
                    help='NLL, CrossEntropy')
# feature extraction options
parser.add_argument('--window_size', default=.02,
                    help='window size for the stft')
parser.add_argument('--window_stride', default=.01,
                    help='window stride for the stft')
parser.add_argument('--window_type', default='hamming',
                    help='window type for the stft')
parser.add_argument('--normalize', default=True,
                    help='boolean, wheather or not to normalize the spect')
parser.add_argument('--datacleaning', default=False,
                    help='boolean, removes the data with negligible values')

args = parser.parse_args()
print(args)
args.cuda = args.cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# loading data

train_dataset = SpeechDataLoader(args.train_path, args.input_format, window_size=args.window_size, window_stride=args.window_stride,
                               window_type=args.window_type, normalize=args.normalize, max_len=args.max_len, clean_data=args.datacleaning)


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=20, pin_memory=args.cuda, sampler=None)


valid_dataset = SpeechDataLoader(args.valid_path, args.input_format, window_size=args.window_size, window_stride=args.window_stride,
                               window_type=args.window_type, normalize=args.normalize, max_len=args.max_len, clean_data=args.datacleaning)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=args.batch_size, shuffle=None,
    num_workers=20, pin_memory=args.cuda, sampler=None)


test_dataset = SpeechDataLoader(args.test_path,args.input_format, window_size=args.window_size, window_stride=args.window_stride,
                              window_type=args.window_type, normalize=args.normalize, max_len=args.max_len, clean_data=args.datacleaning)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.test_batch_size, shuffle=None,
    num_workers=20, pin_memory=args.cuda, sampler=None)

# build model
if args.arc == 'LeNet':
  
    if args.datacleaning:
        if(args.input_format=='STFT'):
            model = LeNet(10360)
        elif(args.input_format=='MEL40'):
            model = LeNet(1960)
        elif(args.input_format=='MEL100'):
            model = LeNet(6160)
    else:

        if(args.input_format=='STFT'):
            model = LeNet(16280)
        elif(args.input_format=='MEL40'):
            model = LeNet(3080)
        elif(args.input_format=='MEL100'):
            model = LeNet(9680)


elif args.arc.startswith('VGG'):
    # only using STFT
    if args.datacleaning:
        model = VGG(args.arc, 5120)
    else:
        model = VGG(args.arc, 7680)

elif args.arc.startswith('ResNet'):
    

    if args.datacleaning:
        if(args.input_format=='MEL32'):
            model = model.create_resnet_model(model_name=args.arc,num_classes=30, in_channels=1, last_layer_dim=1536)
        else:
            model = model.create_resnet_model(model_name=args.arc,num_classes=30, in_channels=1, last_layer_dim=3072)

    else:

        if(args.input_format=='MEL32'):
            model = model.create_resnet_model(model_name=args.arc,num_classes=30, in_channels=1, last_layer_dim=2048)
        else:
            model = model.create_resnet_model(model_name=args.arc,num_classes=30, in_channels=1, last_layer_dim=4096)


else:
    
    if args.datacleaning:
        if(args.input_format=='STFT'):
            model = LeNet(10360)
        elif(args.input_format=='MEL40'):
            model = LeNet(1960)
        elif(args.input_format=='MEL100'):
            model = LeNet(6160)
    else:
        if(args.input_format=='STFT'):
            model = LeNet(16280)
        elif(args.input_format=='MEL40'):
            model = LeNet(3080)
        elif(args.input_format=='MEL100'):
            model = LeNet(9680)



if args.cuda:
    print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model).cuda()

# define optimizer
if args.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer.lower() == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)

else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)

best_valid_loss = np.inf
iteration = 0
epoch = 1
itr=0


# training with early stopping
while (epoch < args.epochs + 1) and (iteration < args.patience) and (itr<args.patience):
    train(train_loader, model, optimizer, epoch, args.cuda, args.log_interval, args.loss_func)
    valid_loss = val(valid_loader, model, args.cuda,args.loss_func)
    if valid_loss > best_valid_loss:
        iteration += 1
        print('Loss was not improved, iteration {0}'.format(str(iteration)))
        if(iteration==1):
            itr+=1
    else:
        print('Saving model...')
        iteration = 0
        best_valid_loss = valid_loss
        state = {
            'net': model.module if args.cuda else model,
            'acc': valid_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
    epoch += 1

# test model
test(test_loader, model, args.cuda, args.loss_func)
