from __future__ import print_function
import torch.nn.functional as F
from torch.autograd import Variable

import torch 
def trainp(loader, model, optimizer, epoch, cuda, log_interval, loss_func, verbose=True):
    model.train()
    global_epoch_loss = 0
    
    if(loss_func=='CrossEntropy'):
        criterion = torch.nn.CrossEntropyLoss()
        criterion2=torch.nn.CrossEntropyLoss(reduction='sum')
    if(loss_func=='NLL'):
        criterion = torch.nn.NLLLoss()
        criterion2 = torch.nn.NLLLoss(reduction='sum')



    for batch_idx, data  in enumerate(loader):
        
        data1, data2, target = data[0], data[1], data[2]
        if cuda:
            data1, data2, target = data1.cuda(), data2.cuda(), target.cuda()
        data1, data2, target = Variable(data1), Variable(data2), Variable(target)

        optimizer.zero_grad()
        output = model(data1, data2)
        loss=criterion(output,target)

        loss.backward()
        optimizer.step()
        global_epoch_loss += criterion2(output, target).data
        if verbose:
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data1), len(loader.dataset), 100.
                    * batch_idx / len(loader), loss.data))
    global_epoch_loss=global_epoch_loss / len(loader.dataset)
    print('\nTrain set: Average loss: {:.4f}\n'.format(
            global_epoch_loss))
    return global_epoch_loss


def testp(loader, model, cuda,loss_func, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0
    if(loss_func=='CrossEntropy'):
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    if(loss_func=='NLL'):
        criterion = torch.nn.NLLLoss(reduction='sum')

    for data1, data2, target in loader:
        if cuda:
            data1, data2, target = data1.cuda(), data2.cuda(), target.cuda()
        data1, data2, target = Variable(data1, volatile=True), Variable(data2, volatile=True), Variable(target)
        output = model(data1, data2)
        test_loss += criterion(output, target).data  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(loader.dataset)
    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))
    return test_loss

def valp(loader, model, cuda, loss_func,verbose=True):
    model.eval()
    test_loss = 0
    correct = 0
    if(loss_func=='CrossEntropy'):
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    if(loss_func=='NLL'):
        criterion = torch.nn.NLLLoss(reduction='sum')
    
    for data1, data2, target in loader:
        print(data1.shape, data2.shape)
        if cuda:
            data1, target = data1.cuda(), target.cuda()
        data1, target = Variable(data1, volatile=True), Variable(target)
        output = model(data1, data2)

        
        

        test_loss += criterion(output, target).data  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(loader.dataset)
    if verbose:
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))
    return test_loss
