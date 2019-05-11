from __future__ import print_function
import torch.nn.functional as F
from torch.autograd import Variable

import pickle

import torch 
def train(loader, model, optimizer, epoch, cuda, log_interval, loss_func, verbose=True):
    """ Train the model """
    model.train()
    global_epoch_loss = 0
    
    if(loss_func=='CrossEntropy'):
        criterion = torch.nn.CrossEntropyLoss()
        criterion2=torch.nn.CrossEntropyLoss(reduction='sum')
    if(loss_func=='NLL'):
        criterion = torch.nn.NLLLoss()
        criterion2 = torch.nn.NLLLoss(reduction='sum')

    for batch_idx, (data, target) in enumerate(loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss=criterion(output,target)

        loss.backward()
        optimizer.step()
        global_epoch_loss += criterion2(output, target).data
        if verbose:
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(loader.dataset), 100.
                    * batch_idx / len(loader), loss.data))
    global_epoch_loss=global_epoch_loss / len(loader.dataset)
    print('\nTrain set: Average loss: {:.4f}\n'.format(
            global_epoch_loss))
    return global_epoch_loss


def test(loader, model, cuda,loss_func, verbose=True):
    """Test the trained model """
    model.eval()
    test_loss = 0
    correct = 0
    prediction = []
    actual = []
    
    if(loss_func=='CrossEntropy'):
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    if(loss_func=='NLL'):
        criterion = torch.nn.NLLLoss(reduction='sum')

    for data, target in loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        test_loss += criterion(output, target).data  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability

        prediction.extend([element.item() for element in pred.flatten()])
        actual.extend([element.item() for element in target.data.view_as(pred).flatten()])

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(loader.dataset)
    
    with open('prediction.pickle','wb') as f:
        pickle.dump(prediction,f)
        
    with open('actual.pickle','wb') as f:
        pickle.dump(actual, f)

    precision, recall = calculate_precision_recall(prediction, actual)
    print("Precision = ",prec)
    print("Recall = ", rec)

    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))
    return test_loss

def val(loader, model, cuda, loss_func,verbose=True):
    """Validate the trained model """
    model.eval()
    test_loss = 0
    correct = 0
    if(loss_func=='CrossEntropy'):
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    if(loss_func=='NLL'):
        criterion = torch.nn.NLLLoss(reduction='sum')
    
    for data, target in loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        test_loss += criterion(output, target).data  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(loader.dataset)
    if verbose:
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))
    return test_loss

def calculate_precision_recall(prediction, actual):
    """ Calculate the precision/recall of the model """
    # Number of classes = 30
    tp = [0] * 30  # True positives
    fp = [0] * 30  # True negatives
    actual_count = [0] * 30
    precision = [0] * 30
    recall = [0] * 30
    
    for p,a in zip(prediction,actual):
        if p == a:
            tp[p] += 1
        else:
            fp[p]+= 1

    for a in actual:
        actual_count[a] += 1

    print("tp = ", tp)
    print("fp = ", fp)
    print("actual_count = ", actual_count)
    for i in range(30):
        if tp[i] != 0 or fp[i] != 0:
            precision[i] = tp[i]/(tp[i] + fp[i])
        recall[i] = tp[i]/actual_count[i]

    return precision, recall


