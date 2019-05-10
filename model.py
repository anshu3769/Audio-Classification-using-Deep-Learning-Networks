import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

def create_resnet_model(model_name, num_classes, in_channels, last_layer_dim):
    if model_name == "ResNet18":
        #print("Create RESNET18 model")
        model = resnet18(num_classes=num_classes, in_channels=in_channels, last_layer_dim=last_layer_dim)
    if model_name == "ResNet34":
        model = resnet34(num_classes=num_classes, in_channels=in_channels, last_layer_dim=last_layer_dim)
    elif model_name == "ResNet50":
        model = resnet50(num_classes=num_classes, in_channels=in_channels, last_layer_dim=last_layer_dim)
    elif model_name == "ResNet101":
        model = resnet101(num_classes=num_classes, in_channels=in_channels, last_layer_dim=last_layer_dim)
    elif model_name == "ResNet152":
        model = resnet152(num_classes=num_classes, in_channels=in_channels, last_layer_dim=last_layer_dim)
    else:
        model = resnet18(num_classes=num_classes, in_channels=in_channels, last_layer_dim=last_layer_dim)
    return model


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        #print("BasicBlock x is ", x.size())
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=30, in_channels=3, last_layer_dim=2048):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.fc = nn.Linear(last_layer_dim * block.expansion, 512)
        self.fc1=nn.Linear(512,num_classes)
        #print("RESNET CONST expansion = ", block.expansion)
        #print("RESNET CONST num_classes = ", num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        #print("RESNET FORWARD x", x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        #print("RESNET FORWARD x before fc ", x.size())
        x = self.fc(x)
        x = self.fc1(x)
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

def _make_layers(cfg):
    layers = []
    in_channels = 1
    for x in cfg:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                       nn.BatchNorm2d(x),
                       nn.ReLU(inplace=True)]
            in_channels = x
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return nn.Sequential(*layers)


class CNNRNN(nn.Module):
    def __init__(self):
        super(CNNRNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=[2,1])
        self.bn1 = nn.BatchNorm2d(32)
        self.conv1_drop = nn.Dropout2d(p=0.2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2_drop = nn.Dropout2d(p=0.25)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3_drop = nn.Dropout2d(p=0.3)
        
        self.rnn = nn.LSTM(256, 128, 2, batch_first=True)
        
        self.fc1 = nn.Linear(1280,640)
        self.fc1_drop = nn.Dropout(p=0.4)
    
        self.fc2 = nn.Linear(640,30)
        self.fc2_drop = nn.Dropout(p=0.5)
    
    
    def forward(self, x):
        #print(x.shape)
        x = F.max_pool2d(self.bn1(self.conv1_drop(F.relu(self.conv1(x)))), 2)
        #print(x.shape)
        x = F.max_pool2d(self.bn2(self.conv2_drop(F.relu(self.conv2(x)))), 2)
        #print(x.shape)
        x = F.max_pool2d(self.bn3(self.conv3_drop(F.relu(self.conv3(x)))), 2)
        #print(x.shape)
        
        x = x.transpose(1, -1)
        #print(x.shape)
        
        batch, time = x.size()[:2]
        x = x.reshape(batch, time, -1)
        #print(x.shape)
        
        x, hidden = self.rnn(x)
        
        #print(x.shape)
        x = x.reshape(-1, 10*128)
        #print(x.shape)
        x = F.relu(self.fc1_drop(self.fc1(x)))
        x = F.relu(self.fc2_drop(self.fc2(x)))
        #print(x.shape)
        return F.log_softmax(x, dim=1)



    


class LeNet(nn.Module):
    def __init__(self,linear_layer_dim):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(linear_layer_dim, 1000)
        self.fc2 = nn.Linear(1000, 30)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)





cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name,linear_layer_dim):
        super(VGG, self).__init__()
        self.features = _make_layers(cfg[vgg_name])
        self.fc1 = nn.Linear(linear_layer_dim, 512)
        self.fc2 = nn.Linear(512, 30)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)





class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 128, kernel_size=80, stride=4)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv1_drop = nn.Dropout(p=0.3)
        self.max_pool_1=nn.MaxPool1d(4)
        
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv2_drop = nn.Dropout(p=0.3)
        self.max_pool_2=nn.MaxPool1d(4)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv3_drop = nn.Dropout(p=0.3)
        self.max_pool_3=nn.MaxPool1d(4)
        
        
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv4_drop = nn.Dropout(p=0.3)
        self.max_pool_4=nn.MaxPool1d(4)
        
        
        
        self.conv5 = nn.Conv1d(256, 512, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm1d(512)
        self.conv5_drop = nn.Dropout(p=0.3)
        self.avg_pool_5=nn.AvgPool1d(4)
        self.rnn = nn.LSTM(64, 32, 2, batch_first=True, bidirectional=True)
        
        #self.fc1 = nn.Linear(9600,1024) 100 kernel size
        self.fc1 = nn.Linear(7199,1024)
        self.fc1_drop = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(1536,30)
        self.fc2_drop = nn.Dropout(p=0.4)
    
    def forward(self, x):
        #print(x.shape)
        x = self.max_pool_1(self.bn1(F.relu(self.conv1(x))))
        #print(x.shape)
        x = self.max_pool_2(self.bn2(F.relu(self.conv2(x))))
        #print(x.shape)
        
        x = self.max_pool_3(self.bn3(F.relu(self.conv3(x))))
        #print(x.shape)
        x = self.max_pool_4(self.bn4(F.relu(self.conv4(x))))
        #print(x.shape)
        
        x = self.avg_pool_5(self.bn5(F.relu(self.conv5(x))))
        #print(x.shape)
        
        #x = x.transpose(1, -1)
        #print(x.shape)
      
        
        #x, hidden = self.rnn(x)
        #print(x.shape)
        #conv_seq_len = x.size(1)
        
        #x = x.reshape(-1, 64 * conv_seq_len)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        #x = F.relu(self.fc1_drop(self.fc1(x)))
        #print(x.shape)
        x = F.relu(self.fc2_drop(self.fc2(x)))
        #print(x.shape)
        return F.log_softmax(x, dim=1)



class CNN1DRNN(nn.Module):
    def __init__(self):
        super(CNN1DRNN, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 128, kernel_size=80, stride=4)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv1_drop = nn.Dropout(p=0.3)
        self.max_pool_1=nn.MaxPool1d(4)
        
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv2_drop = nn.Dropout(p=0.3)
        self.max_pool_2=nn.MaxPool1d(4)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv3_drop = nn.Dropout(p=0.3)
        self.max_pool_3=nn.MaxPool1d(4)
        
        
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv4_drop = nn.Dropout(p=0.3)
        self.max_pool_4=nn.MaxPool1d(4)
        
        
        
        self.conv5 = nn.Conv1d(256, 512, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm1d(512)
        self.conv5_drop = nn.Dropout(p=0.3)
        self.avg_pool_5=nn.AvgPool1d(4)
        self.rnn = nn.LSTM(256, 128, 2, batch_first=True, bidirectional=True)
        
        #self.fc1 = nn.Linear(9600,1024) 100 kernel size
        self.fc1 = nn.Linear(7199,1024)
        self.fc1_drop = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(3584,30)
        self.fc2_drop = nn.Dropout(p=0.4)
    
    def forward(self, x):
        #print(x.shape)
        x = self.max_pool_1(self.bn1(F.relu(self.conv1(x))))
        #print(x.shape)
        x = self.max_pool_2(self.bn2(F.relu(self.conv2(x))))
        #print(x.shape)
        
        x = self.max_pool_3(self.bn3(F.relu(self.conv3(x))))
        #print(x.shape)
        x = self.max_pool_4(self.bn4(F.relu(self.conv4(x))))
        #print(x.shape)
        
        #x = self.avg_pool_5(self.bn5(F.relu(self.conv5(x))))
        #print(x.shape)
        
        x = x.transpose(1, -1)
        #print(x.shape)
        
        
        x, hidden = self.rnn(x)
        #print(x.shape)
        conv_seq_len = x.size(1)
        
        x = x.reshape(-1, 256 * conv_seq_len)
        #print(x.shape)
        #x = x.view(x.size(0), -1)
        #print(x.shape)
        #x = F.relu(self.fc1_drop(self.fc1(x)))
        #print(x.shape)
        x = F.relu(self.fc2_drop(self.fc2(x)))
        #print(x.shape)
        return F.log_softmax(x, dim=1)
