import os
import os.path

import librosa
import numpy as np
import torch
import torch.utils.data as data

AUDIO_EXTENSIONS = [
    '.wav', '.WAV',
]


def is_audio_file(filename):
    """Return true if the file is an audio file"""
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def find_classes(dir):
    """ Returns a tuple of class of the audio file and id associated with it"""
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    spects = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    spects.append(item)
    return spects


def cleanData(y):
  """ Remove the points from the raw audio array have negligible values""" 
  y_new = []
  data_array_len = 7000  # Fic the length for each input data. Will be useful for training RNN
  for i in range(len(y)):
    if  -0.005  > y[i]  or 0.005 < y[i]: 
      y_new.append(y[i])

  y_new.extend([float(0)] * (data_array_len - len(y_new)))
      
  return np.array(y_new)


def spect_loader(path, window_size, window_stride, window, normalize, input_format,  max_len=101, clean_data=False):
    y, sr = librosa.load(path, sr=None)
    # n_fft = 4096
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)
    spect=np.array([])
	

    # Clean the input
    if(clean_data):
        y = cleanData(y)

    # Data processing: Convert audio files to the desired format based on the given input_format 

    # 1. STFT: allows one to see how different frequencies change over time.
    if(input_format=="STFT"):
    	D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    	spect, phase = librosa.magphase(D)
    	# S = log(S+1)
    	spect = np.log1p(spect)
	
    #2. Mel
    if(input_format=="MEL32"):
        S=librosa.feature.melspectrogram(y, sr=sr,n_fft=n_fft, hop_length=hop_length, n_mels=32)
        spect = librosa.power_to_db(abs(S))
	
    if(input_format=="MEL40"):
        S=librosa.feature.melspectrogram(y, sr=sr,n_fft=n_fft, hop_length=hop_length, n_mels=40)
        spect = librosa.power_to_db(abs(S))

    if(input_format=="MEL100"):
        S=librosa.feature.melspectrogram(y, sr=sr,n_fft=n_fft, hop_length=hop_length, n_mels=100)
        spect = librosa.power_to_db(abs(S))
    
    if(input_format=="RAW"):
        max_len = 16000
        spect = y
        if spect.shape[0] < max_len:
        
            pad = np.zeros((max_len - spect.shape[0]))
            spect = np.hstack((spect, pad))
        elif spect.shape[0] > max_len:
    
            spect = spect[:, :max_len]
        spect = np.resize(spect, (1, spect.shape[0]))
        spect = torch.FloatTensor(spect)

        return spect
    

    
    # make all spects with the same dims
    # TODO: change that in the future
    if spect.shape[1] < max_len:
        
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        
        spect = spect[:, :max_len]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    spect = torch.FloatTensor(spect)

    # z-score normalization
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)
    
    return spect


class SpeechDataLoader(data.Dataset):
    """A google speech command data set loader where the wavs are arranged in this way: ::
        root/one/xxx.wav
        root/one/xxy.wav
        root/one/xxz.wav
        root/head/123.wav
        root/head/nsdf3.wav
        root/head/asd932_.wav
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        window_size: window size for the stft, default value is .02
        window_stride: window stride for the stft, default value is .01
        window_type: typye of window to extract the stft, default value is 'hamming'
        normalize: boolean, whether or not to normalize the spect to have zero mean and one std
        max_len: the maximum length of frames to use
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        spects (list): List of (spects path, class_index) tuples
        STFT parameter: window_size, window_stride, window_type, normalize
    """

    def __init__(self, root, input_format,  transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=101, clean_data=False):
        classes, class_to_idx = find_classes(root)
        spects = make_dataset(root, class_to_idx)
        if len(spects) == 0:
            raise (RuntimeError("Found 0 sound files in subfolders of: " + root + "Supported audio file extensions are: " + ",".join(AUDIO_EXTENSIONS)))

        self.root = root
        self.spects = spects
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = spect_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len
        self.input_format=input_format
        self.clean_data=clean_data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        path, target = self.spects[index]
        spect = self.loader(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.input_format,  self.max_len, self.clean_data)
        if self.transform is not None:
            spect = self.transform(spect)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return spect, target

    def __len__(self):
        return len(self.spects)
