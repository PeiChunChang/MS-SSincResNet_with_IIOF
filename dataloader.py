import torch
import torch.utils.data as data

import os
import numpy as np
import scipy.io as sio

torch.manual_seed(1) # cpu
torch.cuda.manual_seed(1) #gpu
np.random.seed(1) #numpy

def random_scale(data):
    rand_scale = torch.rand(data.size()[0], 1) * 0.2 + 0.9
    rand_scale = rand_scale.repeat(1, data.size()[1])
    rand_scale.unsqueeze_(dim=1)
    return torch.mul(data, rand_scale)
def random_noise(data):
    rand_noise = (torch.rand(data.size()) - 0.5) * 0.02
    return data + rand_noise
def transform(data):
    data = random_scale(torch.from_numpy(data).float().unsqueeze_(dim=0))
    data = random_noise(data)
    return data

# only static annotations (VA)
def loader_clip_DEAM(data_name, anno_name, clip_num, train=True, fs=16000):
    data = np.load(data_name)
    data = data / np.max(np.absolute(data))
    data = data.reshape(1, data.shape[0])
    annos = sio.loadmat(anno_name)
    V_anno = annos['g_m_valence']
    A_anno = annos['g_m_arousal']
    annos = (np.concatenate((V_anno, A_anno), axis=0) - 1.0)/4.0 - 1.0

    segment_size = data.shape[1]//clip_num
    new_data = np.zeros((clip_num, 1, segment_size))
    new_annos = np.zeros((clip_num, 2))

    for i in range(clip_num):
        new_data[i] = data[:, i * segment_size : (i + 1) * segment_size]
        new_annos[i] = annos.T
    return new_data, new_annos
def loader_clip_PMEmo(data_name, anno_name, clip_num, train=True, fs=16000):
    data = np.load(data_name)
    data = data / np.max(np.absolute(data))
    data = data.reshape(1, data.shape[0])
    annos = sio.loadmat(anno_name)

    V_anno = annos['static_anno_V']
    A_anno = annos['static_anno_A']
    annos = (np.concatenate((V_anno, A_anno), axis=0) - 0.5) *2

    segment_num = (data.shape[1] / fs) // 9
    segment_size = int(fs * 9) 

    if train == True:
        new_data = np.zeros((clip_num, 1, segment_size))
        new_annos = np.zeros((clip_num, 2))
        rand_segment = np.random.randint(segment_num, size=clip_num)
    else:
        new_data = np.zeros((int(segment_num), 1, segment_size))
        new_annos = np.zeros((int(segment_num), 2))
        rand_segment = np.arange(segment_num)

    for (i, index) in enumerate(rand_segment):
        new_data[i] = data[:, int(index) * segment_size: (int(index) + 1) * segment_size]
        new_annos[i] = annos.T
    return new_data, new_annos

class myDataset(data.Dataset):
    def __init__(self, datapath, filename, loader, transform=False, train=True, clip_num=5):
        self.datapath = datapath
        self.annopath = datapath + 'anno_mat/'
        self.transform = transform
        self.loader = loader
        self.data = []
        self.train = train
        self.clip_num = clip_num
        filename = np.load(filename)
        for (i, data) in enumerate(filename):
            self.data.append(data)

    def __getitem__(self, index):
        data_name = self.datapath + 'npy/' + self.data[index]
        anno_name = self.annopath + self.data[index][:-3] + 'mat'

        data, label = self.loader(data_name, anno_name, clip_num=self.clip_num, train=self.train)
        torch_data = torch.zeros(data.shape)
        if self.transform == True:
            for i in range(self.clip_num):
                torch_data[i] = transform(data[i])
            label = torch.from_numpy(label)
        else:
            torch_data = torch.from_numpy(data).float()
            label = torch.from_numpy(label)
        return torch_data, label.type(torch.float32)

    def __len__(self):
        return len(self.data)