import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable

from dataloader import *
from models import *
from utils import progress_bar

import csv
import math
import argparse
import scipy.io as sio
from sklearn.metrics import r2_score, mean_squared_error

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True
torch.manual_seed(1) # cpu
torch.cuda.manual_seed(1) #gpu
np.random.seed(1) #numpy

parser = argparse.ArgumentParser()
parser.add_argument('--cv_num', type=int, default=10, help='fold number')
parser.add_argument('--dataset_path', type=str, default='/media/maplepig/Data2/Datasets/PMEmo/',help='dataset path')
parser.add_argument('--max_epoch', type=int, default=200, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--init_lr', type=float, default=0.0001, help='initialization of learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

EPOCH_MAX = args.max_epoch
BATCH_SIZE = args.batch_size
CV_NUM = args.cv_num
INIT_LR = args.init_lr

def adjust_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.5

def train(data, label):
    model.train()
    data, label = data.to(device), label.to(device)
    output = model(data)
    loss =  criteria(output, label)
    with torch.autograd.set_detect_anomaly(True):
        loss.backward()
    return loss

def test(data):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        output = model(data)
    return output

def CC(predict, gt):
    vx = predict - torch.mean(predict)
    vy = gt -torch.mean(gt)
    cc = torch.sum(vx * vy) / ((torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))))
    return cc

for cv in range(CV_NUM):
    TRAIN_PATH = args.dataset_path + 'CV_10_with_val/fold_%d_train.npy' % cv
    TEST_PATH = args.dataset_path + 'CV_10_with_val/fold_%d_test.npy' % cv
    VAL_PATH = args.dataset_path + 'CV_10_with_val/fold_%d_val.npy' % cv

    OUTPUT_PATH = './output_PEMmo/fold_%d' % (cv)
    if not os.path.exists(OUTPUT_PATH + '/checkpoints'):
        os.makedirs(OUTPUT_PATH + '/checkpoints')

    TrainDataLoader = torch.utils.data.DataLoader(
        myDataset(args.dataset_path, TRAIN_PATH, loader=loader_clip_PMEmo, transform=True, train=True, clip_num=5), 
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8)
    TestDataLoader = torch.utils.data.DataLoader(
        myDataset(args.dataset_path, TEST_PATH, transform=False, loader=loader_clip_PMEmo, train=False), 
        batch_size=1, shuffle=False, num_workers=4)
    ValDataLoader = torch.utils.data.DataLoader(
        myDataset(args.dataset_path, VAL_PATH, transform=False, loader=loader_clip_PMEmo, train=False), 
        batch_size=1, shuffle=False, num_workers=4)

    model = MS_SSincResNet_IIOF()
    model = nn.DataParallel(model)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=INIT_LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    optimizer.zero_grad()
    criteria = nn.L1Loss()
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    np_V_CC = np.zeros(EPOCH_MAX)
    np_A_CC = np.zeros(EPOCH_MAX)
    np_V_RMSE = np.zeros(EPOCH_MAX)
    np_A_RMSE = np.zeros(EPOCH_MAX)
    np_V_R2 = np.zeros(EPOCH_MAX)
    np_A_R2 = np.zeros(EPOCH_MAX)
    np_V_MSE = np.zeros(EPOCH_MAX)
    np_A_MSE = np.zeros(EPOCH_MAX)
    np_test_epoch = np.zeros(EPOCH_MAX)
    np_test_A_R2 = np.zeros(EPOCH_MAX)
    np_test_V_R2 = np.zeros(EPOCH_MAX)
    np_test_A_CC = np.zeros(EPOCH_MAX)
    np_test_V_CC = np.zeros(EPOCH_MAX)
    np_test_A_RMSE = np.zeros(EPOCH_MAX)
    np_test_V_RMSE = np.zeros(EPOCH_MAX)
    np_test_A_MSE = np.zeros(EPOCH_MAX)
    np_test_V_MSE = np.zeros(EPOCH_MAX)

    BEST_epoch = 0
    BEST_np_Avg_R2 = -100.0

    # opening the csv file in 'w' mode 
    file = open(OUTPUT_PATH + '_record.csv', 'w', newline ='', encoding='utf-8-sig')
    header = ['Epoch', 'Train_loss', 'V_RMSE', 'V_CC', 'V_R2', 'V_MSE', 'A_RMSE', 'A_CC', 'A_R2', 'A_MSE', 'BEST_test_epoch', 'BEST_Avg_R2'] 
    writer = csv.DictWriter(file, fieldnames = header) 
    writer.writeheader()
    for epoch in range(EPOCH_MAX):
        if epoch % 30 == 0 and epoch != 0 and epoch < 120:
            adjust_learning_rate(optimizer, epoch)
        
        total_train_loss = 0.0
        for batch_idx, (data, label) in enumerate(TrainDataLoader):
            data = data.view(data.size()[0] * data.size()[1], data.size()[2], data.size()[3])
            label = label.view(label.size()[0] * label.size()[1], label.size()[2])
            loss = train(data, label)
            total_train_loss += loss

            progress_bar(batch_idx, len(TrainDataLoader), 
            'Fold_%d Ep %d/%d avg. loss = %.4f' %(cv, epoch, EPOCH_MAX-1, total_train_loss/(batch_idx+1)))
            optimizer.step()
            optimizer.zero_grad()

        output_v = torch.zeros(len(ValDataLoader)).to(device)
        output_a = torch.zeros(len(ValDataLoader)).to(device)
        GT_v = torch.zeros(len(ValDataLoader)).to(device)
        GT_a  = torch.zeros(len(ValDataLoader)).to(device)

        for batch_idx, (data, label) in enumerate(ValDataLoader):
            data = data.view(data.size()[0] * data.size()[1], data.size()[2], data.size()[3])
            label = label.view(label.size()[0] * label.size()[1], label.size()[2])
            output = test(data)
            output = torch.mean(output, dim=0)

            output_v[batch_idx] = output[0]
            output_a[batch_idx] = output[1]
            GT_v[batch_idx] = label[0, 0]
            GT_a[batch_idx] = label[0, 1]

        V_CC = CC(output_v, GT_v)
        A_CC = CC(output_a, GT_a)
        V_RMSE = torch.sqrt(F.mse_loss(output_v, GT_v))
        A_RMSE = torch.sqrt(F.mse_loss(output_a, GT_a))
        V_R2 = r2_score(GT_v.cpu().numpy(), output_v.cpu().numpy())
        A_R2 = r2_score(GT_a.cpu().numpy(), output_a.cpu().numpy())
        V_MSE = mean_squared_error(output_v.cpu().numpy(), GT_v.cpu().numpy())
        A_MSE = mean_squared_error(output_a.cpu().numpy(), GT_a.cpu().numpy())
        print('**VAL: [V]CC/RMSE/R2/MSE: %.3f/%.3f/%.3f/%.3f. [A]CC/RMSE/R2/MSE: %.3f/%.3f/%.3f/%.3f' 
                                            %(V_CC,V_RMSE,V_R2,V_MSE,A_CC,A_RMSE,A_R2,A_MSE))
        np_V_CC[epoch] = V_CC.data.cpu().numpy()
        np_A_CC[epoch] = A_CC.data.cpu().numpy()
        np_V_RMSE[epoch] = V_RMSE.data.cpu().numpy()
        np_A_RMSE[epoch] = A_RMSE.data.cpu().numpy()
        np_V_R2[epoch] = V_R2
        np_A_R2[epoch] = A_R2
        np_V_MSE[epoch] = V_MSE
        np_A_MSE[epoch] = A_MSE

        if (np_A_R2[epoch] + np_V_R2[epoch]) > BEST_np_Avg_R2:
            BEST_epoch = epoch
            BEST_np_Avg_R2 = np_A_R2[epoch] + np_V_R2[epoch]

            savefilename = OUTPUT_PATH + '/checkpoints/BEST_checkpoint.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss/len(TrainDataLoader)
            }, savefilename)

            test_output_v = torch.zeros(len(TestDataLoader)).to(device)
            test_output_a = torch.zeros(len(TestDataLoader)).to(device)
            test_GT_v = torch.zeros(len(TestDataLoader)).to(device)
            test_GT_a  = torch.zeros(len(TestDataLoader)).to(device)

            for batch_idx, (data, label) in enumerate(TestDataLoader):
                data = data.view(data.size()[0] * data.size()[1], data.size()[2], data.size()[3])
                label = label.view(label.size()[0] * label.size()[1], label.size()[2])
                output = test(data)
                output = torch.mean(output, dim=0)

                test_output_v[batch_idx] = output[0]
                test_output_a[batch_idx] = output[1]
                test_GT_v[batch_idx] = label[0, 0]
                test_GT_a[batch_idx] = label[0, 1]

            test_V_CC = CC(test_output_v, test_GT_v)
            test_A_CC = CC(test_output_a, test_GT_a)
            test_V_RMSE = torch.sqrt(F.mse_loss(test_output_v, test_GT_v))
            test_A_RMSE = torch.sqrt(F.mse_loss(test_output_a, test_GT_a))
            test_np_V_R2 = r2_score(test_GT_v.cpu().numpy(), test_output_v.cpu().numpy())
            test_np_A_R2 = r2_score(test_GT_a.cpu().numpy(), test_output_a.cpu().numpy())
            test_np_V_MSE = mean_squared_error(test_output_v.cpu().numpy(), test_GT_v.cpu().numpy())
            test_np_A_MSE = mean_squared_error(test_output_a.cpu().numpy(), test_GT_a.cpu().numpy())

            test_np_V_CC = test_V_CC.data.cpu().numpy()
            test_np_A_CC = test_A_CC.data.cpu().numpy()
            test_np_V_RMSE = test_V_RMSE.data.cpu().numpy()
            test_np_A_RMSE = test_A_RMSE.data.cpu().numpy()

        np_test_epoch[epoch] = BEST_np_Avg_R2
        np_test_A_R2[epoch] = test_np_A_R2
        np_test_A_CC[epoch] = test_np_A_CC
        np_test_A_RMSE[epoch] = test_np_A_RMSE
        np_test_A_MSE[epoch] = test_np_A_MSE
        np_test_V_R2[epoch] = test_np_V_R2
        np_test_V_CC[epoch] = test_np_V_CC
        np_test_V_RMSE[epoch] = test_np_V_RMSE
        np_test_V_MSE[epoch] = test_np_V_MSE
        
        writer.writerow({'Epoch': epoch,  
                'Train_loss': (total_train_loss/len(TrainDataLoader)).data.cpu().numpy(),
                'V_RMSE': V_RMSE.data.cpu().numpy(),
                'V_CC': V_CC.data.cpu().numpy(),
                'V_R2': V_R2,
                'V_MSE': V_MSE,
                'A_RMSE': A_RMSE.data.cpu().numpy(),
                'A_CC': A_CC.data.cpu().numpy(),
                'A_R2': A_R2,
                'A_MSE': A_MSE,
                'BEST_test_epoch': BEST_epoch,
                'BEST_Avg_R2': BEST_np_Avg_R2})
        file.flush()

        print('**Test(%d): [V]CC/RMSE/R2/MSE: %.3f/%.3f/%.3f/%.3f. [A]CC/RMSE/R2/MSE: %.3f/%.3f/%.3f/%.3f' 
                %(BEST_epoch,test_np_V_CC,test_np_V_MSE,test_np_V_R2,test_np_V_MSE,test_np_A_CC,test_np_A_RMSE,test_np_A_R2,test_np_A_MSE))

    sio.savemat(OUTPUT_PATH + '_CM.mat', {"V_CC": np_V_CC, "A_CC": np_A_CC, "V_RMSE": np_V_RMSE, "A_RMSE": np_A_RMSE,
                                          "V_R2": np_V_R2, "A_R2": np_A_R2, "V_MSE": np_V_MSE, "A_MSE": np_A_MSE,
                                          "test_V_CC": test_np_V_CC, "test_A_CC": test_np_A_CC, "test_V_RMSE": test_np_V_RMSE, "test_A_RMSE": test_np_A_RMSE,
                                          "test_V_R2": test_np_V_R2, "test_A_R2": test_np_A_R2, "test_V_MSE": test_np_V_MSE, "test_A_MSE": test_np_A_MSE,
                                          "test_epoch_best": np_test_epoch, 
                                          "test_V_CC_array": np_test_V_CC, "test_A_CC_array": np_test_A_CC, "test_V_RMSE_array": np_test_V_RMSE, "test_A_RMSE_array": np_test_A_RMSE,
                                          "test_V_R2_array": np_test_V_R2, "test_A_R2_array": np_test_A_R2, "test_V_MSE_array": np_test_V_MSE, "test_A_MSE_array": np_test_A_MSE})
    file.close()
