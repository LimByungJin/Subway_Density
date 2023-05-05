from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn as nn

class CustomData(Dataset):
    def __init__(self):
        origin_text = open('data/clean_raw_data.txt')
        np_origin = np.array([])
        while(True):
            tem = origin_text.readline()
            if not tem:
                break
            split_tem = tem.split(',')
            add_tem = np.array([])
            for i in range(len(split_tem)):
                add_tem = np.append(add_tem,np.array(int(float(split_tem[i]))))
            np_origin = np.append(np_origin,add_tem,axis=0)
        np_origin = np_origin.reshape(-1,len(add_tem))
        
        data_num = np.zeros((len(np_origin), len(np_origin[0]) - 2))
        for i in range(len(np_origin[0]) - 2):
            for j in range(len(np_origin)):
                data_num[j][i] = np_origin[j][i + 2]
                
        self.np_origin = torch.from_numpy(np_origin)
        self.origin_data = torch.from_numpy(data_num)
                
        origin_sum = np.zeros(len(data_num))
        for i in range(len(data_num)):
            origin_sum[i] = data_num[i].sum()
                       
        origin_den = np.zeros((len(data_num),len(data_num[0])))
        for i in range(len(data_num)):
            for j in range(len(data_num[i])):
                origin_den[i][j] = data_num[i][j]/origin_sum[i]
                               
        self.origin_sum = torch.from_numpy(origin_sum)
        self.origin_den = torch.from_numpy(origin_den)
        
        remake_text = open('data/uniform_perturbed_0.1.txt')
        np_remake = np.array([])
        while(True):
            tem = remake_text.readline()
            if not tem:
                break
            split_tem = tem.split(',')
            add_tem = np.array([])
            for i in range(len(split_tem)):
                add_tem = np.append(add_tem,np.array(int(float(split_tem[i]))))
            np_remake = np.append(np_remake,add_tem,axis=0)
            
        np_remake = np_remake.reshape(-1,len(add_tem))
        self.remake_data = torch.from_numpy(np_remake)
        
        remake_sum = np.zeros(len(np_remake))
        for i in range(len(np_remake)):
            remake_sum[i] = np_remake[i].sum()
            
        remake_den = np.zeros((len(np_remake),len(np_remake[0])))
        for i in range(len(np_remake)):
            for j in range(len(np_remake[i])):
                remake_den[i][j] = np_remake[i][j]/remake_sum[i]
                
        self.remake_sum = torch.from_numpy(remake_sum)
        self.remake_den = torch.from_numpy(remake_den)
            
    def __len__(self): 
        return len(self.origin_data)

    def __getitem__(self, idx): 
        x = torch.DoubleTensor(self.origin_data[idx]).int()
        y = torch.DoubleTensor(self.remake_data[idx]).int()
        a = self.origin_sum[idx]
        b = self.remake_sum[idx]
        c = self.origin_den[idx]
        d = self.remake_den[idx]
        e = self.np_origin[idx][0]
        f = self.np_origin[idx][1]
        l = 18*e+f
        return e,f,l,idx,c,d
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(nn.Linear(239, 256),
                                nn.LeakyReLU(0.2),
                                nn.Linear(256,512),
                                nn.LeakyReLU(0.2),
                                nn.Linear(512,1024),
                                nn.LeakyReLU(0.2),
                                nn.Linear(1024,114),
                                nn.Sigmoid())

    def forward(self,noise,label):
        gen_input = torch.cat((noise,label),-1)
        x = self.gen(gen_input.float())
        x = nn.functional.normalize(x,p=1,dim=-1)
        return x
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dis = nn.Sequential(nn.Linear(253,128),
                                nn.LeakyReLU(0.2),
                                nn.Linear(128,64),
                                nn.LeakyReLU(0.2),
                                nn.Linear(64,32),
                                nn.LeakyReLU(0.2),
                                nn.Linear(32,1),
                                nn.Sigmoid())

    def forward(self, img, label):
        dis_input = torch.cat((img,label),-1)
        x = self.dis(dis_input.float())
        return x