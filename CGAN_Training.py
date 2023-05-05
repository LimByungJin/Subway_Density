import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from torch import optim
from Custom_Classes import CustomData, Generator, Discriminator

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("USING :",device)

lr = 2e-4
beta1 = 0.5
beta2 = 0.999
batch_count = 0
num_epochs = 300

dataset = CustomData()
train_dataset, test_dataset = torch.utils.data.random_split(dataset,[10500,2658],generator=torch.Generator().manual_seed(0))
dataloader = DataLoader(train_dataset,batch_size=128,shuffle=True,drop_last=False)

def initialize_weights(model):
    classname = model.__class__.__name__
    if(classname.find('Linear') != -1):
        nn.init.normal_(model.weight.data,0.0,0.02)
        nn.init.constant_(model.bias.data,0)
    elif(classname.find('BatchNorm') != -1):
        nn.init.normal_(model.weight.data,1.0,0.02)
        nn.init.constant_(model.bias.data,0)
        
model_gen = Generator().to(device)
model_gen.apply(initialize_weights);
model_dis = Discriminator().to(device)
model_dis.apply(initialize_weights);

loss_func = nn.BCELoss()

opt_dis = optim.Adam(model_dis.parameters(), lr = lr, betas=(beta1,beta2))
opt_gen = optim.Adam(model_gen.parameters(), lr = lr, betas=(beta1,beta2))

loss_history={'gen':[],'dis':[]}

start_time = time.time()
model_dis.train()
model_gen.train()
for epoch in range(num_epochs):
    for a1,b1,c1,d1,e1,f1 in dataloader:
        ba_si = e1.shape[0]
        
        xb = e1.to(device)
        yb = c1.to(device)
        yb_real = torch.ones(ba_si, 1).to(device)
        yb_fake = torch.zeros(ba_si, 1).to(device)

        model_gen.zero_grad()
        noise = torch.randn(ba_si,100).to(device)
        
        one_hot = np.zeros([ba_si,25])
        for i in range(len(c1)):
            one_hot[i][int(a1[i])]=1
            one_hot[i][int(b1[i])+7]=1
        one_hot = torch.from_numpy(one_hot).to(device)
        gen_label = torch.cat((one_hot,f1.to(device)),-1)
        out_gen = model_gen(noise, gen_label)
        out_dis = model_dis(out_gen, gen_label)
        
        loss_gen = loss_func(out_dis, yb_real)
        loss_gen.backward()
        opt_gen.step()

        model_dis.zero_grad()
        
        out_dis = model_dis(xb, gen_label)
        loss_real = loss_func(out_dis, yb_real)

        out_dis = model_dis(out_gen.detach(),gen_label)
        loss_fake = loss_func(out_dis,yb_fake)

        loss_dis = (loss_real + loss_fake) / 2
        loss_dis.backward()
        opt_dis.step()

        loss_history['gen'].append(loss_gen.item())
        loss_history['dis'].append(loss_dis.item())
        
        batch_count += 1  
        if batch_count % 83 == 0: # 10500 / 128 = 82.xx
            print('Epoch: %.0f, G_Loss: %.6f, D_Loss: %.6f, time: %.2f min' %(epoch, loss_gen.item(), loss_dis.item(), (time.time()-start_time)/60))

torch.save(model_gen,'station_weight/gen_model_0.1.pth')
torch.save(model_dis,'station_weight/dis_model_0.1.pth')