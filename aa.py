
import torch
import torchvision.models as models
import thop
import numpy as np
import torch.utils.data as Data

test = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
input = torch.tensor(np.array([test[i:i+3] for i in range(10)]))
target = torch.tensor(np.array([test[i:i+1] for i in range(10)]))
dataset = Data.TensorDataset(input,target)
batch = 3

def collate_fn(x):
    value, target = zip(*x)
    return torch.stack(value,0),torch.stack(target,0)


loader = Data.DataLoader(dataset=dataset,batch_size=batch,collate_fn=collate_fn)

for i , j in loader:
    print(i, j)

loader2 = Data.DataLoader(dataset=dataset,batch_size=batch,collate_fn=lambda x:x)

for v in loader2:
    print(v)