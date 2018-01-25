from datetime import datetime

import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import torch
import torch.utils.data

import util


torch.backends.cudnn.benchmark = True
Variable = torch.autograd.Variable


def distance(x):
    aa = bb = torch.sum(x * x, dim=1, keepdim=True)
    ab = torch.mm(x, x.t())
    return aa + bb.t() - 2 * ab


def similarity(x, k, beta):
    d = distance(x)
    s = torch.exp(-d / beta)
    t, _ = d.sort()
    s = s * (d < t[:, k+1].clone().view(-1, 1)).type_as(d) * (1 - torch.eye(d.size(0))).cuda()
    return s


class Metric(object):

    def __init__(self):
        self.sum = 0.
        self.count = 0

    def update(self, loss, count):
        self.sum += loss * count
        self.count += count

    def get(self):
        return self.sum / self.count


class MLP(torch.nn.Module):
    
    def __init__(self, i, h, o):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(i, h)
        self.act1 = torch.nn.ReLU(True)
        self.fc2 = torch.nn.Linear(h, o)
        

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.fc2(x)
        return x


class GLSNN(object):

    def __init__(self, i, h, o, Q):
        self.model = MLP(i, h, o).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0005)
        self.Q = torch.nn.Parameter(Q)

    def train(self, train_loader):
        self.model.train()
        me = Metric()

        for data, _ in train_loader:
            x = Variable(data.cuda())
            o = self.model(x)

            # eta = 0.5, k = 20, beta = 1
            loss = 0.5 * torch.mean(torch.pow(x - torch.mm(o, self.Q.t()), 2)) + 0.5 * torch.mean(distance(o) * Variable(similarity(x.data, 20, 1)))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
           
            try:
                U, _, V = torch.svd(torch.mm(x.data.t(), o.data))
            except:
                continue
            self.Q.data[:] = torch.mm(U, V.t())
    
            me.update(loss.data.cpu().numpy()[0], data.size(0))
        return me.get()

    def predict(self, data_loader):
        self.model.eval()

        def _predict(data):
            x = Variable(data[0].cuda())
            o = self.model(x)
            return o.data.cpu().numpy()
        
        return np.vstack(list(map(_predict, data_loader)))

    
def get_data():
    train_data, _ = util.read_data(error=0, is_train=True)
    test_data = np.vstack(list(map(lambda x: util.read_data(x, False)[0], range(22))))
    scaler = preprocessing.StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    return train_data, test_data


def get_loader(train_data, test_data):
    train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(train_data), torch.ones(train_data.shape[0])), 
            batch_size=500, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(test_data), torch.zeros(test_data.shape[0])), 
            batch_size=128, shuffle=False)
    return train_loader, test_loader


def main():
    train_data, test_data = get_data()
    train_loader, test_loader = get_loader(train_data, test_data)
    
    pca = PCA(27).fit(train_data)
    Q = torch.from_numpy(pca.components_.T).cuda()
    glsnn = GLSNN(52, 27, 27, Q) 

    for i in range(1000):
        loss = glsnn.train(train_loader)
        if i % 10 == 0:
            print('{} Epoch[{}]  loss = {:0.3f}'.format(datetime.now(), i, loss))
    
    pred = glsnn.predict(test_loader)


if __name__ == '__main__':
    main()

