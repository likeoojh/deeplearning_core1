import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler

class MyLinear(nn.Module):

    def __init__(self, input_dim=3, output_dim=2):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        super().__init__()
        
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        # |x| = (batch_size, input_dim)
        y = self.linear(x)
        # |y| = (batch_size, output_dim)
        
        return y
    
#logistic
# Define costum model.
class MyModel(nn.Module):
    
    
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        super().__init__()
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        # |x| = (batch_size, input_dim)
        y = self.act(self.linear(x))
        # |y| = (batch_size, output_dim)
        
        return y
    
# dnn regression
class MyModel(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        super().__init__()
        
        self.linear1 = nn.Linear(input_dim, 3)
        self.linear2 = nn.Linear(3, 3)
        self.linear3 = nn.Linear(3, output_dim)
        self.act = nn.ReLU()
        
    def forward(self, x):
        # |x| = (batch_size, input_dim)
        h = self.act(self.linear1(x)) # |h| = (batch_size, 3)
        h = self.act(self.linear2(h))
        y = self.linear3(h)
        # |y| = (batch_size, output_dim)
        
        return y

## train precedure (regression)
# train/ valid /test ration
ratios = [.6, .2, .2]
train_cnt = int(data.size(0) * ratios[0])
valid_cnt  = int(data.size(0) * ratios[1])
test_cnt  = data.size(0) - train_cnt - valid_cnt 
cnts = [train_cnt, valid_cnt, test_cnt]

# shuffle before split.
indices = torch.randperm(data.size(0))
x = torch.index_select(x, dim = 0, index = indices)
y = torch.index_select(y, dim = 0, indec = indices)

# split train, valid and test tset with each count.
x = list(x.split(cnt, dim = 0))
y = y.split(cnts, dim = 0)

for x_i, y_i in zip(x, y):
    print(x_i.size(), y_i.size())
    
# preprocessiong 
scaler = StandardScaler() 
scaler.fit(x[0].numpy()) # you must fit with train data only

x[0] = torch.from_numpy(scaler.transform(x[0].numpy())).float()
x[1] = torch.from_numpy(scaler.transform(x[1].numpy())).float()
x[2] = torch.from_numpy(scaler.transform(x[2].numpy())).float()

# build model & optimizer
model = nn.Sequential(
    nn.Linear(x[0].size(-1), 6),
    nn.LeakyReLU(),
    nn.Linear(6,5),
    nn.LeakyReLU(),
    nn.Linear(5,4),
    nn.Linear(4,3),
    nn.LeakyReLU(),
    nn.Linear(3, y[0].size(-1)),    
)

optimizer = optim.Adam(model.parameters())

# train
n_epochs = 4000
batch_size = 256
print_interval = 100

from copy import deepcopy

lowest_loss = np.inf
best_model = None 
early_stop = 100
lowest_epoch = np.inf

train_history, valid_history = [], []

for i in range(n_epochs):
    # shuffle before mini-batch split.
    indices = torch.randperm(x[0].size(0))
    x_ = torch.index_select(x[0], dim=0, index=indices)
    y_ = torch.index_select(y[0], dim=0, index=indices)
    # |x_| = (total_size, input_dim)
    # |y_| = (total_size, output_dim)
   
    x_ = x_.split(batch_size, dim=0)
    y_ = y_.split(batch_size, dim=0)
    # |x_[i]| = (batch_size, input_dim)
    # |y_[i]| = (batch_size, output_dim)
   
    train_loss, valid_loss = 0,0
    y_hat= []
   
    for x_i, y_i in zip(x_, y_):
        # |x_i| = |x_[i]|
        # |y_i| = |y_[i]|
        y_hat_i = model(x_i)
        loss = F.mse_loss(y_hat_i, y_i)
       
        optimizer.zero_grad()
        loss.backward()
       
        optimizer.step()
        train_loss += float(loss)
    
    train_loss = train_loss / len(x_)
    
    # you need to declare to pytorch to stop build the computation graph.
    with torch.no_grad():
        # you don't need to shuffle the validation set.
        # only split is needed.
        x_ = x[1].split(batch_size, dim=0)
        y_ = y[1].split(batch_size, dim=0)
        
        valid_loss = 0
        
        for x_i, y_i in zip(x_, y_):
            y_hat_i = model(x_i)
            loss = F.mse_loss(y_hat_i, y_i)
            
            valid_loss += loss
            
            y_hat += [y_hat_i]
            
    valid_loss = valid_loss / len(x_)
    
    # log each loss to plot after training is done.
    train_history += [train_loss]
    valid_history += [valid_loss]
    
    if (i+1) % print_interval == 0:
        print('Epoch %d: train loss=%.4e valid_loss=%.4e lowest_loss=%.4e' % ( 
            i+1,
            train_loss,
            valid_loss,
            lowest_loss
        ))
       
    if valid_loss <= lowest_loss:
        lowest_loss = valid_loss
        lowest_epoch = i
        
        # 'state_dict()' return model weight as key-value.
        # take a deep copy, if the valid loss is lowest ever.
        best_model = deepcopy(model.state_dict())
    else:
        if early_stop > 0 and lowest_epoch + early_stop < i + 1:
            print("There is no imporvement during last %d epochs" % early_stop)
            break

print("The best validation loss from epoch %d: %.4e" % (lowest_epoch + 1, lowest_loss))

# Load best epoch's model.
model.load_state_dict(best_model)


# regularizations
ratios = [0.8, 0.2]

train_cnt = int(x.size(0) * ratios[0])
valid_cnt = int(x.size(0) * ratios[1])
test_cnt = len(test.data)
cnts = [train_cnt, valid_cnt]

indeices = torch.randperm(x.size(0))

x = torch.index_select(x, dim=0, index=indices)
y = torch.index_select(y, dim=0, index=indices)

x = list(x.split(cnts, dim=0))
y = list(y.split(cnts, dim=0))

x += [(test.data.float()/ 255.).vidw(test_cnt, -1)]
y += [test.targets]

for x_i, y_i in zip(x,y):
    print(x_i.size(), y_i.size())

class Block(nn.Module):
    
    def __init__(self, 
                 input_size,
                 output_size,
                 use_batch_norm=True,
                 dropout_p=.4):
        
        self.input_size = input_size
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p
        
        super().__init__()
        
        def get_regularization(use_batch_norm, size):
            return nn.BatchNorm1d(size) if use_batch_norm else nn.Dropout(dropout_p)
        
        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(),
            get_regularization(use_batch_norm, output_size)
        )
        
    def forward(self, x):
        # |x| = (batch_size, input_size)
        y = self.block(x)
        # |y| = (batch_size, output_size)
        
        return y

class MyModel(nn.Module):
    
    def __init__(self,
                 input_size,
                 output_size,
                 use_batch_norm=True,
                 dropout_p=.4):
        
        super().__init__()
        
        self.layers = nn.Sequential(
            Block(input_size, 500, use_batch_norm, dropout_p),
            Block(500, 400, use_batch_norm, dropout_p),
            Block(400, 300, use_batch_norm, dropout_p),
            Block(300, 200, use_batch_norm, dropout_p),
            Block(200, 100, use_batch_norm, dropout_p),
            nn.Linear(100, output_size),
            nn.LogSoftmax(dim=1),
        )
    
    def foward(self, x):
        # |x| = (batch_size, input_size)
        y = self.layers(x)
        # |y| = (batch_size, output_size)
        
        return y

model = MyModel(input_size,
                output_size,
                use_batch_norm=True)

crit = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())


n_epochs = 1000
batch_sizse = 256
print_interval = 10

train_history, valid_history = [], []

for i in range(n_epochs):
    model.train()
    
    x_ = torch.index_select(x[0], dim=0, index=indices)
    y_ = torch.index_select(y[0], dim=0, index=indices)
    
    x_ = x_.split(batch_size, dim=0)
    y_ = y_.split(batch_size, dim=0)
    
    train_loss, valid_loss = 0, 0
    y_hat = []
    
    for x_i, y_i in zip(x_, y_):
        y_hat_i = model(x_i)
        loss = crit(y_hat_i,y_i.squeeze())
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        train_loss += float(loss)
    
    train_loss = train_loss / len(x_)
    
    model.eval()
    with torch.no_grad():
        x_ = x[1].split(batch_size, dim=0)
        y_ = y[1].split(batch_size, dim=0)
        
        valid_loss=0
        
        for x_i, y_i in zip(x_,y_):
            y_hat_i = model(x_i)
            loss = crit(y_hat_i, y_i.squeeze())
            
            valid_loss += float(loss)
            
            y_hat += [y_hat_i]
    
    valid_loss = valid_loss / len(x_)
    
    train_history += [train_loss]
    valid_history += [valid_loss]
        
    if (i + 1) % print_interval == 0:
        print('Epoch %d: train loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e' % (
            i + 1,
            train_loss,
            valid_loss,
            lowest_loss,
        ))
        
    if valid_loss <= lowest_loss:
        lowest_loss = valid_loss
        lowest_epoch = i
        
        best_model = deepcopy(model.state_dict())
    else:
        if early_stop > 0 and lowest_epoch + early_stop < i + 1:
            print("There is no improvement during last %d epochs." % early_stop)
            break

print("The best validation loss from epoch %d: %.4e" % (lowest_epoch + 1, lowest_loss))
model.load_state_dict(best_model)    



## model.py 

class Block(nn.Module):
    
    def __init__(self,
                 input_size,
                 output_size,
                 use_batch_norm=True,
                 dropout_p=.4):
        self.input_size = input_size
        self.output_size = output_size
        self.use_batrch_norm = use_batch_norm
        self.dropout_p = dropout_p
        
        super().__init__()
        
        def get_regularization(use_batch_norm, size):
            return nn.BatchNorm1d(size) if use_batch_norm else nn.Dropout(dropout_p)
        
        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(),
            get_regularization(use_batch_norm, output_size)    
        )
    
    def forward(self, x):
        # |x| = (batch_size, input_size)
        y = self.block(x)
        # |y| = (batch_size, output_size)
        
        return y

class ImageClassifier(nn.Module):
    
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes=[500,400,300,200,100],
                 use_batch_norm=True,
                 dropout=.3):
        
        super().__init__()
        
        assert len(hidden_sizes) > 0, "you need to specify hidden layers"
        
        last_hidden_size = input_size 
        
        for hidden_size in hidden_sizes:
            blocks += [Block(
                last_hidden_size,
                hidden_size,
                use_batch_norm,
                dropout
            )]
        
        self.layers = nn.Sequential(
            *blocks,
            nn.Linear(last_hidden_size, output_size),
            nn.LogSoftmax(dim=-1),
        )
    
    def forward(self, x):
        # |x| = (batch_size, input_size)        
        y = self.layers(x)
        # |y| = (batch_size, output_size)
        
        return y
       
 # trainer.py 
from copy import deepcopy
import numpy as np
import torch

class Trainer():
    
    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit 
        
        # super().__init__() 
    
    def _batchify(self, x, y, batch_size, random_split=True):
        if random_split:
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indices)
            y = torch.index_select(y, dim=0, index=indices)
        
        x = x.split(batch_size, dim=0)
        y = y.split(batch_size, dim=0)
        
        return x, y
    
    def _train(self, x, y, config):
        self.model.train()
        
        x, y = self._batchify(x, y, config.batch_size)
        total_loss = 0
        
        for i, (x_i, y_i) in enumerate(zip(x, y)):
            y_hat_i = self.model(x_i)
            loss_i = self.crit(y_hat_i, y_i.squeeze())
            
            # Initialize the gradient of the model.
            self.optimizer.zero_grad()
            loss_i.backward()
            
            self.optimizer.step()
            
            if config.verbose >= 2:
                print("Train Iteration(%d/%d): loss=%.4e" % (i+1, len(x), float(loss_i)))
            total_loss += float(loss_i)
        
        return total_loss / len(x)
    
    def _validate(self, x, y, config):
        # Turn evaluation mode on.
        self.model.eval()
        
        # Turn on the no_grad mode to make more efficiently.
        with torch.no_grad():
            x, y = self._batchify(x, y, config.batch_size, random_split=False)
            total_loss = 0
            
            for i, (x_i, y_i) in enumerate(zip(x,y)):
                y_hat_i = self.model(x_i)
                loss_i = self.crit(y_hat_i, y_i.squeeze())
                
                if config.verbose >= 2:
                   print("Valid Iteration(%d/%d): loss=%.4e" % (i+1, len(x), float(loss_i)))

                total_loss += float(loss_i)
                
            return total_loss / len(x)
    
    def train(self, train_data, valid_data, config):
        lowest_loss = np.inf
        best_model = None
        
        for epoch_index in range(config.n_epochs):
            train_loss = self._train(train_data[0], train_data[1], config)
            valid_loss = self._validate(valid_data[0], valid_data[1], config)
            
            # You must use deep copy to take a snapshot of current best weights.
            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())
                
            print("Epoch(%d/%d): train_loss=%.4e valid_loss=%.4e lowest_loss=%.4e" % (
                epoch_index + 1, 
                config.n_epochs,
                train_loss,
                valid_loss,
                lowest_loss,
            ))
        
        # Restore to best model.
        self.model.load_state_dict(best_model)
    

 
 