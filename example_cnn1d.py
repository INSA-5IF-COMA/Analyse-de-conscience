import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import copy
import sklearn.metrics as metrics
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError
from torchinfo import summary

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, bidirectional=False) #lstm
        self.fc = nn.Linear(hidden_size, 2)
        self.relu = nn.ReLU()
        #self.selu = nn.SELU()
        self.selu = nn.ELU()
        self.sig = nn.Sigmoid()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for next dense layer 
        #out = self.relu(hn)
        #out = self.relu(out) #relu
        #out = self.selu(self.fc(hn)) #Final Output # SD
        out = self.fc(hn)
        out[:,1] = self.sig(out[:,1])
        return out

    def forward_inference(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for next dense layer 
        #out = self.relu(hn)
        #out = self.relu(out) #relu
        #out = self.selu(self.fc(hn)) #Final Output # SD
        out = self.fc(hn)
        out[:,1] = self.sig(out[:,1])
        out[torch.where(out[:,1]<0.8)[0],0] = 0  # set invisible values to 0
        return out


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_length):
        super(RNN, self).__init__()
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #rnn
        self.fc = nn.Linear(hidden_size, 1) #Dense
        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        # Propagate input through LSTM
        output, hn = self.rnn(x, h_0) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for next dense layer 
        out = self.relu(hn)
        out = self.relu(out) #relu
        out = self.relu(self.fc(out)) #Final Output
        return out


class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_length):
        super(CNN, self).__init__()
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.kernel_size1 = 7
        self.kernel_size2 = 7
        self.fc_input_size = hidden_size*(seq_length-self.kernel_size1+1-self.kernel_size2+1)
        self.conv1 = nn.Conv2d(1, hidden_size, (self.kernel_size1,1))   # output size : seq_length - 4
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, (self.kernel_size2,1))   # output size : seq_length - 8
        self.fc = nn.Linear(self.fc_input_size, 2)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
    
    def forward(self,x):
        x = torch.unsqueeze(x, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1, self.fc_input_size) #reshaping the data for next dense layer
        x = self.fc(x)
        x[:,1] = self.sig(x[:,1])
        return x

    def forward_inference(self,x):
        x = torch.unsqueeze(x, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1, self.fc_input_size) #reshaping the data for next dense layer 
        x = self.fc(x)
        x[:,1] = self.sig(x[:,1])
        x[torch.where(x[:,1]<0.8)[0],0] = 0  # set invisible values to 0
        return x



def smape(true, pred):
    with np.errstate(divide='ignore', invalid='ignore'):
        tmp = 2 * np.abs(pred-true) / (np.abs(true) + np.abs(pred))
    tmp[np.isnan(tmp)] = 0
    return np.sum(tmp) / len(tmp) * 100

class Visible_MSE_Loss(nn.Module):
    def __init__(self): #, custom_parameters):
      super(Visible_MSE_Loss, self).__init__()
      #self.custom_parameters = custom_parameters

    def forward(self, pred, true):
      invisible_idx = torch.where(true[:,1]==0)[0]
      pred[invisible_idx,0] = 0.5*(pred[invisible_idx,0]-true[invisible_idx, 0])
      return torch.mean((pred-true)**2)
      #return torch.mean((pred[:,0]-true[:,0])**2) + torch.mean(torch.nn.functional.cross_entropy(pred[:,1], true[:,1]))


def train_model(cell, X_train, y_train, X_test, y_test, original_y_train, num_epochs=1500, lr=0.001, 
                input_size=1, hidden_size=10, num_layers=1):
    if cell == "LSTM":
        model = LSTM(input_size=input_size, hidden_size=hidden_size,
                      num_layers=num_layers, seq_length=X_train.shape[1])
        #summary(model, input_size=X_train.shape)
    elif cell == "CNN":
        model = CNN(input_size=input_size, hidden_size=hidden_size,
                      num_layers=num_layers, seq_length=X_train.shape[1])
    else:
        model = RNN(input_size=input_size, hidden_size=hidden_size,
                      num_layers=num_layers, seq_length=X_train.shape[1])
    print(f"Number of parameters: {sum([param.nelement() for param in model.parameters()])}")

    #criterion = torch.nn.MSELoss()
    criterion = Visible_MSE_Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    min_val_metric = np.inf
    for e in range(num_epochs):
        outputs = model.forward(X_train)
        optimizer.zero_grad() # Zero the gradient at every new epoch
        
        outputs = torch.squeeze(outputs, 1)
        loss = criterion(outputs, y_train)
        loss.backward()

        optimizer.step() # Backpropagation
        if e%100 == 0:
            eval_metric = evaluate(model, X_test, y_test, [metrics.mean_squared_error, metrics.mean_absolute_error, metrics.r2_score, metrics.mean_absolute_percentage_error, smape], original_y_train)
            print(f"Epoch: {e}, loss: {loss.item():1.5f}, test MSE: {eval_metric[0]:1.5f} MAE: {eval_metric[1]:1.5f} R2: {eval_metric[2]:1.5f} MAPE: {eval_metric[3]:1.5f} sMAPE: {eval_metric[4]:1.5f} MASE: {eval_metric[5]:1.5f}")
            # Keep best model
            if eval_metric[0] < min_val_metric:
                min_val_metric = eval_metric[0]
                #print(f"***** Best MSE: {eval_metric[0]:1.5f} MAE: {eval_metric[1]:1.5f} R2: {eval_metric[2]:1.5f} MAPE: {eval_metric[3]:1.5f} sMAPE: {eval_metric[4]:1.5f} MASE: {eval_metric[5]:1.5f}")
                best_model = copy.deepcopy(model)

    eval_metric = evaluate(model, X_test, y_test, [metrics.mean_squared_error, metrics.mean_absolute_error, metrics.r2_score, metrics.mean_absolute_percentage_error, smape], original_y_train)
    print(f"Test MSE: {eval_metric[0]:1.5f} MAE: {eval_metric[1]:1.5f} R2: {eval_metric[2]:1.5f} MAPE: {eval_metric[3]:1.5f} sMAPE: {eval_metric[4]:1.5f} MASE: {eval_metric[5]:1.5f}")
    
    return model, best_model


def evaluate(model, X_test, y_test, metrics=[], y_train=None):
    model.eval()
    result = []
    with torch.no_grad():
        preds = model.forward_inference(X_test)
        preds = torch.squeeze(preds, 1)
        for m in metrics:
          eval_ = m(np.array(y_test), np.array(preds))
          result.append(eval_)
        if y_train is not None:
          mase = MeanAbsoluteScaledError()
          eval_ = mase(np.array(y_test), np.array(preds), y_train = np.array(y_train))
          result.append(eval_)
        else:
          result.append(None)
    model.train()
    return result

def predict(X_ss, y_mm, model):
    pred = model.forward_inference(X_ss) # Forward pass # SD
    #pred = model(X_ss) # Forward pass
    np_pred = pred.data.numpy() # Numpy conversion
    #eval_metric = evaluate(model, X_ss, y_mm)
    return np_pred

