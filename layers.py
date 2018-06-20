import torch
import torch.nn as nn
import torch.nn.functional as F


class StackedBiLSTM(nn.Module):
    '''
    stacking LSTMs together to form a stacked LSTM 
    each LSTM taking in the front LSTM's output as input
    the last LSTM generates the final result
    '''
    def __init__(self, input_size, hidden_size, num_layers,
            dropout_rate=0):
        super(StackedBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.rnns = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
        bidirectional=True)
    
    def forward(self, x):
        '''
        input: 
            x: [batch * len * dim]   a sequence
        output 
            x_encoded: [batch * len * dim_encoded]
        '''
        # Forward
        output, _ = self.rnns(x)
        output = output[:,:,:self.hidden_size] + output[:,:,self.hidden_size:]

        # Dropout on output layer
        if self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
                            
        return output



class SeqAttnMatch(nn.Module):
    '''
    Given sequences X and Y, match sequence Y to each element in X.
    '''
    def __init__(self, input_size):
        super(SeqAttnMatch, self).__init__()
        self.linear = nn.Linear(input_size, input_size) #shared weighted
    
    def forward(self, x, y):
        '''
        inputs: 
        x: [batch * len_p * dim] passage
        y: [batch * len_q * dim] question
        '''
        x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
        x_proj = F.relu(x_proj)
        y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
        y_proj = F.relu(y_proj)

        scores = x_proj.bmm(y_proj.transpose(2, 1)) #[batch len_p len_q]

        alpha_flat = F.softmax(scores.view(-1, y.size(1)), dim=-1) 
        alpha = alpha_flat.view(-1, x.size(1), y.size(1)) #[batch len_p len_q]

        match_seq = alpha.bmm(y) # [batch len_p dim]
        return match_seq
    

class BilinearSeqAttn(nn.Module):
    '''
    A bilinear attention layer over a sequence X W y
    '''
    def __init__(self, x_size, y_size):
        super(BilinearSeqAttn, self).__init__()
        self.linear = nn.Linear(y_size, x_size)
    
    def forward(self, x, y):
        '''
        inputs:
            x: [batch * len * dim_x]   passage
            y: [batch * dim_y]   question
        
        output:
            alpha: [batch * len]  probabilities
        '''
        Wy = self.linear(y)
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        if self.training:
            alpha = F.log_softmax(xWy ,dim=-1)
        else:
            alpha = F.softmax(xWy ,dim=-1)
        return alpha


class LinearSeqAttn(nn.Module):
    '''
    self attention for a sequence
    '''
    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        '''
        input x: [batch * len * dim]
        output alpha [batch * len]  return weights
        '''
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        alpha = F.softmax(scores, dim=-1)
        return alpha 


'''
def test_StackedBiLSTM():
    batch = 32
    len = 10
    dim = 300
    hdim = 128
    num_layers = 3
    lstm = StackedBiLSTM(dim, hdim, num_layers, 0.4)

    input = torch.randn(batch, len, dim)
    output = lstm(input)
    print(output.shape)

def test_SeqAttnMatch():
    dim = 300
    batch = 32
    len_p = 100
    len_q = 10
    attn = SeqAttnMatch(dim)
    x = torch.randn(batch, len_p, dim)
    y = torch.randn(batch, len_q, dim)
    output = attn(x,y)
    print(output.shape) # [32, 100, 300]

if __name__=='__main__':
    test_StackedBiLSTM()
'''

        

        
        
