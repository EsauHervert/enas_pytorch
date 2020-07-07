## Modifications of the code provided in:
    ## https://github.com/TDeVries/enas_pytorch
## Modifications made by Esau Alain Hervert Hernandez

import torch
import torch.nn as nn

from torch.autograd import Variable
    ## Deprecated, torch.tensors work as variables
    ## https://pytorch.org/docs/stable/autograd.html

from torch.distributions.categorical import Categorical

## This is the controller that will generate the child architectures.
## This will use Reinforcement Learning (RL) to train and learn to output modifications to the child s.t. we maximize accuracy
class Controller(nn.Module):
    '''
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py
    '''
    ## The "macro" refers to the fact that we will be looking for a CNN architecture as a whole rather than in cells, as in "micro".
    def __init__(self,
                 search_for="macro",
                 search_whole_channels=True,
                 num_layers=12,
                 num_branches=6,
                 out_filters=36,
                 lstm_size=32,
                 lstm_num_layers=2,
                 tanh_constant=1.5,
                 temperature=None,
                 skip_target=0.4,
                 skip_weight=0.8):
        super(Controller, self).__init__()
        ## The controller will contain LSTM cells.

        self.search_for = search_for
        self.search_whole_channels = search_whole_channels
        self.num_layers = num_layers
        self.num_branches = num_branches
        self.out_filters = out_filters

        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.tanh_constant = tanh_constant
        self.temperature = temperature

        self.skip_target = skip_target
        self.skip_weight = skip_weight

        self._create_params()

    ## Here we create the parameters of the Controller.
    def _create_params(self):
        '''
        https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L83
        '''
        ## This is the lstm portion of the network.
        ## We have that the input and hidden state will be both of size "lstm_size". 
            ## The inputs are of size 32 both.
        ## We will have that this will be a stacked LSTM where the number of layers stacked is given by "lstm_num_layers".
            ## The stacked LSTM will take outputs of previous LSTM cells and use them as inputs.
            ## This network will use 2 LSTM cells stacked.
        self.w_lstm = nn.LSTM(input_size=self.lstm_size,
                              hidden_size=self.lstm_size,
                              num_layers=self.lstm_num_layers)
        ## https://pytorch.org/docs/master/generated/torch.nn.Embedding.html
            ## "A simple lookup table that stores embeddings of a fixed dictionary and size."
            ## "This module is often used to store word embeddings and retrieve them using indices."
            ## "The input to the module is a list of indices, and the output is the corresponding word embeddings."
        self.g_emb = nn.Embedding(1, self.lstm_size)  # Learn the starting input

        ## Not sure what this does...
        if self.search_whole_channels:
            self.w_emb = nn.Embedding(self.num_branches, self.lstm_size)
            self.w_soft = nn.Linear(self.lstm_size, self.num_branches, bias=False)
        else:
            assert False, "Not implemented error: search_whole_channels = False"
        
        ## Feedforwards Neural Network
            ## Here we will have a FNN which will take in the output of the LSTM.
            ## The first layer will take the output and return a hidden layer of the same size, using no bias.
            ## The second layer will do the same.
            ## The last layer will take in the second hidden state and return one value.
        self.w_attn_1 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.w_attn_2 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)

        self._reset_params()

    def _reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1) ## Will make the weights all in the range (-0.1, 0.1)

        ## Initializing weights to be in (-0.1, 0.1)
        nn.init.uniform_(self.w_lstm.weight_hh_l0, -0.1, 0.1)
        nn.init.uniform_(self.w_lstm.weight_ih_l0, -0.1, 0.1)

    def forward(self):
        '''
        https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L126
        '''
        ## Since we have that LSTMs require two inputs at each time step, x_i and h_i, we have that at time t = 0,
        ## the input for the hidden state will be h_0 = vector(0)
        h0 = None  # setting h0 to None will initialize LSTM state with 0s
        

        ## Not sure what this does...
        anchors = []
        anchors_w_1 = []

        arc_seq = {} ## This will contain the sequence of architectures that are generated.
        entropys = []
        log_probs = []
        skip_count = [] ## Counts how any skip connects there are.
        skip_penaltys = []

        inputs = self.g_emb.weight
        skip_targets = torch.tensor([1.0 - self.skip_target, self.skip_target]).cuda()

        for layer_id in range(self.num_layers):
            if self.search_whole_channels:
                inputs = inputs.unsqueeze(0) ## Will return a tensor with dimension 1xdim(inputs).
                
                ## Feed in the input tensor which specifies the input and the hidden state from the previous step
                output, hn = self.w_lstm(inputs, h0) 
                
                output = output.squeeze(0) ## Will return a tensor with dimension 1xdim(inputs).
                h0 = hn ## Have the hidden output be the initial hidden input for the next step.

                logit = self.w_soft(output) ## Using the output and passing it through a linear layer.
                
                ## Here the logits go through scaling and then through tanh which will return values in (-tan_constant, tan_constant)
                if self.temperature is not None:
                    logit /= self.temperature
                if self.tanh_constant is not None:
                    logit = self.tanh_constant * torch.tanh(logit)

                ## Create a distribution to sample from.
                    ## The logits are generated from the weights of the controller.
                    ## They determine which actions have what probability of being taken.
                branch_id_dist = Categorical(logits=logit)
                branch_id = branch_id_dist.sample() ## Sample from the distribution.

                ## Save the modification to the sequence of architectures.
                arc_seq[str(layer_id)] = [branch_id]

                ## Here we have the log probabilities and entropy of the distripution.
                log_prob = branch_id_dist.log_prob(branch_id)
                log_probs.append(log_prob.view(-1))
                entropy = branch_id_dist.entropy()
                entropys.append(entropy.view(-1))

                inputs = self.w_emb(branch_id)
                inputs = inputs.unsqueeze(0)
            else:
                # https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L171
                assert False, "Not implemented error: search_whole_channels = False"

            ## Now, we pass the input from the modification.
            ## We pass the hidden state from the previous use of the LSTM.
            output, hn = self.w_lstm(inputs, h0)
            output = output.squeeze(0) #

            ## For the case where the layer is not the input layer.
            if layer_id > 0:
                query = torch.cat(anchors_w_1, dim=0)
                query = torch.tanh(query + self.w_attn_2(output))
                query = self.v_attn(query)
                logit = torch.cat([-query, query], dim=1)
                if self.temperature is not None:
                    logit /= self.temperature
                if self.tanh_constant is not None:
                    logit = self.tanh_constant * torch.tanh(logit)

                skip_dist = Categorical(logits=logit)
                skip = skip_dist.sample()
                skip = skip.view(layer_id)

                arc_seq[str(layer_id)].append(skip)

                skip_prob = torch.sigmoid(logit)
                kl = skip_prob * torch.log(skip_prob / skip_targets)
                kl = torch.sum(kl)
                skip_penaltys.append(kl)

                log_prob = skip_dist.log_prob(skip)
                log_prob = torch.sum(log_prob)
                log_probs.append(log_prob.view(-1))

                entropy = skip_dist.entropy()
                entropy = torch.sum(entropy)
                entropys.append(entropy.view(-1))

                # Calculate average hidden state of all nodes that got skips
                # and use it as input for next step
                skip = skip.type(torch.float)
                skip = skip.view(1, layer_id)
                skip_count.append(torch.sum(skip))
                inputs = torch.matmul(skip, torch.cat(anchors, dim=0))
                inputs /= (1.0 + torch.sum(skip))

            else:
                inputs = self.g_emb.weight

            anchors.append(output)
            anchors_w_1.append(self.w_attn_1(output))

        self.sample_arc = arc_seq

        entropys = torch.cat(entropys)
        self.sample_entropy = torch.sum(entropys)

        log_probs = torch.cat(log_probs)
        self.sample_log_prob = torch.sum(log_probs)

        skip_count = torch.stack(skip_count)
        self.skip_count = torch.sum(skip_count)

        skip_penaltys = torch.stack(skip_penaltys)
        self.skip_penaltys = torch.mean(skip_penaltys)
