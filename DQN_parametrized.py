import torch
import torch.nn as nn
import torch.nn.functional as F

import optuna


class DQN_parametrized(nn.Module):
    """ Implements a Q-Learning convolutional neural network with parameters.

    Args:
            height (int): height of the input image
            width (int): width of the input image
            n_actions (int): number of output dimensions (actions)

            trial : optuna trial for hyperparameters tuning
    """

    def __init__(self, height, width, n_actions, trial):
        # TODO make modular
        super(DQN_parametrized, self).__init__()

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2, padding = 0):
            return (size + 2*padding - (kernel_size - 1) - 1) // stride + 1

        #Return a activation function by name
        def get_activation_function(act_name):
            if act_name == 'relu':
                return nn.ReLU()
            if act_name == 'selu':
                return nn.SELU()
            if act_name == 'hardswish':
                return nn.Hardswish()


        ### CONVOLUTIONS ###
        n_conv_blocks = trial.suggest_int('n_conv_blocks', 1, 3)

        # heigt and width have to be greather than or equal to 40 with those parameters
        self.conv = []
        self.conv_activation = []
        self.bn = []
        convw = width
        convh = height
        for i in range(n_conv_blocks):
            kernel_size = trial.suggest_int(f'kernel_size_{i}', 2, 4)
            stride = trial.suggest_int(f'stride_{i}', 1, 3)
            padding = trial.suggest_int(f'padding_{i}', 0, 1)
            in_channel = 3 if i==0 else out_channel
            out_channel = trial.suggest_int(f'out_channel_{i}', 3, 12)

            convw = conv2d_size_out(convw, kernel_size, stride, padding)
            convh = conv2d_size_out(convh, kernel_size, stride, padding)

            self.conv.append(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding))

            act_name = trial.suggest_categorical(f'activation_{i}', ['relu', 'selu', 'hardswish'])
            self.conv_activation.append(get_activation_function(act_name))

            self.bn.append(nn.BatchNorm2d(out_channel))

        ### LINEAR LAYERS ###
        n_lin_blocks_1 = trial.suggest_int('nlin1blocks', 1, 3)  
        in_features = out_channel * convw * convh
        self.lin_1 = []
        self.lin_activation_1 = []
        for i in range(n_lin_blocks_1):
            out_features = trial.suggest_int(f'out_nlin_features_1_{i}', 256, 1024)
            self.lin_1.append(nn.Linear(in_features, out_features))
            in_features = out_features

            act_name = trial.suggest_categorical(f'nlin_activation_1_{i}', ['relu', 'selu', 'hardswish'])
            self.lin_activation_1.append(get_activation_function(act_name))


        ### HIDDEN SPACE ###
        #Linear layers for getting the expected value and the variance
        hidden_size = trial.suggest_int('hidden_size', in_features // 2, in_features)
        self.exp_val = nn.Linear(in_features, hidden_size)
        self.log_var = nn.Linear(in_features, hidden_size)

        ### LINEAR LAYERS ###
        n_lin_blocks_2 = trial.suggest_int('n_lin2_blocks', 0, 2)  
        self.lin_2 = []
        self.lin_activation_2 = []
        in_features = hidden_size
        for i in range(n_lin_blocks_2):
            out_features = trial.suggest_int(f'out_lin2_features_2_{i}', 16, 512)
            self.lin_2.append(nn.Linear(in_features, out_features))
            in_features = out_features

            act_name = trial.suggest_categorical(f'lin2_activation_2_{i}', ['relu', 'selu', 'hardswish'])
            self.lin_activation_2.append(get_activation_function(act_name))

        self.lin_2.append(nn.Linear(in_features, n_actions))


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        """Returns a matrix of scores associated with each action.

        Args:
            x (batch_size * height * width tensor): a batch of images to process;
                batch_size can be 1 to determine the next action.

        Returns:
            batch_size * n_actions tensor: action scores
        """
        for i in range(len(self.conv)):
            x = self.conv[i](x)
            x = self.conv_activation[i](x)
            x = self.bn[i](x)
        
        x = torch.flatten(x, start_dim = 1)

        for i in range(len(self.lin_1)):
            x = self.lin_1[i](x)
            x = self.lin_activation_1[i](x)

        mu  = self.exp_val(x)
        log_var = self.log_var(x)

        #reparameterization
        x = mu + torch.exp(log_var) * torch.randn(list(log_var.shape))

        for i in range(len(self.lin_2) - 1):
            x = self.lin_2[i](x)
            x = self.lin_activation_2[i](x)

        return self.lin_2[-1](x)

def ojective(trial):
    size = 80
    model = DQN_parametrized(size, size, 4, trial)

    #training loop
    #...

    #define a matric that is used to hyperparameters tuning
    #metric =  ...

    #evaluation over a test dataset or test session

    return metric
'''
Create study:
study = optuna.create_study(direction='minimize')

Tuning:
study.optimize(objective, n_trials=200)
'''



'''
An exemple of objective function for tests
def objective(trial):
    size = 80
    model = DQN_parametrized(size, size, 4, trial)
    loss_func = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr = 0.001)
    for epoch in range(2):
        inp = torch.randn(1, 3, size, size)
        tar = torch.ones(1,4)
        opt.zero_grad()

        pred = model(inp)
        loss = loss_func(pred, tar)
        loss.backward()
        opt.step()
    

    with torch.no_grad():
        inp = torch.randn(1, 3, size, size)
        tar = torch.ones(1,4)
        pred = model(inp)
        mse = loss_func(pred, tar)

    del model
    gc.collect()
    return mse
'''


