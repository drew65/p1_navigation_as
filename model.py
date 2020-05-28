import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    '''
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    '''
    def __init__(self, state_size, action_size, seed, hidden_layer_sizes=[512, 256, 128, 64]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(int(seed))
        "*** YOUR CODE HERE ***"
        drop_p = 0.2
        self.hidden_layer_sizes = hidden_layer_sizes
        print(self.hidden_layer_sizes)
        layer_sizes = zip(self.hidden_layer_sizes[:-1], self.hidden_layer_sizes[1:])
        # Create ModuleList and add input layer
        self.input_layer = nn.Linear(state_size, self.hidden_layer_sizes[0])
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.dropout = nn.Dropout(p=0.2)
        self.output_layer = nn.Linear(self.hidden_layer_sizes[-1], action_size)
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x, device=self.device, dtype=torch.float32)
        x = x.unsqueeze(0)
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        
        return x