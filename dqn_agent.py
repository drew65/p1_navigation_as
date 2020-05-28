import numpy as np
import random
import os
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 7e-4               # learning rate 
UPDATE_EVERY = 15        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, filepath):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.avarage_score = 0
        self.start_epoch = 0
        self.seed = random.randint(0, seed)
        random.seed(seed)
        print("seed ",seed,"  self.seed ",self.seed)
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, self.seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, self.seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        if filepath: 
            self.load_model(filepath)
        
        
        # Replay memory
        print("buffer size ", BUFFER_SIZE)
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, self.seed)
        print("memory ",self.memory)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                #print("experiences ",experiences)
                self.learn_DDQN(experiences, GAMMA)
                self.t_step = (self.t_step + 1) % UPDATE_EVERY
                if self.t_step == 0:
                    self.update_network(self.qnetwork_local, self.qnetwork_target)
                    

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn_DDQN(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # Get max predicted Q values (for next states) from target model
        Q_targets_next_argmax = self.qnetwork_local(next_states).squeeze(0).detach().max(1)[1].unsqueeze(1)
        #Q_targets_next0 = self.qnetwork_target(next_states).squeeze(0).detach()
        #Q_targets_next = Q_targets_next0.max(1)[0].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).squeeze(0).gather(1, Q_targets_next_argmax)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).squeeze(0).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        #self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)              
        
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next0 = self.qnetwork_target(next_states).squeeze(0).detach()
        Q_targets_next = Q_targets_next0.max(1)[0].unsqueeze(1)
        
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).squeeze(0).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        #self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def save_model(self, filepath, epoch, score, last=False):
        checkpoint = {'input_size': self.state_size,
              'output_size': self.action_size,
              'hidden_layers': [each.in_features for each in self.qnetwork_local.hidden_layers],
              'state_dict': self.qnetwork_local.state_dict(),
              'optimizer_state_dict': self.optimizer.state_dict(),
              'epoch': epoch,
              'avarage_score': score}
        checkpoint['hidden_layers'].append(self.qnetwork_local.hidden_layers[-1].out_features)
        torch.save(checkpoint, filepath)
        if last:
            torch.save(self.qnetwork_local.state_dict(),'{}_state_dict_{}.pt'.format(last,epoch))
        #print("checkpoint['hidden_layers'] ",checkpoint['hidden_layers'])
    
    def load_model(self, filepath):
        print("seed ",self.seed)
        if os.path.isfile(filepath):
            print("=> loading checkpoint '{}'".format(filepath))
            checkpoint = torch.load(filepath)
            print("checkpoint['hidden_layers'] ",checkpoint['hidden_layers'])
            self.qnetwork_local = QNetwork(checkpoint['input_size'],
                             checkpoint['output_size'],
                             self.seed,
                             checkpoint['hidden_layers']).to(device)
            self.qnetwork_local.load_state_dict(checkpoint['state_dict'])
            self.qnetwork_local.to(device)
            self.qnetwork_target = QNetwork(checkpoint['input_size'],
                             checkpoint['output_size'],
                             self.seed,
                             checkpoint['hidden_layers']).to(device)
            self.qnetwork_target.load_state_dict(checkpoint['state_dict'])
            self.qnetwork_target.to(device)
            if 'optimizer_state_dict' in checkpoint: 
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
                print(self.optimizer)
            if 'epoch' in checkpoint: 
                self.start_epoch = checkpoint['epoch']
            if 'avarage_score' in checkpoint:
                self.avarage_score = checkpoint['avarage_score']
            
            print(self.qnetwork_target)
            print(self.optimizer)
        else:
            print("=> no checkpoint found at '{}'".format(filepath))
    
    def update_network(self, local_model, target_model):
        for target , local in zip(target_model.parameters(), local_model.parameters()):
            target.data.copy_(local.data)
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)