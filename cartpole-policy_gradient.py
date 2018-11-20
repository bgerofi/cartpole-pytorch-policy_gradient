import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


env = gym.make('CartPole-v1')
env.seed(1)
torch.manual_seed(1)

# Hyperparameters
learning_rate = 0.005
gamma = 0.99


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n

        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, self.action_space, bias=False)

        self.gamma = gamma

        # Episode policy and reward history 
        self.policy_history = []
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

        self.model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        #optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)


    def forward(self, x):
        return self.model(x)


    def select_action(self, state):
        # Select an action (0 or 1) by running policy model and choosing
        # based on the probabilities in state
        probs = self.forward(torch.tensor(state, dtype=torch.float32))
        c = Categorical(probs)
        action = c.sample()

        self.policy_history.append(c.log_prob(action))

        return action


    def update(self):
        R = 0
        rewards = []

        # Discount future rewards back to the present using gamma
        for r in self.reward_episode[::-1]:
            R = r + self.gamma * R
            rewards.insert(0,R)

        # Scale rewards
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())

        # Calculate loss
        loss = torch.sum(torch.mul(torch.stack(self.policy_history),
            rewards).mul(-1), -1)

        # Update network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Save and intialize episode history counters
        self.loss_history.append(loss.item())
        self.reward_history.append(np.sum(self.reward_episode))
        self.policy_history = []
        self.reward_episode= []



policy = Policy()



def main(episodes):
    running_reward = 10
    for episode in range(episodes):
        state = env.reset() # Reset environment and record the starting state
        done = False

        for time in range(1000):
            if False and episode % 10 == 0:
                env.render()

            action = policy.select_action(state)
            # Step through environment using chosen action
            state, reward, done, _ = env.step(action.item())

            # Save reward
            policy.reward_episode.append(reward)
            if done:
                break
        
        # Used to determine when the environment is solved.
        running_reward = (running_reward * 0.99) + (time * 0.01)

        policy.update()

        if episode % 10 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(episode, time, running_reward))

        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, time))
            break


if __name__ == '__main__':
    main(10000)

