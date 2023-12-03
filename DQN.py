import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from utils.bic import *
from utils.env import *
from deep_q_learning.model import QNetwork

class QLearningAgent:
    def __init__(self, env, predict_network, target_network, memory,gamma = 0.99, device = torch.device("cpu")):
        self.env = env
        self.predict_network = predict_network
        self.target_network = target_network
        self.optimizer = optim.Adam(predict_network.parameters(), lr=0.001)
        self.memory = memory
        self.gamma = gamma
        self.device = device
        self.plot_list = []
        self.loss_list = []


    def optimize_model(self, BATCH_SIZE):
        if len(self.memory) < BATCH_SIZE:
            return
        
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        total_loss = 0
        for element in transitions:
            state_tensor, action, new_state_tensor, reward = element.state, element.action, element.next_state, element.reward
            
            current_q_value = self.predict_network(state_tensor)[action]
            with torch.no_grad():
                target_q_value = reward + self.gamma * torch.max(self.target_network(new_state_tensor)).item()
                target_q_value = torch.tensor(target_q_value, dtype = torch.float).to(self.device)
            total_loss += nn.functional.mse_loss(current_q_value, target_q_value)

        # Optimize the model
        self.optimizer.zero_grad()

        total_loss.backward()
        # In-place gradient clipping
        #torch.nn.utils.clip_grad_value_(self.predict_network.parameters(), 100)
        self.optimizer.step()
    def train(self,base_estimate, episodes=1000, update_rate = 0.1, c = 1, epsilon = 0.5, batch_size = 10):
        space = self.env.n
        count = [1 for i in range (self.env.action_space.n)]
        timestep = 0
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                epsilon -= 0.5*episode/episodes
                timestep += 1
                # Convert state to tensor for PyTorch
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                # Epsilon-greedy strategy for action selection
                if random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    q_values = self.predict_network(state_tensor)
                    confidence_bound = np.array([timestep for i in range (self.env.action_space.n)])
                    confidence_bound = np.divide(confidence_bound, np.array(count))
                    confidence_bound = torch.tensor(np.sqrt(confidence_bound), dtype = torch.float).to(self.device)
                    action = torch.argmax(q_values + confidence_bound).item()
                count[int(action)] += 1
                # Take action and observe new state and reward
                new_state, bic_score, done, _ = self.env.step(action, base_estimate)
                
                # Calculate BIC score based change as reward
                old_bic_score = calculate_bic_score(state, base_estimate)
                new_bic_score = calculate_bic_score(new_state, base_estimate)

                reward =  new_bic_score - old_bic_score
                # Convert new state to tensor
                new_state_tensor = torch.FloatTensor(new_state).unsqueeze(0).to(self.device)

                self.memory.push(state_tensor, action, new_state_tensor, reward)

                # Calculate target Q-value
                with torch.no_grad():
                  target_q_value = reward + self.gamma * torch.max(self.target_network(new_state_tensor)).item()
                target_q_value = torch.tensor(target_q_value, dtype = torch.float)
                # Get current Q-value

                current_q_value = self.predict_network(state_tensor)[action]
                #print(nn.functional.mse_loss(float(current_q_value), target_q_value))
                # Calculate loss
                loss = nn.functional.mse_loss(current_q_value.to(self.device), target_q_value.to(self.device))

                self.loss_list.append(float(loss))

                # Backpropagation
                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                state = np.copy(new_state)
                total_reward += reward
            self.plot_list.append(total_reward)
            if episode % 10 == 0:
                
                self.optimize_model(batch_size)
                target_net_state_dict = self.target_network.state_dict()
                predict_net_state_dict = self.predict_network.state_dict()
                for key in predict_net_state_dict:
                    target_net_state_dict[key] = predict_net_state_dict[key]* update_rate + target_net_state_dict[key]*(1- update_rate)
                self.target_network.load_state_dict(target_net_state_dict)
            print(f"Episode {episode}, Total Reward: {total_reward}")

if __name__ == "__main__":
    data = pd.read_csv("Asia.csv")
    data = data.drop('Unnamed: 0', axis=1)
    base_estimate = BaseEstimator(data)

    epsilon = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CausalGraphEnv(n=8)
    predict_network = QNetwork(n=8).to(device)
    target_network = QNetwork(n=8).to(device)
    memory = ReplayMemory(1000)
    agent = QLearningAgent(env, predict_network, target_network, memory, gamma = 0.9)

    agent.train(base_estimate ,episodes=100, update_rate = 0.05 )