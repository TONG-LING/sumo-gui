'''
Created on 04/08/2014

@author: Gabriel de O. Ramos <goramos@inf.ufrgs.br>
'''
from turtle import done

from torch import nn
from agent_core import Learner
import torch.optim as optim
from replay import ReplayBuffer
import torch
from DQN import DQN
import random
import numpy as np
import math
import os
from datetime import datetime
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QLearner(Learner):

    def __init__(self, name, env, starting_state, goal_state, alpha, gamma, model, replay,
                 target_update, eps_start, eps_end, eps_decay, input_dim, output_dim, batch_size, network_file, device_override=None):
        # 在构造函数开始处添加：
        if device_override is not None:
            global device
            device = device_override
            self.device = device_override
        else:
            self.device = device

        super(QLearner, self).__init__(name, env, self)
        self.output_dim = output_dim
        self.train = None
        self._starting_state = starting_state
        self._goal_state = goal_state
        self.cuda = (device.type == 'cuda')  # 根据device自动设置

        self.steps_done = 0
        # 根据模式设置不同的探索参数
        if model == 'eval':
            self.eps_start = eps_end  # 评估模式使用较低的探索率
            self.eps_end = eps_end
            self.eps_decay = eps_decay
        else:
            self.eps_start = eps_start
            self.eps_end = eps_end
            self.eps_decay = eps_decay

        self._alpha = alpha
        self._gamma = gamma
        self.model = model
        self.replay = replay
        self.target_update = target_update
        self.gamma = gamma

        self.n_actions = output_dim
        self.batch_size = batch_size
        self.random_threshold = 0.6

        self.min_replay_size = 32
        self.per_beta = 0.4
        self.per_beta_frames = 200000

        self.network_file = network_file
        self.policy_net = DQN(input_dim, output_dim).to(device)
        self.target_net = DQN(input_dim, output_dim).to(device)


        self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=0.001)



        if network_file:
            self.policy_net.load_state_dict(torch.load(network_file, map_location=device))
            self.policy_net.eval()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.learn_steps = 0
        self.best_reward = float('-inf')
        self.best_loss = float('inf')
        self.best_model_path = config.RL_MODEL_PATH
        self.best_weights_path = f'{config.WEIGHTS_DIR}/best_weights_{name}.pth'

        # Add a variable to store the last action
        self.last_action = None

    def get_action_randomly(self):
        action = 0 if random.random() < self.random_threshold else 1
        return action


    def get_optim_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device)

        with torch.no_grad():
            q_value = self.policy_net(state)

        _, action_index = torch.max(q_value, dim=0)
        action = action_index.cpu().item()

        return action

    def get_action(self, state):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if self.model == 'train' and random.random() <= eps_threshold:
            with torch.no_grad():
                act = self.get_action_randomly()
        else:
            with torch.no_grad():
                act = self.get_optim_action(state)
        return act

    def save_best_model(self):
        torch.save(self.model.state_dict(), self.best_model_path)
        print(f"Saved best model parameters to {self.best_model_path}")

    def load_best_model(self):
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
            print(f"Loaded best model parameters from {self.best_model_path}")
        else:
            print(f"No best model found at {self.best_model_path}")

    def update_best_reward(self, current_reward):
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            self.save_best_model()

    def save_model(self, filename='best_model3.pth'):
        torch.save(self.policy_net.state_dict(), filename)

    def set_train(self):
        self.train = True
        self.model.train()

    def set_eval(self):
        self.train = False
        self.model.eval()

    def learn(self, experiences):
        if self.model == 'train':
            loss_fn = nn.MSELoss()

            if self.replay.steps_done <= 10:
                return

            batch,_,_ = self.replay.sample(self.batch_size)


            state_batch = torch.cat(batch.state).float().to(device)
            action_batch = torch.cat(batch.action).view(self.batch_size, 1).to(device)
            next_state_batch = torch.cat(batch.next_state).float().to(device)
            reward_batch = torch.cat(batch.reward).view(self.batch_size, 1).float().to(device)
            # print(f"state:{state_batch},next_state:{next_state_batch},action:{action_batch},reward:{reward_batch}")

            num_actions = self.policy_net(state_batch).size(1)
            if torch.any(action_batch >= num_actions):
                print("Invalid action index found in action_batch!")
                print("action_batch:", action_batch)
                return

            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

            # Double DQN 的目标Q值计算：选择动作使用 policy_net，评估动作使用 target_net
            with torch.no_grad():
                argmax_actions = self.policy_net(next_state_batch).max(1)[1].view(self.batch_size, 1)  # 通过行为网络选择最大动作
                next_state_values = self.target_net(next_state_batch).gather(1, argmax_actions)  # 使用目标网络计算Q值
                expected_state_action_values = reward_batch + self.gamma * next_state_values
            # #DQN
            # with torch.no_grad():
            #     argmax_action = self.policy_net(next_state_batch).max(1)[1].view(self.batch_size, 1)
            #     expected_state_action_values = reward_batch + self.gamma * self.target_net(next_state_batch).gather(1, argmax_action)

            loss = loss_fn(expected_state_action_values,state_action_values)

            self.optimizer.zero_grad()
            loss.backward()

            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)

            self.optimizer.step()
            self.learn_steps += 1
            # if self.learn_steps % self.target_update == 0:
            #     self.target_net.load_state_dict(self.policy_net.state_dict())
            #     time = str(datetime.now()).split('.')[0]
            #     time = time.replace('-', '').replace(' ', '_').replace(':', '')
            #     torch.save(self.policy_net.state_dict(), 'weights/weights_{0}_{1}.pth'.format(time, self.learn_steps))

            if self.learn_steps % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if self.learn_steps == 1 or loss.item() < self.best_loss:
                # print(f"[✔] learn_step={self.learn_steps}, loss={loss.item():.4f} → 保存模型")
                self.best_loss = loss.item()
                torch.save(self.policy_net.state_dict(), self.best_model_path)



    def act_last(self, state, tlID):
        action = self.get_action(state)
        self.last_action = action  # Store the last action
        return state, action

    def feedback_last(self, reward, next_state, state):
        # 这里只收集经验，不进行学习
        if isinstance(state, list):
            state = torch.tensor(state, dtype=torch.float32).to(device)

        if isinstance(next_state, list):
            next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
            
        # 直接返回经验元组，不调用learn
        return (state.cpu().numpy().tolist(), self.last_action, next_state.cpu().numpy().tolist(), reward)

    def batch_learn(self, experiences_buffer):
        if self.model != 'train' or len(experiences_buffer) == 0:
            return

        # Step 1: 把本轮经验写入重放池，并用 TD-error 初始化优先级
        for (state, action, next_state, reward) in experiences_buffer:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                ns = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
                a = torch.tensor(action, dtype=torch.long).view(1, 1).to(device)
                r = torch.tensor(reward, dtype=torch.float32).view(1, 1).to(device)

                q_sa = self.policy_net(s).gather(1, a)
                a_next = self.policy_net(ns).max(1)[1].view(1, 1)
                q_next = self.target_net(ns).gather(1, a_next)
                target = r + self.gamma * q_next
                td_error = (target - q_sa).item()

            self.replay.add(state, next_state, action, reward, td_error)

        # Step 2: 从重放池按优先级采样进行训练（重要性采样权重修正 + 优先级回写）
        min_replay_size = getattr(self, 'min_replay_size', 32)
        if self.replay.size < min_replay_size:
            return

        gradient_steps = math.ceil(len(experiences_buffer) / self.batch_size)
        for _ in range(gradient_steps):
            beta = getattr(self, 'per_beta', 0.4)
            sample = self.replay.sample(self.batch_size, beta=beta)
            if sample is None:
                break
            batch, weights, indices = sample

            state_batch = torch.cat(batch.state).float().to(device)
            action_batch = torch.cat(batch.action).view(-1, 1).to(device)
            next_state_batch = torch.cat(batch.next_state).float().to(device)
            reward_batch = torch.cat(batch.reward).view(-1, 1).to(device)
            weights = weights.to(device)  # 确保weights在正确设备上

            q_sa = self.policy_net(state_batch).gather(1, action_batch)
            with torch.no_grad():
                argmax_actions = self.policy_net(next_state_batch).max(1)[1].view(-1, 1)
                q_next = self.target_net(next_state_batch).gather(1, argmax_actions)
                target = reward_batch + self.gamma * q_next

            td_errors = target - q_sa
            loss = (weights * td_errors.pow(2)).mean()

            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            # 回写新的优先级
            self.replay.update_priorities(indices, td_errors.detach().squeeze(1).abs().cpu().numpy())

            self.learn_steps += 1
            if self.learn_steps % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            if self.learn_steps == 1 or loss.item() < self.best_loss:
                self.best_loss = loss.item()
                torch.save(self.policy_net.state_dict(), self.best_model_path)
                # print("更新Q网络")

        return self.learn_steps

    def sequential_learn_current_epoch(self, experiences_buffer):
        """
        只使用本次epoch收集的经验进行顺序学习，不使用重放池
        """
        if self.model != 'train' or len(experiences_buffer) == 0:
            return
        
        
        # 将经验按收集顺序进行学习
        for i, (state, action, next_state, reward) in enumerate(experiences_buffer):
            # 转换为tensor
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            ns = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
            a = torch.tensor(action, dtype=torch.long).view(1, 1).to(device)
            r = torch.tensor(reward, dtype=torch.float32).view(1, 1).to(device)
            
            # 计算Q值和目标值
            q_sa = self.policy_net(s).gather(1, a)
            with torch.no_grad():
                argmax_action = self.policy_net(ns).max(1)[1].view(1, 1)
                q_next = self.target_net(ns).gather(1, argmax_action)
                target = r + self.gamma * q_next
            
            # 计算损失
            loss = nn.functional.mse_loss(q_sa, target)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            
            self.optimizer.step()
            
            self.learn_steps += 1
            
            # 更新target网络
            if self.learn_steps % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # 保存最佳模型
            if self.learn_steps == 1 or loss.item() < self.best_loss:
                self.best_loss = loss.item()
                torch.save(self.policy_net.state_dict(), self.best_model_path)
                if i % 50 == 0:  # 每50步打印一次
                    #print(f"{i+1}/{len(experiences_buffer)}, Loss: {loss.item():.4f}")
                    pass
        return self.learn_steps

    def beta_by_frame(self, frame_idx, beta_start=None, beta_frames=None):
        if beta_start is None:
            beta_start = getattr(self, 'per_beta', 0.4)
        if beta_frames is None:
            beta_frames = getattr(self, 'per_beta_frames', 200000)
        return min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)


