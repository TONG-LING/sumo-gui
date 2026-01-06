import os
import multiprocessing

import filelock
from openpyxl.reader.excel import load_workbook

from traffic_simulator import SUMOTrafficLights
from neural_agent import QLearner
from random_policy import EpsilonGreedy
import datetime
import warnings
from replay import ReplayBuffer

from DQN import DQN
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import numpy as np
import torch.optim as optim
from datetime import datetime
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing as mp
import socket
import torch.nn.functional as F
import ray
import subprocess
import psutil

# 导入配置文件
import config

warnings.filterwarnings(action='ignore')


def parse_state_column(column):
    result = []
    for row in column:
        if pd.isna(row) or row is None:
            continue

        row_str = str(row).strip()

        if row_str.startswith('[') and row_str.endswith(']'):
            row_str = row_str[1:-1]
        elif row_str.startswith('['):
            row_str = row_str[1:]
        elif row_str.endswith(']'):
            row_str = row_str[:-1]

        try:
            values = []
            for x in row_str.split(','):
                x = x.strip()
                if x and x != '':
                    values.append(float(x))

            if values:
                result.append(values)
        except ValueError as e:
            print(f"Warning: Skipping unparseable row data '{row}': {e}")
            continue

    return result


class TrafficDataset(Dataset):
    def __init__(self, states_rewards, conditions, weights):
        self.states = states_rewards
        self.conditions = conditions
        self.weights = weights

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.conditions[idx], self.weights[idx]


# 全局缓存归一化统计量（避免重复计算）
_reward_norm_cache = {"min": None, "max": None}


def compute_reward_normalization_stats(force_recompute=False):
    """
    从DiffData.csv动态计算reward的归一化边界（基于百分位数）
    使用缓存避免重复计算
    """
    global _reward_norm_cache

    # 如果已缓存且不强制重新计算，直接返回
    if not force_recompute and _reward_norm_cache["min"] is not None:
        return _reward_norm_cache["min"], _reward_norm_cache["max"]

    # 读取训练数据
    cpm_path = config.CPM_DATA_PATH
    if not os.path.exists(cpm_path):
        print(f"[WARNING] {cpm_path} not found, using default normalization for unbounded negative rewards")
        # 修正默认值：适配移除clip后的奖励范围（负排队长度可能很大）
        # 使用一个合理的默认范围，后续会被实际数据更新
        _reward_norm_cache["min"] = -500.0  # 假设最坏情况排队500辆车
        _reward_norm_cache["max"] = 0.0  # 最好情况无排队
        return -500.0, 0.0

    try:
        df = pd.read_csv(cpm_path)

        # 解析所有reward列
        all_rewards = []
        for col in ['state_reward_0', 'state_reward_1', 'state_reward_2', 'state_reward_3']:
            if col in df.columns:
                # 从state_reward中提取最后一个元素（reward值）
                rewards = df[col].apply(
                    lambda x: float(str(x).split(',')[-1].strip().rstrip(']')) if pd.notna(x) else 0.0)
                all_rewards.extend(rewards.values)

        all_rewards = np.array(all_rewards)

        # 过滤掉占位0值
        nonzero_rewards = all_rewards[all_rewards != 0]

        if len(nonzero_rewards) == 0:
            print("[WARNING] No non-zero rewards found, using default normalization for unbounded negative rewards")
            reward_min, reward_max = -500.0, 0.0  # 适配移除clip后的范围
        else:
            # 使用配置的百分位数计算边界
            reward_min = float(np.percentile(nonzero_rewards, config.REWARD_PERCENTILE_LOW))
            reward_max = float(np.percentile(nonzero_rewards, config.REWARD_PERCENTILE_HIGH))

            # 如果min和max相同，扩展范围避免除零
            if abs(reward_max - reward_min) < 1e-6:
                reward_min -= 1.0
                reward_max += 1.0
                print(f"[WARNING] Reward range too small, expanded to [{reward_min:.2f}, {reward_max:.2f}]")

            print(f"[NORMALIZATION] Computed reward bounds from {len(nonzero_rewards)} samples:")
            print(f"  P{config.REWARD_PERCENTILE_LOW}: {reward_min:.2f}")
            print(f"  P{config.REWARD_PERCENTILE_HIGH}: {reward_max:.2f}")

        # 缓存结果
        _reward_norm_cache["min"] = reward_min
        _reward_norm_cache["max"] = reward_max

        return reward_min, reward_max

    except Exception as e:
        print(f"[ERROR] Failed to compute normalization stats: {e}")
        print("[WARNING] Using default normalization for unbounded negative rewards")
        _reward_norm_cache["min"] = -500.0  # 适配移除clip后的范围
        _reward_norm_cache["max"] = 0.0
        return -500.0, 0.0


def normalize_rewards(rewards, reward_min=None, reward_max=None):
    """归一化rewards到[-1, 1]范围，用于DDPM训练（兼容torch.Tensor和numpy.ndarray）"""
    if reward_min is None or reward_max is None:
        # 动态计算归一化边界
        reward_min, reward_max = compute_reward_normalization_stats()

    # 归一化到[-1, 1]
    # 兼容numpy和torch：使用通用运算符
    normalized = (rewards - reward_min) / (reward_max - reward_min + 1e-8) * 2 - 1
    return normalized


def denormalize_rewards(rewards_normalized, reward_min=None, reward_max=None):
    """反归一化rewards从[-1, 1]到原始尺度（兼容torch.Tensor和numpy.ndarray）"""
    if reward_min is None or reward_max is None:
        # 使用缓存的归一化边界
        reward_min, reward_max = compute_reward_normalization_stats()

    # 从[-1, 1]反归一化
    # 兼容numpy和torch：使用通用运算符
    denormalized = (rewards_normalized + 1) / 2 * (reward_max - reward_min) + reward_min
    return denormalized


class CustomDDPMScheduler:
    def __init__(
            self,
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            device="cuda",
            use_parallel=False,
    ):

        if use_parallel:
            device = "cpu"

        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.device = device
        self.use_parallel = use_parallel

        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps).to(device)

        elif beta_schedule == "scaled_linear":
            self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps) ** 2
            self.betas = self.betas.to(device)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

    def add_noise(self, original_samples, noise, timesteps):
        timesteps = timesteps.to(self.device)
        original_samples = original_samples.to(self.device)
        noise = noise.to(self.device)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1)

        # Handle different tensor dimensions dynamically
        if len(original_samples.shape) == 4:
            # For 4D tensors [batch, channels, height, width]
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1)
        elif len(original_samples.shape) == 3:
            # For 3D tensors [batch, seq_len, features]
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1)
        else:
            # For 2D tensors [batch, features]
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1)

        noisy_samples = sqrt_alphas_cumprod_t * original_samples + \
                        sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_samples

    def step(self, model_output, timestep, sample):
        t = timestep
        prev_t = t - 1 if t > 0 else t

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod_prev[t]
        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

        # Clip预测的原始样本到[-1, 1]范围，防止去噪过程中数值爆炸
        # 因为训练数据被归一化到[-1, 1]，所以生成的样本也应该在这个范围内
        pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)

        variance = 0
        if t > 0:
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * self.betas[t]

        if t > 0:
            noise = torch.randn_like(sample)
            pred_prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + \
                               variance ** 0.5 * noise
        else:
            pred_prev_sample = pred_original_sample

        return type('PrevSampleOutput', (), {'prev_sample': pred_prev_sample})()

    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.linspace(self.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long)


class DiffusionModel(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim=512, num_heads=8, num_lstm_layers=1, dropout=0.2):
        super(DiffusionModel, self).__init__()
        self.input_dim = input_dim + condition_dim + 1
        self.hidden_dim = hidden_dim

        self.dropout = nn.Dropout(p=dropout)

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        # Use fewer heads to reduce memory usage
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=min(num_heads, 4),  # Cap at 4 heads to save memory
            batch_first=True,
            dropout=dropout
        )
        self.fc1 = nn.Linear(self.hidden_dim, input_dim)

    def forward(self, x, condition, timesteps):
        # 归一化timestep到0-1范围，避免与reward/condition特征尺度差异过大
        timesteps_normalized = timesteps.float() / config.NUM_TRAIN_TIMESTEPS
        timesteps_normalized = timesteps_normalized.unsqueeze(-1).unsqueeze(-1).expand(-1, x.shape[1], 1)

        combined = torch.cat([x, condition, timesteps_normalized], dim=2)
        # print('combined',combined.shape)
        lstm_out, _ = self.lstm(combined)
        lstm_out = self.dropout(lstm_out)

        # Use simple self-attention instead of multi-head attention to save memory
        try:
            attn_output, _ = self.attn(lstm_out, lstm_out, lstm_out)
            attn_output = self.dropout(attn_output)
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Fallback to simple linear transformation if OOM
                print(f"Memory error in attention, using fallback: {e}")
                attn_output = lstm_out
            else:
                raise e

        output = self.fc1(attn_output)
        # print('output',output.shape)
        return output


def train_diffusion_model(model, dataloader, optimizer, num_epochs, device, val_dataloader=None):
    """
    训练扩散模型，支持可选的验证集评估

    Args:
        model: 扩散模型
        dataloader: 训练数据加载器
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 设备
        val_dataloader: 可选的验证数据加载器，用于模型选择
    """
    losses = []
    if os.path.exists(config.BEST_DDP_MODEL_PATH):
        model.load_state_dict(torch.load(config.BEST_DDP_MODEL_PATH, map_location=device))

    noise_scheduler = CustomDDPMScheduler(
        num_train_timesteps=config.NUM_TRAIN_TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        beta_schedule=config.BETA_SCHEDULE,
        device=device,
        use_parallel=False
    )
    model.train()

    # ===== 新增：区分训练loss和验证loss =====
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    best_model_path = config.BEST_DDP_MODEL_PATH

    for epoch in range(num_epochs):

        epoch_losses = []
        batch_count = 0  # 用于梯度累积计数
        for batch_x, batch_c, batch_w in dataloader:
            # Clear cache before each batch to free up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            batch_x, batch_c, batch_w = batch_x.to(device), batch_c.to(device), batch_w.to(device)
            timesteps = torch.randint(0, config.NUM_TRAIN_TIMESTEPS, (batch_x.shape[0],), device=device)
            noise = torch.randn_like(batch_x).to(device)
            noisy_x = noise_scheduler.add_noise(batch_x, noise, timesteps)

            # 预测噪声
            pred_noise = model(noisy_x, batch_c, timesteps)
            # 正确的加权方式：先计算per-sample loss，再乘权重，避免梯度被平方放大
            per_sample_loss = F.mse_loss(pred_noise, noise, reduction='none').mean(dim=(1, 2))  # [B]
            loss = (per_sample_loss * batch_w).mean()

            # 直接backward，不需要除以4（梯度会自动累积）
            loss.backward()
            batch_count += 1

            # 每4个batch更新一次权重（梯度累积）
            if batch_count % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_losses.append(loss.item())

        # Epoch结束时，如果有残留梯度，执行一次更新
        if batch_count % 4 != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_train_loss)

        # ===== 新增：验证集评估 =====
        if val_dataloader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_c, batch_w in val_dataloader:
                    batch_x, batch_c, batch_w = batch_x.to(device), batch_c.to(device), batch_w.to(device)
                    timesteps = torch.randint(0, config.NUM_TRAIN_TIMESTEPS, (batch_x.shape[0],), device=device)
                    noise = torch.randn_like(batch_x).to(device)
                    noisy_x = noise_scheduler.add_noise(batch_x, noise, timesteps)
                    pred_noise = model(noisy_x, batch_c, timesteps)
                    per_sample_loss = F.mse_loss(pred_noise, noise, reduction='none').mean(dim=(1, 2))
                    loss = (per_sample_loss * batch_w).mean()
                    val_losses.append(loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
            model.train()

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # 根据验证loss保存模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_train_loss = avg_train_loss
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
                print(
                    f"[VALIDATION] Model saved with best val loss: {best_val_loss:.4f} (train loss: {best_train_loss:.4f})")
        else:
            # 无验证集，使用训练loss（原逻辑）
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}")

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
                print(f"Model parameters saved, current best loss: {best_train_loss:.4f}")

    loss_df = pd.DataFrame({'Epoch': range(1, len(losses) + 1), 'Loss': losses})
    file_path = config.TRAINING_LOSSES_PATH
    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    lock_path = file_path + ".lock"
    with filelock.FileLock(lock_path):
        if os.path.exists(file_path):
            existing_df = pd.read_excel(file_path)
            combined_df = pd.concat([existing_df, loss_df], ignore_index=True)
        else:
            combined_df = loss_df
        combined_df = combined_df.head(1000000)
        combined_df.to_excel(file_path, index=False, engine="openpyxl")


def generate_new_samples(model, condition, num_samples, device, use_parallel=False, use_passed_model=False):
    # 如果在并行环境中，强制使用CPU
    actual_device = "cpu" if use_parallel else device

    # 确保模型在正确设备上
    model = model.to(actual_device)
    # 如果存在预训练模型则加载（只有在不使用传入模型时才加载文件）
    if not use_passed_model and os.path.exists(config.BEST_DDP_MODEL_PATH):
        model.load_state_dict(torch.load(config.BEST_DDP_MODEL_PATH, map_location=actual_device))

    # # 使用自定义噪声调度器，并传入device
    noise_scheduler = CustomDDPMScheduler(
        num_train_timesteps=config.NUM_TRAIN_TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        beta_schedule=config.BETA_SCHEDULE,
        device=actual_device,
        use_parallel=use_parallel
    )

    # 设置推理时间步（使用配置的推理步数）
    noise_scheduler.set_timesteps(num_inference_steps=config.NUM_TRAIN_TIMESTEPS)
    # 设置模型为评估模式
    model.eval()
    # 初始化噪声，直接生成完整尺寸（4个reward列应该有独立的噪声）
    x = torch.randn(num_samples, 1000, 4, device=actual_device)  # [B,1000,4]

    # 将 condition 扩展到 B= num_samples
    condition = condition.to(actual_device)
    if condition.dim() == 2:  # [1000, C] to [1,1000,C]
        condition = condition.unsqueeze(0)
    if condition.size(0) == 1 and num_samples > 1:
        condition_expanded = condition.expand(num_samples, condition.size(1), condition.size(2))
    else:
        # 若你以后传入已是 [B,1000,C]，需断言 B 一致
        assert condition.size(0) == num_samples, \
            f"condition batch={condition.size(0)} does not match num_samples={num_samples}"
        condition_expanded = condition

    # 逐步去噪过程
    generated_samples = []
    with torch.no_grad():  # 不计算梯度
        for t in range(config.NUM_TRAIN_TIMESTEPS - 1, -1, -1):
            timesteps = torch.full((num_samples,), t, device=actual_device)
            # 预测噪声
            pred_noise = model(x, condition_expanded, timesteps)
            # 使用调度器进行去噪
            x = noise_scheduler.step(pred_noise, t, x).prev_sample
        generated_samples.append(x)

    # 将生成的样本拼接成一个张量
    generated_samples = torch.cat(generated_samples, dim=0)

    # 反归一化：从[-1, 1]恢复到原始reward尺度
    if config.REWARD_NORMALIZE:
        generated_samples = denormalize_rewards(generated_samples)

    return generated_samples


def find_free_port():
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]


@ray.remote
def evaluate_single_sample(sample_id, experiences_per_tl, base_state_path, epoch, j, tl_ids):
    # 强制在子进程中使用CPU设备
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 隐藏CUDA设备

    # 添加这行：明确设置子进程使用CPU
    subprocess_device = torch.device("cpu")

    sample_replay_buffers = {}
    sample_learners = {}
    try:
        base_state_dict = torch.load(base_state_path, map_location='cpu')

        port_dummy = find_free_port()
        train_like_env = SUMOTrafficLights(config.SUMO_CONFIG_PATH, port_dummy, False, 32)

        for tlID in tl_ids:
            sample_replay_buffers[tlID] = ReplayBuffer(capacity=10000)
            learner = QLearner(
                tlID, train_like_env, 0, 0, 0.01, 0.01, 'train',
                sample_replay_buffers[tlID],
                15, 1.0, 0.2, 10000, 5, 2, 32, None, subprocess_device
            )
            # 强制设置learner使用CPU
            learner.device = subprocess_device

            learner.policy_net.load_state_dict(base_state_dict[tlID]["policy"], strict=True)
            learner.target_net.load_state_dict(base_state_dict[tlID]["target"], strict=True)
            learner.policy_net.to(subprocess_device)
            learner.target_net.to(subprocess_device)

            # 恢复学习状态
            learner.learn_steps = base_state_dict[tlID]["learn_steps"]

            # 安全恢复优化器状态 - 处理设备兼容性
            optimizer_state = base_state_dict[tlID]["optimizer"]
            # 确保优化器状态中的张量都在正确设备上
            for state in optimizer_state['state'].values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(subprocess_device)

            learner.optimizer.load_state_dict(optimizer_state)

            sample_learners[tlID] = learner

        for tlID in tl_ids:
            exp_list = experiences_per_tl.get(str(tlID), []) or experiences_per_tl.get(tlID, [])
            if len(exp_list) > 0:
                sample_learners[tlID].sequential_learn_current_epoch(exp_list)

        port_eval = find_free_port()
        eval_env = SUMOTrafficLights(config.SUMO_CONFIG_PATH, port_eval, False, 32)
        eval_learners = {}
        eval_replay_buffers = {}
        for tlID in tl_ids:
            eval_replay_buffers[tlID] = ReplayBuffer(capacity=10000)
            eval_learner = QLearner(
                tlID, eval_env, 0, 0, 0.01, 0.01, 'eval', eval_replay_buffers[tlID],
                15, 0, 0, 1, 5, 2, 32, None, subprocess_device
            )
            eval_learner.policy_net.load_state_dict(sample_learners[tlID].policy_net.state_dict())
            eval_learner.target_net.load_state_dict(eval_learner.policy_net.state_dict())
            eval_learner.policy_net.eval()
            eval_learner.target_net.eval()
            eval_learner.policy_net.to(subprocess_device)
            eval_learner.target_net.to(subprocess_device)
            eval_learners[tlID] = eval_learner

        eval_env.learners = eval_learners
        eval_env.replay_buffers = eval_replay_buffers
        # 为 SUMO 启动/连接异常增加最多 5 次重试；每次失败后重建 eval_env 并更换端口
        metrics = None
        last_err = None
        for attempt in range(5):
            try:
                metrics = eval_env.run_episode(config.SIMULATION_DURATION_SEC, None, epoch, 'eval', sample_id=sample_id,
                                               save_outputs=False)
                last_err = None
                break
            except Exception as e:
                last_err = e
                # 关闭并重建 eval_env，使用新的端口
                try:
                    if 'eval_env' in locals() and eval_env is not None:
                        if hasattr(eval_env, 'close'):
                            eval_env.close()
                except:
                    pass
                try:
                    port_eval = find_free_port()
                    eval_env = SUMOTrafficLights(config.SUMO_CONFIG_PATH, port_eval, False, 32)
                except Exception as _:
                    pass
        if last_err is not None:
            raise last_err
        # 评估指标：优先使用“尾段鲁棒”，并加入按方向的稳定性检查
        inner_list = metrics.get("average_queue_inner_per_road")
        inner_sum = float(metrics.get("average_queue_inner_sum") or 0.0)
        robust = metrics.get('avg_tail_robust')
        p95 = metrics.get('p95_tail_robust')
        mx = metrics.get('max_tail_robust')
        tail_dir_avg = metrics.get('avg_tail_robust_per_road') or []
        tail_dir_p95 = metrics.get('p95_tail_robust_per_road') or []
        tail_dir_max = metrics.get('max_tail_robust_per_road') or []

        stable = True
        if robust is not None:
            try:
                if p95 is not None and mx is not None:
                    if p95 > config.ROBUST_P95_FACTOR * robust or mx > config.ROBUST_MAX_FACTOR * robust:
                        stable = False
            except Exception:
                pass
        # 方向级稳定性：每个方向的 p95/max 相对均值不超过阈值
        stable_dir = True
        try:
            for i in range(min(4, len(tail_dir_avg))):
                a = tail_dir_avg[i]
                p = tail_dir_p95[i] if i < len(tail_dir_p95) else None
                m = tail_dir_max[i] if i < len(tail_dir_max) else None
                if a is None or p is None or m is None:
                    continue
                if p > getattr(config, 'ROBUST_P95_FACTOR', 1.3) * a or m > getattr(config, 'ROBUST_MAX_FACTOR', 1.6) * a:
                    stable_dir = False
                    break
        except Exception:
            pass

        # 一致的比较指标：单方向最差（尾段）
        if tail_dir_avg:
            worst_tail_dir_avg = max([v for v in tail_dir_avg if v is not None], default=None)
        else:
            worst_tail_dir_avg = robust if robust is not None else inner_sum

        # 返回兼容字段：combined_metric 仍写入 avg_inner_sum，用于旧逻辑；同时返回 worst_tail_dir_avg
        combined_metric = float(worst_tail_dir_avg) if (worst_tail_dir_avg is not None and stable and stable_dir) else float('inf')

        # 在返回前，保存学习后的网络状态
        learned_network_state = {}
        for tlID in tl_ids:
            learned_network_state[tlID] = {
                "policy": {k: v.detach().cpu().clone() for k, v in
                           sample_learners[tlID].policy_net.state_dict().items()},
                "target": {k: v.detach().cpu().clone() for k, v in
                           sample_learners[tlID].target_net.state_dict().items()},
                "learn_steps": sample_learners[tlID].learn_steps,
                "steps_done": sample_learners[tlID].steps_done,  # 新增：探索步数
                "best_loss": sample_learners[tlID].best_loss,  # 新增：最佳损失
                "optimizer": {
                    k: v.cpu() if torch.is_tensor(v) else v
                    for k, v in sample_learners[tlID].optimizer.state_dict().items()
                }
            }

        return {
            "epoch": int(epoch),
            "round": int(j),
            "sample_id": int(sample_id),
            # 最内圈多目标（按 top,right,bottom,left 顺序）
            "avg_inner_top": float(inner_list[0]) if inner_list and inner_list[0] is not None else None,
            "avg_inner_right": float(inner_list[1]) if inner_list and inner_list[1] is not None else None,
            "avg_inner_bottom": float(inner_list[2]) if inner_list and inner_list[2] is not None else None,
            "avg_inner_left": float(inner_list[3]) if inner_list and inner_list[3] is not None else None,
            # 新增：每方向尾段鲁棒均值（与目标体系对齐）
            "tail_top": float(tail_dir_avg[0]) if tail_dir_avg and tail_dir_avg[0] is not None else None,
            "tail_right": float(tail_dir_avg[1]) if len(tail_dir_avg) > 1 and tail_dir_avg[1] is not None else None,
            "tail_bottom": float(tail_dir_avg[2]) if len(tail_dir_avg) > 2 and tail_dir_avg[2] is not None else None,
            "tail_left": float(tail_dir_avg[3]) if len(tail_dir_avg) > 3 and tail_dir_avg[3] is not None else None,
            # 兼容字段：组合指标写入 avg_inner_sum（此处为 worst_tail_dir_avg 或 inf）
            "avg_inner_sum": float(combined_metric),
            # 明确返回总体与最差方向尾段指标
            "avg_tail_robust": float(robust) if robust is not None else None,
            "worst_tail_dir_avg": float(worst_tail_dir_avg) if worst_tail_dir_avg is not None else None,
            "network_state": learned_network_state  # 学习后的网络状态
        }
    finally:
        # 强制关闭SUMO环境
        try:
            if 'train_like_env' in locals():
                # 强制终止SUMO进程
                if hasattr(train_like_env, '_sumo_process'):
                    try:
                        train_like_env._sumo_process.terminate()
                        train_like_env._sumo_process.wait(timeout=1)
                    except:
                        try:
                            train_like_env._sumo_process.kill()
                        except:
                            pass
                if hasattr(train_like_env, 'close'):
                    train_like_env.close()
                del train_like_env
        except:
            pass

        try:
            if 'eval_env' in locals():
                # 强制终止SUMO进程
                if hasattr(eval_env, '_sumo_process'):
                    try:
                        eval_env._sumo_process.terminate()
                        eval_env._sumo_process.wait(timeout=1)
                    except:
                        try:
                            eval_env._sumo_process.kill()
                        except:
                            pass
                if hasattr(eval_env, 'close'):
                    eval_env.close()
                del eval_env
        except:
            pass

        # 清理learners和buffers
        try:
            if 'eval_learners' in locals():
                for tlID in list(eval_learners.keys()):
                    del eval_learners[tlID]
                del eval_learners
        except:
            pass

        try:
            if 'eval_replay_buffers' in locals():
                for tlID in list(eval_replay_buffers.keys()):
                    del eval_replay_buffers[tlID]
                del eval_replay_buffers
        except:
            pass

        try:
            if 'sample_learners' in locals():
                for tlID in list(sample_learners.keys()):
                    del sample_learners[tlID]
                del sample_learners
        except:
            pass

        try:
            if 'sample_replay_buffers' in locals():
                for tlID in list(sample_replay_buffers.keys()):
                    del sample_replay_buffers[tlID]
                del sample_replay_buffers
        except:
            pass

        # 强制垃圾回收
        import gc
        gc.collect()


def save_best_network(train_learners, current_metric, epoch, global_best_metric, global_best_epoch):
    """
    保存当前最佳网络参数（基于排队长度）

    Args:
        train_learners: 训练网络learners字典
        current_metric: 当前评估的排队长度
        epoch: 当前epoch
        global_best_metric: 当前全局最佳排队长度
        global_best_epoch: 当前全局最佳epoch

    Returns:
        tuple: (new_global_best_metric, new_global_best_epoch)
    """
    from datetime import datetime

    # 检查是否需要更新最佳网络
    if current_metric < global_best_metric:
        print(f"[BEST NETWORK] Better network found! {global_best_metric:.2f} to {current_metric:.2f} (epoch {epoch})")

        # 确保weights文件夹存在
        os.makedirs(config.WEIGHTS_DIR, exist_ok=True)

        # 保存网络参数 - 采用与best_model3.pth相同的格式
        best_delay_path = config.BEST_DELAY_MODEL_PATH

        # 保存每个交通灯的policy_net参数（多路口合并保存，便于独立评估复现）
        combined_state = {}
        for tlID in train_learners.keys():
            combined_state[str(tlID)] = train_learners[tlID].policy_net.state_dict()
        torch.save(combined_state, best_delay_path)

        # 写入日志文件
        log_path = config.BEST_DELAY_LOG_PATH
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_entry = f"[{current_time}] Epoch {epoch}: Saved best network parameters, queue length: {current_metric:.2f}\n"

        # 追加写入日志（如果文件不存在会创建）
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)

        print(f"[BEST NETWORK] Network parameters saved to {best_delay_path}")
        print(f"[BEST NETWORK] Log updated to {log_path}")

        # 更新全局最佳记录
        return current_metric, epoch
    else:
        print(f"[BEST NETWORK] Current queue({current_metric:.2f}) >= Global best({global_best_metric:.2f}), no change")
        return global_best_metric, global_best_epoch


def save_bad_network(train_learners, epoch, note: str = ""):
    # 已废弃：占位以兼容调用路径（不执行保存）
    return


def main():
    global generated_samples
    epoches = config.EPOCHS
    device = config.DEVICE
    print(f"[DEVICE] Using device: {device}")
    if device.type == 'cuda':
        print(f"[DEVICE] CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    # Create necessary directories
    from create_directories import create_directories
    create_directories()

    # 初始化Ray (禁用日志输出，解决端口冲突)
    if not ray.is_initialized():
        try:
            # 使用所有CPU核心，通过分批评估来控制资源
            ray.init(num_cpus=84, log_to_driver=False, logging_level='ERROR')
        except ValueError as e:
            if "port" in str(e).lower():
                print("[RAY] Port conflict detected, attempting to close existing Ray instance and reinitialize...")
                try:
                    ray.shutdown()  # 强制关闭可能存在的Ray实例
                    time.sleep(2)  # 等待端口释放
                except:
                    pass
                # 使用更保守的配置重新初始化
                ray.init(num_cpus=min(84, mp.cpu_count()),
                         log_to_driver=False,
                         logging_level='ERROR',
                         include_dashboard=False,
                         ignore_reinit_error=True)
            else:
                raise e

    # 初始化全局最佳排队长度记录
    global_best_queue = float('inf')
    global_best_epoch = -1
    # 尝试从历史日志中恢复上次运行的全局最佳排队长度
    try:
        best_log_path = config.BEST_DELAY_LOG_PATH
        if os.path.exists(best_log_path):
            with open(best_log_path, 'r', encoding='utf-8') as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            # 取最后一条改进记录（日志按时间追加，最后一条即历史最优）
            import re
            pattern = re.compile(r"Epoch\s+(\d+):\s+Saved best network parameters, queue length:\s+([\d\.]+)")
            for ln in reversed(lines):
                m = pattern.search(ln)
                if m:
                    global_best_epoch = int(m.group(1))
                    global_best_queue = float(m.group(2))
                    print(
                        f"[RESUME] Loaded previous global best queue={global_best_queue:.2f} (epoch {global_best_epoch})")
                    break
    except Exception as e:
        print(f"[RESUME] Failed to load previous best queue: {e}")

    # 本地断点文件：记录应当从哪个 epoch 开始运行
    epoch_marker_file = config.EPOCH_MARKER_PATH
    start_epoch = 0
    if os.path.exists(epoch_marker_file):
        try:
            with open(epoch_marker_file, 'r', encoding='utf-8') as f:
                start_epoch = int(f.read().strip() or '0')
            print(f"Checkpoint detected, starting from epoch={start_epoch}")
        except Exception as e:
            print(f"Failed to read checkpoint: {e}, starting from epoch=0")

    print("[INIT] Creating Training Environment")
    port = find_free_port()
    train_env = SUMOTrafficLights(config.SUMO_CONFIG_PATH, port, False, 32)

    # —— 在第一次创建 learners 之前，确保初始 checkpoint 存在 ——
    model_filename = config.RL_MODEL_PATH
    if not os.path.isfile(model_filename):
        os.makedirs(os.path.dirname(model_filename), exist_ok=True)
        # 2) 构造一个和 QLearner policy_net 一样结构的 DQN，并保存它的初始 state_dict
        dummy_net = DQN(5, 2).to(device)  # 网络维度要和 QLearner 中一致 :contentReference[oaicite:0]{index=0}
        torch.save(dummy_net.state_dict(), model_filename)

    train_learners = {}
    train_replay_buffers = {}
    for tlID in train_env.get_trafficlights_ID_list():
        train_replay_buffers[tlID] = ReplayBuffer(capacity=10000)
        train_learners[tlID] = QLearner(tlID, train_env, 0, 0, 0.01, 0.01, 'train', train_replay_buffers[tlID], 15, 1.0,
                                        0.2, 10000,
                                        5, 2, 32, config.RL_MODEL_PATH, device)

    train_env.learners = train_learners
    train_env.replay_buffers = train_replay_buffers
    # 如果存在历史全局最佳排队长度，并且有对应的最佳权重文件，则在启动时加载到所有learner
    try:
        if global_best_queue != float('inf') and os.path.exists(config.BEST_DELAY_MODEL_PATH):
            best_state = torch.load(config.BEST_DELAY_MODEL_PATH, map_location=device)
            # 仅支持“多路口合并保存”的新格式：{tlID(str): state_dict}
            for tlID in train_env.get_trafficlights_ID_list():
                key = str(tlID)
                if key not in best_state:
                    raise RuntimeError(f"Missing key '{key}' in best-state file {config.BEST_DELAY_MODEL_PATH}")
                train_learners[tlID].policy_net.load_state_dict(best_state[key], strict=False)
                train_learners[tlID].target_net.load_state_dict(best_state[key], strict=False)
            print(f"[RESUME] Loaded per-TL best weights from {config.BEST_DELAY_MODEL_PATH}")
    except Exception as e:
        print(f"[RESUME] Skip loading best model: {e}")

    for epoch in range(start_epoch, epoches):
        import gc
        gc.collect()

        starttime = datetime.now().time()
        start_seconds = starttime.hour * 3600 + starttime.minute * 60 + starttime.second
        print(f"[TRAINING] Starting Epoch {epoch}")
        # 根据配置选择扩散训练轮数（不再硬编码）
        if epoch == 0:
            num_epochs_0 = config.DIFFUSION_EPOCHS_MAIN
        else:
            num_epochs_0 = config.DIFFUSION_EPOCHS_REGULAR

        # # 动态确定n_episodes
        cpm_final_path = config.CPM_DATA_PATH
        try:
            if os.path.exists(cpm_final_path):
                df = pd.read_csv(cpm_final_path)
                current_rows = len(df)
            else:
                current_rows = 0
        except Exception as e:
            print(f"Failed to read {cpm_final_path}: {e}, setting current_rows to 0")
            current_rows = 0

        # 计算需要运行的episode次数（从配置读取目标行数）
        target_rows = getattr(config, "CPM_TARGET_ROWS", 500000)
        if current_rows >= target_rows:
            n_episodes = 1
        else:
            needed = target_rows - current_rows
            n_episodes = (needed + 99) // 1000  # 向上取整

        print(f"Current data rows: {current_rows}, need to run {n_episodes} episodes")

        for j in range(n_episodes):
            # 训练阶段 - 运行一轮
            print(f"[TRAINING] Round {j + 1}/{n_episodes}")
            train_env.run_episode(config.SIMULATION_DURATION_SEC, None, epoch, 'train')
            if epoch == 0:
                print(f"[LEARNING] Starting Batch Learning")

                for tlID in train_env.get_trafficlights_ID_list():
                    # 获取本轮收集的经验
                    if str(tlID) in train_env.collected_experiences:
                        experiences = train_env.collected_experiences[str(tlID)]

                        if len(experiences) > 0:
                            print(f"Traffic Light {tlID}: Collected {len(experiences)} experiences this round")

                            # **学习**
                            learner = train_learners[tlID]
                            learner.sequential_learn_current_epoch(experiences)

                            # 清空本轮经验（避免重复学习）
                            train_env.collected_experiences[str(tlID)] = []
                        else:
                            print(f"Traffic Light {tlID}: No experience this round")
                    else:
                        print(f"Traffic Light {tlID}: No experience record found")

                # ===== Epoch=0 学习完成后进行独立评估 =====
                print("[EVALUATION] Starting Independent Network Evaluation")
                port_eval = find_free_port()
                eval_env = SUMOTrafficLights(config.SUMO_CONFIG_PATH, port_eval, False, 32)

                eval_learners = {}
                eval_replay_buffers = {}
                for tlID in train_env.get_trafficlights_ID_list():
                    eval_replay_buffers[tlID] = ReplayBuffer(capacity=10000)
                    eval_learner = QLearner(
                        tlID, eval_env, 0, 0, 0.01, 0.01, 'eval',
                        eval_replay_buffers[tlID], 15, 0, 0, 1, 5, 2, 32,
                        None, device
                    )
                    # 复制刚学习过的主网络状态到评估网络
                    eval_learner.policy_net.load_state_dict(train_learners[tlID].policy_net.state_dict())
                    eval_learner.target_net.load_state_dict(train_learners[tlID].target_net.state_dict())
                    eval_learner.policy_net.eval()
                    eval_learner.target_net.eval()
                    eval_learners[tlID] = eval_learner

                eval_env.learners = eval_learners
                eval_env.replay_buffers = eval_replay_buffers

                print("Starting evaluation mode simulation")
                eval_metrics = eval_env.run_episode(config.SIMULATION_DURATION_SEC, None, epoch, 'eval')

                # 使用“最差方向的尾段鲁棒”作为一致的比较指标
                robust = eval_metrics.get('avg_tail_robust')
                p95 = eval_metrics.get('p95_tail_robust')
                mx = eval_metrics.get('max_tail_robust')
                tail_dir_avg = eval_metrics.get('avg_tail_robust_per_road') or []
                tail_dir_p95 = eval_metrics.get('p95_tail_robust_per_road') or []
                tail_dir_max = eval_metrics.get('max_tail_robust_per_road') or []

                if tail_dir_avg:
                    worst_tail_dir_avg = max([v for v in tail_dir_avg if v is not None], default=None)
                else:
                    worst_tail_dir_avg = robust

                if worst_tail_dir_avg is not None:
                    current_queue = float(worst_tail_dir_avg)
                    print(
                        f"Epoch=0 independent evaluation: worst_tail_dir_avg = {current_queue:.2f}, overall avg_tail={robust}, p95={p95}, max={mx}, tail_steps = {eval_metrics.get('tail_steps')}")
                else:
                    current_queue = float(eval_metrics.get('average_queue_inner_sum') or 0.0)
                    print(f"Epoch=0 independent evaluation: fallback avg_inner_sum = {current_queue:.2f}")

                stable = True
                try:
                    if robust is not None and p95 is not None and mx is not None:
                        if p95 > config.ROBUST_P95_FACTOR * robust or mx > config.ROBUST_MAX_FACTOR * robust:
                            stable = False
                except Exception:
                    pass
                # 方向级稳定性
                stable_dir = True
                try:
                    for i in range(min(4, len(tail_dir_avg))):
                        a = tail_dir_avg[i]
                        p = tail_dir_p95[i] if i < len(tail_dir_p95) else None
                        m = tail_dir_max[i] if i < len(tail_dir_max) else None
                        if a is None or p is None or m is None:
                            continue
                        if p > getattr(config, 'ROBUST_P95_FACTOR', 1.3) * a or m > getattr(config, 'ROBUST_MAX_FACTOR', 1.6) * a:
                            stable_dir = False
                            break
                except Exception:
                    pass

                if stable and stable_dir:
                    # 保存最佳网络参数
                    global_best_queue, global_best_epoch = save_best_network(
                        train_learners, current_queue, epoch, global_best_queue, global_best_epoch
                    )
                else:
                    print(f"[BEST NETWORK] Skip update due to instability: overall(p95={p95}, max={mx}, mean={robust}), per-dir stable={stable_dir}")

                # 清理评估环境
                try:
                    eval_env.close()
                    del eval_env, eval_learners, eval_replay_buffers
                except:
                    pass



            # 从第1个epoch开始：用本回合的 (s,a) 600步序列，生成奖励并离线学习
            elif epoch >= 1:
                sa_path = config.SA_SEQ_PATH
                if not os.path.exists(sa_path):
                    print(f"[WARN] File not found {sa_path}, skipping reward generation and offline learning")
                else:
                    sa_df = pd.read_excel(sa_path)
                    # 解析4列 state_action_0..3，每行是长度=6 的列表字符串
                    sa_lists = []
                    for c in ['state_action_0', 'state_action_1', 'state_action_2', 'state_action_3']:
                        if c in sa_df.columns:
                            sa_lists.append(parse_state_column(sa_df[c]))
                        else:
                            sa_lists.append([])

                    # 对齐4列长度，取最短的1000
                    min_len = min([len(x) for x in sa_lists if len(x) > 0]) if any(len(x) > 0 for x in sa_lists) else 0
                    min_len = min(min_len, 1000)
                    sa_list = []
                    for t in range(min_len):
                        combined = sa_lists[0][t] + sa_lists[1][t] + sa_lists[2][t] + sa_lists[3][t]
                        sa_list.append(combined)

                    if len(sa_list) > 0:
                        sa_arr = np.array(sa_list, dtype=np.float32).reshape(1, min_len, 24)
                        if min_len < 1000:
                            pad = np.repeat(sa_arr[:, -1:, :], 1000 - min_len, axis=1)
                            sa_arr = np.concatenate([sa_arr, pad], axis=1)
                        sa_tensor = torch.tensor(sa_arr, dtype=torch.float32).to(device)  # [1,1000,24]

                        # 准备自适应扩散模型并生成当前回合奖励
                        input_dim = 4
                        condition_dim = 24
                        # 创建统一的自适应扩散模型
                        diffusion_model = DiffusionModel(input_dim, condition_dim, hidden_dim=256, dropout=0.3).to(
                            device)

                        # 优化: 直接加载预训练模型(如果存在),提高代码可读性
                        if os.path.exists(config.BEST_DDP_MODEL_PATH):
                            diffusion_model.load_state_dict(torch.load(config.BEST_DDP_MODEL_PATH, map_location=device))
                            print(f"[MODEL] Loaded pretrained diffusion model from {config.BEST_DDP_MODEL_PATH}")

                        expanded_target = sa_tensor  # [1,1000,24]
                        gen = generate_new_samples(
                            diffusion_model,
                            expanded_target,
                            num_samples=config.POPULATION_SIZE,
                            device=device,
                            use_parallel=False,
                            use_passed_model=True,  # 修改为True,因为已经手动加载
                        )  # [POPULATION_SIZE,1000,4] - 使用预训练权重
                        gen_np = gen.detach().cpu().numpy()

                        # 释放GPU tensor，防止显存累积
                        del gen
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        # 保存生成的样本到二进制文件（保持浮点数）
                        os.makedirs(config.SAMPLES_DIR, exist_ok=True)
                        all_samples_path = config.get_sample_path(epoch, j)
                        gen_float_np = gen_np  # 保持原始浮点数，不进行四舍五入
                        np.save(all_samples_path, gen_float_np)

                        # 初始化本轮种群内的样本级“最佳”占位，避免与全局网络最佳变量同名冲突
                        sample_best_sample = None  # [1000, 4] 本轮样本级最佳（未使用保留）
                        sample_best_queue = float('inf')  # 本轮样本级最佳延误（未使用保留）
                        sample_best_info = None  # 本轮样本级最佳信息（未使用保留）

                        # Evolution tracking (aligned with main_bak)
                        evolution_best_sample = None
                        evolution_best_queue = float('inf')
                        evolution_best_info = None
                        evolution_best_network_state = None  # 内存中的全局最优网络状态（用于缺文件兜底）
                        # 移除模型状态备份：让模型能够累积学习，而不是每次恢复原点

                        # ===== 新增: 最佳微调模型追踪 =====
                        best_finetuned_model_state = None  # 保存最优迭代的微调模型权重
                        best_finetuned_queue = float('inf')  # 最优微调对应的延误
                        best_finetuned_iter = -1  # 最优微调所在的迭代

                        print(f"[EVOLUTION] Starting adaptive evolution with {config.EVOLUTION_ITERATIONS} iterations")

                        # 扩散变异进化算法
                        for iter_idx in range(config.EVOLUTION_ITERATIONS):
                            # 每次迭代：精英选择 + 扩散变异

                            # ===== 加载当前种群 =====
                            current_population = np.load(all_samples_path)  # 当前种群

                            # 1) 快照"最新训练网络"的权重，确保每个样本的副本起点一致
                            base_state_dict = {
                                tlID: {
                                    "policy": {k: v.detach().cpu().clone() for k, v in
                                               train_learners[tlID].policy_net.state_dict().items()},
                                    "target": {k: v.detach().cpu().clone() for k, v in
                                               train_learners[tlID].target_net.state_dict().items()},
                                    "learn_steps": train_learners[tlID].learn_steps,
                                    "optimizer": {
                                        k: v.cpu() if torch.is_tensor(v) else v
                                        for k, v in train_learners[tlID].optimizer.state_dict().items()
                                    }  # 确保优化器状态也在CPU上
                                }
                                for tlID in train_env.get_trafficlights_ID_list()
                            }

                            # 在训练循环之前准备一个结果列表
                            sample_results = []  # [{epoch, round, sample_id, average_queue}]
                            tl_ids = list(train_env.get_trafficlights_ID_list())  # 转换为list

                            # 保存基准权重到文件，避免在进程间大对象传输
                            os.makedirs(config.WEIGHTS_DIR, exist_ok=True)
                            base_state_path = config.get_base_state_path(epoch, j)
                            torch.save(base_state_dict, base_state_path)

                            # 预先在主进程构造每个样本的经验，以便传给子进程
                            tasks = []
                            for sample_id in range(current_population.shape[0]):  # POPULATION_SIZE
                                rewards_array = current_population[sample_id]  # [1000,4] 使用当前种群Dn
                                exp_per_tl = train_env.build_experiences_from_rewards_array(j, rewards_array)
                                tasks.append((sample_id, exp_per_tl))

                            # 关键修复：分批评估，避免一次性提交500个任务导致资源耗尽
                            batch_size = config.EVALUATION_BATCH_SIZE
                            total_tasks = len(tasks)
                            num_batches = (total_tasks + batch_size - 1) // batch_size

                            print(
                                f"[ITER {iter_idx + 1}] Evaluating {total_tasks} samples in {num_batches} batches (batch_size={batch_size})")

                            # 分批提交任务
                            for batch_idx in range(num_batches):
                                start_idx = batch_idx * batch_size
                                end_idx = min(start_idx + batch_size, total_tasks)
                                batch_tasks = tasks[start_idx:end_idx]

                                print(
                                    f"[ITER {iter_idx + 1}] Processing batch {batch_idx + 1}/{num_batches} (samples {start_idx}-{end_idx - 1})")

                                # 提交当前批次的任务
                                futures = [
                                    evaluate_single_sample.remote(
                                        sample_id,
                                        exp_per_tl,
                                        base_state_path,
                                        epoch,
                                        j,
                                        tl_ids
                                    )
                                    for sample_id, exp_per_tl in batch_tasks
                                ]

                                # 等待当前批次完成
                                batch_results = ray.get(futures)
                                sample_results.extend(batch_results)

                                print(
                                    f"[ITER {iter_idx + 1}] Batch {batch_idx + 1}/{num_batches} completed, collected {len(batch_results)} results")

                                # 批次间延迟，确保SUMO进程完全清理
                                if batch_idx < num_batches - 1:
                                    print(
                                        f"[ITER {iter_idx + 1}] Waiting {config.INTER_BATCH_DELAY}s for process cleanup...")
                                    time.sleep(config.INTER_BATCH_DELAY)

                            # 清理进程池后的残留
                            subprocess.call(['pkill', '-f', 'sumo'], stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL)
                            print(f"[ITER {iter_idx + 1}] All {total_tasks} samples evaluated successfully")

                            os.makedirs(config.SAMPLES_DIR, exist_ok=True)

                            # ===== 关键修正：按sample_id重新排序结果 =====
                            df = pd.DataFrame(sample_results)
                            df = df.sort_values('sample_id').reset_index(drop=True)  # 按sample_id排序！！！

                            # 多目标：使用按方向的尾段鲁棒均值（与保存/微调目标一致）
                            objective_cols = ['tail_top', 'tail_right', 'tail_bottom', 'tail_left']
                            # 计算内圈和
                            if 'avg_inner_sum' not in df.columns:
                                try:
                                    df['avg_inner_sum'] = df[objective_cols].sum(axis=1, skipna=True, min_count=1)
                                except TypeError:
                                    def _row_sum_nan_safe(row):
                                        vals = [v for v in row if pd.notna(v)]
                                        return sum(vals) if len(vals) > 0 else np.nan

                                    df['avg_inner_sum'] = df[objective_cols].apply(_row_sum_nan_safe, axis=1)

                            # 代表性标量（用于统计/当前最佳/日志）：使用单方向最差的尾段鲁棒
                            queues = pd.to_numeric(df["worst_tail_dir_avg"], errors="coerce").to_numpy(dtype=np.float64)

                            # 验证排序正确性 (仅在调试模式下)
                            if config.DEBUG_MODE:
                                expected_sample_ids = list(range(len(current_population)))
                                actual_sample_ids = df['sample_id'].tolist()
                                if actual_sample_ids != expected_sample_ids:
                                    print(
                                        f"[ERROR] sample_id order mismatch! Expected: {expected_sample_ids[:10]}..., Actual: {actual_sample_ids[:10]}...")
                                else:
                                    print(f"[VALIDATION] Sample ordering verified successfully")

                            probs = np.zeros_like(queues, dtype=np.float64)
                            # 多目标选择策略：固定使用 Pareto（无兼容分支）
                            F = df[objective_cols].to_numpy(dtype=np.float64)
                            valid = np.all(np.isfinite(F), axis=1)
                            if valid.any():
                                # Fast Non-Dominated Sorting
                                idxs = np.where(valid)[0]
                                Fv = F[valid]
                                n = Fv.shape[0]
                                dominates = [set() for _ in range(n)]
                                dominated_count = np.zeros(n, dtype=np.int32)
                                fronts = []
                                for i in range(n):
                                    for j in range(i + 1, n):
                                        fi = Fv[i]
                                        fj = Fv[j]
                                        if np.all(fi <= fj) and np.any(fi < fj):
                                            dominates[i].add(j)
                                            dominated_count[j] += 1
                                        elif np.all(fj <= fi) and np.any(fj < fi):
                                            dominates[j].add(i)
                                            dominated_count[i] += 1
                                current_front = [i for i in range(n) if dominated_count[i] == 0]
                                rank = np.full(n, fill_value=np.iinfo(np.int32).max, dtype=np.int32)
                                r = 0
                                while current_front:
                                    fronts.append(current_front)
                                    next_front = []
                                    for p in current_front:
                                        rank[p] = r
                                        for q in dominates[p]:
                                            dominated_count[q] -= 1
                                            if dominated_count[q] == 0:
                                                next_front.append(q)
                                    r += 1
                                    current_front = next_front

                                # Crowding Distance per front
                                cd = np.zeros(n, dtype=np.float64)
                                if n > 0:
                                    m = Fv.shape[1]
                                    for front in fronts:
                                        if len(front) == 0:
                                            continue
                                        front_idx = np.array(front, dtype=np.int32)
                                        cd_front = np.zeros(len(front_idx), dtype=np.float64)
                                        for k in range(m):
                                            vals = Fv[front_idx, k]
                                            order = np.argsort(vals, kind='mergesort')
                                            cd_front[order[0]] = np.inf
                                            cd_front[order[-1]] = np.inf
                                            vmin = vals[order[0]]
                                            vmax = vals[order[-1]]
                                            denom = (vmax - vmin) if (vmax - vmin) > 1e-12 else 1.0
                                            for t in range(1, len(order) - 1):
                                                prevv = vals[order[t - 1]]
                                                nextv = vals[order[t + 1]]
                                                cd_front[order[t]] += (nextv - prevv) / denom
                                        cd[front_idx] = cd_front

                                # 概率映射
                                alpha = getattr(config, 'PARETO_ALPHA', 1.0)
                                beta = getattr(config, 'PARETO_BETA', 1e-3)
                                rank_weight = np.exp(-alpha * rank)
                                # 归一化拥挤距离：将非有限值视为最大拥挤（边界点）
                                if np.any(np.isfinite(cd)):
                                    cd_max = np.nanmax(cd[np.isfinite(cd)])
                                    cd_eff = np.where(np.isfinite(cd), cd, cd_max)
                                    denom = (cd_max + 1e-9) if cd_max > 0 else 1.0
                                    cd_norm = cd_eff / denom
                                else:
                                    cd_norm = np.zeros_like(cd)
                                p_valid = rank_weight * (beta + cd_norm)
                                probs[idxs] = p_valid
                                s = probs.sum()
                                if s <= 0:
                                    probs[valid] = 1.0 / valid.sum()
                                else:
                                    probs = probs / s
                            else:
                                probs[:] = 1.0 / len(queues) if len(queues) > 0 else 0.0

                            df["q"] = probs
                            df.to_csv(config.get_sample_eval_path(epoch, j), index=False)

                            # 找到当前迭代中“最优样本”（用于保存网络与微调的代表性标量）
                            # 代表性标量：使用 worst_tail_dir_avg（单方向最差尾段鲁棒）
                            current_queues = pd.to_numeric(df["worst_tail_dir_avg"], errors="coerce").to_numpy(
                                dtype=np.float64)
                            current_queues_safe = np.where(np.isfinite(current_queues), current_queues, np.inf)

                            # 声明变量避免NameError
                            current_best_sample = None
                            current_best_queue = float('inf')  # 改用排队长度
                            current_best_sample_id = -1

                            if len(current_queues_safe) > 0:
                                # 找到当前迭代的最佳样本（排队最少）
                                current_best_idx = int(np.argmin(current_queues_safe))
                                current_best_sample_id = int(df.iloc[current_best_idx]["sample_id"])
                                current_best_queue = current_queues_safe[current_best_idx]

                                # 获取当前迭代的最佳样本
                                current_best_sample = current_population[
                                    current_best_sample_id].copy()  # [1000, 4] 从当前种群Dn选择

                                print(
                                    f"[ITER {iter_idx + 1}] Current iteration best: sample_id={current_best_sample_id}, queue={current_best_queue:.2f}")

                                # 更新全局最佳样本
                                if iter_idx == 0:
                                    # 第一次迭代，直接保存为全局最佳
                                    evolution_best_sample = current_best_sample.copy()
                                    evolution_best_queue = current_best_queue
                                    evolution_best_info = {
                                        'epoch': epoch,
                                        'round': j,
                                        'iter_idx': iter_idx,
                                        'sample_id': current_best_sample_id,
                                        'queue': current_best_queue
                                    }

                                    # **新增：保存对应的网络状态**
                                    # 从评估结果中找到对应的网络状态
                                    current_best_network_state = None
                                    for result in sample_results:
                                        if result['sample_id'] == current_best_sample_id:
                                            current_best_network_state = result['network_state']
                                            break

                                    if current_best_network_state is not None:
                                        evolution_best_network_state = current_best_network_state
                                        # 保存到文件
                                        evolution_best_network_path = config.get_global_best_network_path(epoch, j)
                                        os.makedirs(os.path.dirname(evolution_best_network_path), exist_ok=True)
                                        torch.save(evolution_best_network_state, evolution_best_network_path)
                                        print(
                                            f"[EVOLUTION] 1st iteration, set evolution best: queue={evolution_best_queue:.2f}, network state saved")
                                    else:
                                        print(f"[WARN] Network state not found for sample_id={current_best_sample_id}")

                                    # ===== 新增: 保存最佳微调模型 =====
                                    best_finetuned_model_state = {k: v.clone() for k, v in
                                                                  diffusion_model.state_dict().items()}
                                    best_finetuned_queue = current_best_queue
                                    best_finetuned_iter = iter_idx
                                    print(
                                        f"[ADAPTIVE] Saved fine-tuned model from iteration {iter_idx + 1}, queue={current_best_queue:.2f}")

                                else:
                                    # 第2次及以后，比较当前最佳与全局最佳
                                    if current_best_queue < evolution_best_queue:
                                        # 当前更好，更新全局最佳
                                        print(
                                            f"[EVOLUTION] Better sample found! {evolution_best_queue:.2f} to {current_best_queue:.2f}")
                                        evolution_best_sample = current_best_sample.copy()
                                        evolution_best_queue = current_best_queue
                                        evolution_best_info = {
                                            'epoch': epoch,
                                            'round': j,
                                            'iter_idx': iter_idx,
                                            'sample_id': current_best_sample_id,
                                            'queue': current_best_queue
                                        }

                                        # 新的全局最佳网络状态，从评估结果中找到对应的网络状态
                                        current_best_network_state = None
                                        for result in sample_results:
                                            if result['sample_id'] == current_best_sample_id:
                                                current_best_network_state = result['network_state']
                                                break

                                        if current_best_network_state is not None:
                                            evolution_best_network_state = current_best_network_state
                                            # 覆盖保存到文件
                                            evolution_best_network_path = config.get_global_best_network_path(epoch, j)
                                            os.makedirs(os.path.dirname(evolution_best_network_path), exist_ok=True)
                                            torch.save(evolution_best_network_state, evolution_best_network_path)
                                            print(
                                                f"[EVOLUTION] Evolution best network state updated (iteration {iter_idx + 1})")
                                        else:
                                            print(
                                                f"[WARN] Network state not found for sample_id={current_best_sample_id}")

                                        # ===== 新增: 保存更优的微调模型 =====
                                        best_finetuned_model_state = {k: v.clone() for k, v in
                                                                      diffusion_model.state_dict().items()}
                                        best_finetuned_queue = current_best_queue
                                        best_finetuned_iter = iter_idx
                                        print(
                                            f"[ADAPTIVE] Updated fine-tuned model from iteration {iter_idx + 1}, queue={current_best_queue:.2f}")
                                    else:
                                        # 当前不如全局最佳，保持不变
                                        print(
                                            f"[EVOLUTION] Current best({current_best_queue:.2f}) >= Evolution best({evolution_best_queue:.2f}), keeping unchanged")
                            else:
                                print(
                                    f"[ITER {iter_idx + 1}] Warning: No valid queue data in current population, skipping best sample update")

                            # 按 q 独立同分布采样 SELECTION_SIZE 个样本
                            n_select = config.SELECTION_SIZE
                            p = df["q"].to_numpy(dtype=np.float64)
                            p = p / p.sum() if p.sum() > 0 else np.full_like(p, 1.0 / len(p))
                            idx = np.random.choice(len(p), size=n_select, replace=True, p=p)

                            # ===== 关键修复: 使用历史全局最佳样本进行微调（确保优化方向一致） =====
                            print(f"[ITER {iter_idx + 1}] Starting fine-tuning diffusion model with best sample...")

                            # 1) 选择微调目标：优先使用历史全局最佳，保证优化方向稳定
                            finetune_sample = None
                            finetune_queue = float('inf')
                            skip_finetune = False

                            if iter_idx == 0:
                                # 第一次迭代，需要有效的当前最佳样本才能微调
                                if current_best_sample is not None and current_best_queue != float('inf'):
                                    finetune_sample = current_best_sample.copy()
                                    finetune_queue = current_best_queue
                                    print(
                                        f"[ITER {iter_idx + 1}] Using current best sample for fine-tuning (queue={finetune_queue:.2f})")
                                else:
                                    skip_finetune = True
                                    print(f"[ITER {iter_idx + 1}] No valid current best sample, skipping fine-tuning")
                            else:
                                # 第2次及以后，始终使用历史全局最佳（关键改进）
                                if evolution_best_sample is not None:
                                    finetune_sample = evolution_best_sample.copy()
                                    finetune_queue = evolution_best_queue
                                    print(
                                        f"[ITER {iter_idx + 1}] Using historical best sample for fine-tuning (queue={finetune_queue:.2f})")
                                else:
                                    skip_finetune = True
                                    print(
                                        f"[ITER {iter_idx + 1}] No historical best sample available, skipping fine-tuning")

                            if not skip_finetune:
                                print(f"[ITER {iter_idx + 1}] Fine-tuning sample queue: {finetune_queue:.2f}")

                                # 关键修复: 增加微调强度，让模型充分学习最佳样本特征
                                # 在进化场景中，充分拟合最佳样本正是目标（而非过拟合）
                                n_finetune = config.FINETUNE_SAMPLES  # 从10增加到100
                                best_samples_n = np.tile(finetune_sample[np.newaxis, :, :], (n_finetune, 1, 1))
                                best_conditions_n = np.tile(expanded_target.cpu().numpy(), (n_finetune, 1, 1))

                                # 归一化最佳样本
                                if config.REWARD_NORMALIZE:
                                    best_samples_n = normalize_rewards(best_samples_n)

                                # 4) 计算权重（所有样本权重相同，因为都是最佳样本，权重值与训练一致）
                                best_weights_n = np.full((n_finetune,), 50.0, dtype=np.float32)  # 固定权重50.0

                                # 5) 转为张量
                                best_rewards_tensor = torch.tensor(best_samples_n, dtype=torch.float32).to(device)
                                best_conditions_tensor = torch.tensor(best_conditions_n, dtype=torch.float32).to(device)
                                best_weights_tensor = torch.tensor(best_weights_n, dtype=torch.float32).to(device)

                                # 6) 创建数据集和数据加载器
                                best_dataset = TrafficDataset(best_rewards_tensor, best_conditions_tensor,
                                                              best_weights_tensor)
                                best_dataloader = DataLoader(best_dataset, batch_size=config.DIFFUSION_BATCH_SIZE,
                                                             shuffle=True)

                                # 7) 使用分层学习率优化器进行微调
                                fine_tune_optimizer = optim.Adam([
                                    {'params': [p for n, p in diffusion_model.named_parameters() if 'lstm' in n],
                                     'lr': 0.00005},  # LSTM层：稳定，小幅调整
                                    {'params': [p for n, p in diffusion_model.named_parameters() if 'attn' in n],
                                     'lr': 0.0001},  # 注意力层：中等调整
                                    {'params': [p for n, p in diffusion_model.named_parameters() if 'fc' in n],
                                     'lr': 0.0002}  # 输出层：灵活调整
                                ])
                                print(f"[ITER {iter_idx + 1}] Starting layered fine-tuning of diffusion model")
                                # 关键修复: 增加训练轮数，让模型充分记忆最佳样本
                                # 100样本×20轮=2000次，在充分拟合和泛化能力间取得平衡
                                finetune_epochs = config.DIFFUSION_FINETUNE_EPOCHS  # 从5增加到20
                                train_diffusion_model(diffusion_model, best_dataloader, fine_tune_optimizer,
                                                      finetune_epochs, device)
                                print(f"[ITER {iter_idx + 1}] Layered fine-tuning completed")

                                # 清理微调过程产生的GPU张量（每次迭代后这些变量不再使用）
                                # 手动清理可加速显存释放，避免等待Python GC
                                del best_rewards_tensor, best_conditions_tensor, best_weights_tensor, best_dataset, best_dataloader, fine_tune_optimizer
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            else:
                                print(f"[ITER {iter_idx + 1}] Diffusion model requires no adjustment")

                            # ===== 精英样本扩散变异 =====
                            # 优化: 调整变异强度，在保持多样性和收敛速度间取得平衡
                            print(f"[ITER {iter_idx + 1}] Performing diffusion mutation on elite samples")

                            # 获取精英样本
                            elite_samples = current_population[idx]  # [SELECTION_SIZE, 1000, 4]

                            # 归一化精英样本（变异需要在归一化空间进行）
                            if config.REWARD_NORMALIZE:
                                elite_samples_normalized = normalize_rewards(elite_samples)
                            else:
                                elite_samples_normalized = elite_samples

                            # 转换为张量作为扩散变异起点
                            x0 = torch.tensor(elite_samples_normalized, dtype=torch.float32, device=device)

                            # 条件扩展到 [SELECTION_SIZE,1000,24]
                            cond = expanded_target.to(device)
                            if cond.dim() == 2:
                                cond = cond.unsqueeze(0)
                            cond = cond.expand(x0.size(0), cond.size(1), cond.size(2))

                            # 创建噪声调度器（复用训练时同一噪声调度）
                            noise_scheduler = CustomDDPMScheduler(
                                num_train_timesteps=config.NUM_TRAIN_TIMESTEPS,
                                beta_start=config.BETA_START,
                                beta_end=config.BETA_END,
                                beta_schedule=config.BETA_SCHEDULE,
                                device=device,
                                use_parallel=False
                            )

                            # 优化: 使用适中的变异强度（30步），避免过度变异导致无法收敛
                            t_start = config.MUTATION_STEPS  # 从100降到30
                            noise = torch.randn_like(x0)
                            x = noise_scheduler.add_noise(x0, noise,
                                                          torch.full((x0.size(0),), t_start, dtype=torch.long,
                                                                     device=device))

                            # 去噪序列：从t到t=1
                            timesteps = torch.arange(t_start, 0, -1, dtype=torch.long, device=device)

                            # 关键修复: 使用累积学习后的扩散模型进行去噪（不再恢复原始状态）
                            diffusion_model.eval()
                            with torch.no_grad():
                                for t in timesteps:
                                    t_int = int(t.item())
                                    t_batch = torch.full((x.size(0),), t_int, dtype=torch.long, device=device)
                                    pred_noise = diffusion_model(x, cond, t_batch)  # 使用自适应扩散模型
                                    x = noise_scheduler.step(pred_noise, t_int, x).prev_sample

                            # 反归一化变异后的样本
                            if config.REWARD_NORMALIZE:
                                x = denormalize_rewards(x)

                            # 保存变异后样本（保持浮点数）
                            x_float_np = x.detach().cpu().numpy()
                            np.save(all_samples_path, x_float_np)

                            # ===== 保存当前迭代的种群统计信息 =====
                            valid_queues = queues[np.isfinite(queues)]  # 过滤无效排队长度
                            if len(valid_queues) > 0:
                                # 迭代内（本次）的最优/最差/均值
                                current_min_queue = float(np.min(valid_queues))
                                worst_queue = float(np.max(valid_queues))
                                avg_queue = float(np.mean(valid_queues))
                                queue_std = float(np.std(valid_queues))  # 新增：种群标准差
                                diversity = worst_queue - current_min_queue  # 新增：种群多样性范围
                                selection_pressure = float(
                                    probs.max() / probs.mean()) if probs.mean() > 0 else 0.0  # 新增：选择压力

                                # 回合内（跨迭代）的最佳（仅用于本回合统计，避免覆盖全局历史最佳）
                                try:
                                    round_best_queue = float(min(current_min_queue, evolution_best_queue))
                                except Exception:
                                    # 若尚未产生 evolution_best_queue，则退回迭代内最优
                                    round_best_queue = current_min_queue

                                # 构造统计记录：将 best_queue 改为回合全局最优
                                stats_record = {
                                    'epoch': epoch,
                                    'round': j,
                                    'iter_idx': iter_idx + 1,  # 迭代次数从1开始
                                    'best_queue': round_best_queue,
                                    'worst_queue': worst_queue,
                                    'avg_queue': avg_queue,
                                    'queue_std': queue_std,
                                    'diversity': diversity,  # 新增
                                    'selection_pressure': selection_pressure,  # 新增
                                    'valid_samples': len(valid_queues),
                                    'total_samples': len(queues)
                                }

                                # 保存到CSV文件
                                stats_file = config.ITERATION_STATS_PATH
                                stats_lock_path = stats_file + ".lock"

                                with filelock.FileLock(stats_lock_path):
                                    if os.path.exists(stats_file):
                                        existing_stats_df = pd.read_csv(stats_file)
                                        stats_df = pd.concat([existing_stats_df, pd.DataFrame([stats_record])],
                                                             ignore_index=True)
                                    else:
                                        stats_df = pd.DataFrame([stats_record])

                                    # 限制行数并保存
                                    stats_df = stats_df.tail(100000)
                                    stats_df.to_csv(stats_file, index=False)

                                # 增强的日志输出
                                print(f"[ITER {iter_idx + 1}] 种群统计:")
                                print(f"  最优排队(回合全局): {round_best_queue:.2f}")
                                print(f"  平均排队: {avg_queue:.2f}")
                                print(
                                    f"  标准差: {queue_std:.2f} {'[OK]' if queue_std > 5 else '[WARNING] 多样性不足'}")
                                print(f"  多样性范围: {diversity:.2f} (最差-最优)")
                                print(
                                    f"  选择压力: {selection_pressure:.2f}x {'[OK]' if selection_pressure > 5 else '[WARNING] 压力不足'}")
                            else:
                                print(f"[ITER {iter_idx + 1}] Warning: No valid queue data in current population")

                            # 关键修复: 移除模型恢复逻辑，保留微调后的知识，让模型持续累积学习
                            # 这样每次迭代都会让模型更加专注于生成最优样本
                            print("------------------------------------")
                            print(f"Iteration {iter_idx + 1} completed, model state preserved for accumulation")
                            print("------------------------------------")

                        # —— 进化迭代循环结束，清理最后一轮产生的GPU张量 ——
                        # 这些变量在最后一次迭代后不再需要，清理以节省显存
                        if 'x0' in locals():
                            del x0
                        if 'cond' in locals():
                            del cond
                        if 'x' in locals():
                            del x
                        if 'noise' in locals():
                            del noise
                        if 'timesteps' in locals():
                            del timesteps
                        if 'noise_scheduler' in locals():
                            del noise_scheduler
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        # —— 最终评估所有样本并选最优 ——
                        final_samples = np.load(all_samples_path)  # [POPULATION_SIZE,1000,4]

                        # 1) 快照“当前训练网络”的权重，确保每个样本的副本起点一致
                        base_state_dict = {
                            tlID: {
                                "policy": {k: v.detach().cpu().clone() for k, v in
                                           train_learners[tlID].policy_net.state_dict().items()},
                                "target": {k: v.detach().cpu().clone() for k, v in
                                           train_learners[tlID].target_net.state_dict().items()},
                                "learn_steps": train_learners[tlID].learn_steps,
                                "optimizer": {
                                    k: v.cpu() if torch.is_tensor(v) else v
                                    for k, v in train_learners[tlID].optimizer.state_dict().items()
                                }  # 确保优化器状态也在CPU上
                            }
                            for tlID in train_env.get_trafficlights_ID_list()
                        }
                        os.makedirs(config.WEIGHTS_DIR, exist_ok=True)
                        final_base_state_path = config.get_final_base_state_path(epoch, j)
                        torch.save(base_state_dict, final_base_state_path)

                        # 2) 构造任务：为每个样本准备经验
                        tasks = []
                        tl_ids = list(train_env.get_trafficlights_ID_list())  # 转换为list以支持pickle序列化
                        for sample_id in range(final_samples.shape[0]):  # POPULATION_SIZE
                            rewards_array = final_samples[sample_id]  # [1000,4]
                            exp_per_tl = train_env.build_experiences_from_rewards_array(j, rewards_array)
                            tasks.append((sample_id, exp_per_tl))

                        # 3) 分批并行评估，避免资源耗尽
                        sample_results = []  # [{epoch, round, sample_id, average_queue}]
                        batch_size = config.EVALUATION_BATCH_SIZE
                        total_tasks = len(tasks)
                        num_batches = (total_tasks + batch_size - 1) // batch_size

                        print(f"[FINAL] Evaluating {total_tasks} final samples in {num_batches} batches")

                        # 分批提交任务
                        for batch_idx in range(num_batches):
                            start_idx = batch_idx * batch_size
                            end_idx = min(start_idx + batch_size, total_tasks)
                            batch_tasks = tasks[start_idx:end_idx]

                            print(
                                f"[FINAL] Processing batch {batch_idx + 1}/{num_batches} (samples {start_idx}-{end_idx - 1})")

                            # 提交当前批次的任务
                            futures = [
                                evaluate_single_sample.remote(
                                    sample_id,
                                    exp_per_tl,
                                    final_base_state_path,
                                    epoch,
                                    j,
                                    tl_ids
                                )
                                for sample_id, exp_per_tl in batch_tasks
                            ]

                            # 等待当前批次完成
                            batch_results = ray.get(futures)
                            sample_results.extend(batch_results)

                            for res in batch_results:
                                try:
                                    print(
                                        f"[FINAL epoch={res['epoch']} round={res['round']}] sample_id={res['sample_id']}, avg_inner_sum={res['avg_inner_sum']:.2f}")
                                except Exception:
                                    pass

                            print(f"[FINAL] Batch {batch_idx + 1}/{num_batches} completed")

                            # 批次间延迟
                            if batch_idx < num_batches - 1:
                                print(f"[FINAL] Waiting {config.INTER_BATCH_DELAY}s for process cleanup...")
                                time.sleep(config.INTER_BATCH_DELAY)

                        # 清理最终评估进程池后的残留
                        subprocess.call(['pkill', '-f', 'sumo'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        print(f"[FINAL] All {total_tasks} final samples evaluated successfully")

                        # 4) 选择最内圈四边之和最低的样本
                        df_final_eval = pd.DataFrame(sample_results)
                        df_final_eval = df_final_eval.sort_values('sample_id').reset_index(drop=True)  # 按sample_id排序！
                        queues = pd.to_numeric(df_final_eval["avg_inner_sum"], errors="coerce").to_numpy(
                            dtype=np.float64)
                        queues_safe = np.where(np.isfinite(queues), queues, np.inf)
                        final_best_pos = int(np.argmin(queues_safe)) if len(queues_safe) > 0 else 0
                        final_best_sample_id = final_best_pos  # 排序后，索引就是sample_id
                        final_best_sample = final_samples[final_best_sample_id]  # [1000,4]
                        final_best_queue = queues_safe[final_best_pos]
                        print(f"Final population: sample_id={final_best_sample_id}, inner_sum={final_best_queue:.2f}")

                        # **新方案：直接比较网络状态，选择排队更低的**
                        if evolution_best_sample is not None:
                            print(
                                f"Evolution best queue: iteration {evolution_best_info['iter_idx'] + 1}, queue={evolution_best_queue:.2f}")

                            # 获取最后一轮最佳样本的网络状态
                            final_best_network_state = None
                            for result in sample_results:
                                if result['sample_id'] == final_best_sample_id:
                                    final_best_network_state = result['network_state']
                                    break

                            if final_best_network_state is not None:
                                # 加载进化过程最佳网络状态
                                evolution_best_network_path = config.get_global_best_network_path(epoch, j)
                                # DEBUG: show path details before loading evolution best
                                try:
                                    print(f"[DEBUG] CWD={os.getcwd()}")
                                except Exception:
                                    pass
                                try:
                                    _abs = os.path.abspath(evolution_best_network_path)
                                    _exists = os.path.exists(evolution_best_network_path)
                                    print(
                                        f"[DEBUG] Evolution best path={evolution_best_network_path}, abs={_abs}, exists={_exists}")
                                except Exception:
                                    pass
                                # 尝试从文件加载进化过程最佳网络；缺失时使用内存或最后一轮最佳
                                evolution_best_network_state_loaded = None
                                if os.path.exists(evolution_best_network_path):
                                    evolution_best_network_state_loaded = torch.load(
                                        evolution_best_network_path, map_location=device
                                    )
                                    try:
                                        print("[DEBUG] Evolution best network loaded successfully")
                                    except Exception:
                                        pass
                                else:
                                    mem_has = 'evolution_best_network_state' in locals() and evolution_best_network_state is not None
                                    if mem_has:
                                        evolution_best_network_state_loaded = evolution_best_network_state
                                        print(
                                            "[WARN] Evolution best file missing; using in-memory evolution best state")
                                    else:
                                        evolution_best_network_state_loaded = final_best_network_state
                                        print(
                                            "[WARN] Evolution best file missing and no memory state; using last-round best")

                                # 直接比较排队长度，选择更优的网络状态
                                if final_best_queue < evolution_best_queue:
                                    chosen_network_state = final_best_network_state
                                    chosen_queue = final_best_queue
                                    best_source = f"Last round population network (queue={final_best_queue:.2f})"
                                    print(
                                        f"Final choice: Last round population network (queue={final_best_queue:.2f} < {evolution_best_queue:.2f})")
                                else:
                                    chosen_network_state = evolution_best_network_state_loaded
                                    chosen_queue = evolution_best_queue
                                    best_source = f"Evolution process best network (queue={evolution_best_queue:.2f})"
                                    print(
                                        f"Starting save: Evolution process best network (queue={evolution_best_queue:.2f} <= {final_best_queue:.2f})")
                            else:
                                print(
                                    f"[WARN] Network state not found for last round sample_id={final_best_sample_id}, using evolution best")
                                evolution_best_network_path = config.get_global_best_network_path(epoch, j)
                                # DEBUG: show path details before loading evolution best (fallback)
                                try:
                                    print(f"[DEBUG] CWD={os.getcwd()}")
                                except Exception:
                                    pass
                                try:
                                    _abs = os.path.abspath(evolution_best_network_path)
                                    _exists = os.path.exists(evolution_best_network_path)
                                    print(
                                        f"[DEBUG] Evolution best path={evolution_best_network_path}, abs={_abs}, exists={_exists}")
                                except Exception:
                                    pass
                                if os.path.exists(evolution_best_network_path):
                                    chosen_network_state = torch.load(
                                        evolution_best_network_path, map_location=device
                                    )
                                    try:
                                        print("[DEBUG] Evolution best network loaded successfully (fallback)")
                                    except Exception:
                                        pass
                                else:
                                    mem_has = 'evolution_best_network_state' in locals() and evolution_best_network_state is not None
                                    if mem_has:
                                        chosen_network_state = evolution_best_network_state
                                        print(
                                            "[WARN] Evolution best file missing; using in-memory evolution best state (fallback)")
                                    else:
                                        raise FileNotFoundError(
                                            f"Evolution best state not found at {evolution_best_network_path} and no in-memory state available")
                                chosen_queue = evolution_best_queue
                                best_source = f"Evolution best network (last round network state missing)"
                        else:
                            # 没有全局最佳，使用最后一轮最佳
                            final_best_network_state = None
                            for result in sample_results:
                                if result['sample_id'] == final_best_sample_id:
                                    final_best_network_state = result['network_state']
                                    break

                            if final_best_network_state is not None:
                                chosen_network_state = final_best_network_state
                                chosen_queue = final_best_queue
                                best_source = f"Last round population network (no historical best comparison)"
                                print(f"Final choice: Last round population network (no historical best network)")
                            else:
                                print(f"[ERROR] No available network state found, skipping network update")
                                continue  # 跳过本回合

                        # **直接复制选定的网络状态给主网络**
                        prev_main_state = {}
                        for tlID in train_env.get_trafficlights_ID_list():
                            prev_main_state[tlID] = {
                                "policy": {k: v.detach().cpu().clone() for k, v in
                                           train_learners[tlID].policy_net.state_dict().items()},
                                "target": {k: v.detach().cpu().clone() for k, v in
                                           train_learners[tlID].target_net.state_dict().items()},
                                "learn_steps": train_learners[tlID].learn_steps,
                                "steps_done": train_learners[tlID].steps_done,
                                "best_loss": train_learners[tlID].best_loss,
                                "optimizer": {
                                    k: (v.cpu() if torch.is_tensor(v) else v)
                                    for k, v in train_learners[tlID].optimizer.state_dict().items()
                                },
                            }

                        print(f"Copying selected network state to main network: {best_source}")
                        for tlID in train_env.get_trafficlights_ID_list():
                            # 复制网络权重
                            train_learners[tlID].policy_net.load_state_dict(chosen_network_state[tlID]["policy"])
                            train_learners[tlID].target_net.load_state_dict(chosen_network_state[tlID]["target"])

                            # 复制学习状态
                            train_learners[tlID].learn_steps = chosen_network_state[tlID]["learn_steps"]
                            train_learners[tlID].steps_done = chosen_network_state[tlID]["steps_done"]  # 新增
                            train_learners[tlID].best_loss = chosen_network_state[tlID]["best_loss"]  # 新增

                            # 复制优化器状态（确保在正确设备上）
                            optimizer_state = chosen_network_state[tlID]["optimizer"]
                            for state in optimizer_state['state'].values():
                                for k, v in state.items():
                                    if torch.is_tensor(v):
                                        state[k] = v.to(device)
                            train_learners[tlID].optimizer.load_state_dict(optimizer_state)
                            torch.save(train_learners[tlID].policy_net.state_dict(),
                                       train_learners[tlID].best_model_path)

                        print("Main network state update completed")

                        # 找到选定的最佳样本结果
                        chosen_sample_result = None
                        if final_best_queue < evolution_best_queue if evolution_best_sample is not None else True:
                            # 使用最后一轮最佳样本
                            for result in sample_results:
                                if result['sample_id'] == final_best_sample_id:
                                    chosen_sample_result = result
                                    break
                        else:
                            # 使用历史全局最佳样本（需要构造结果对象）
                            chosen_sample_result = {
                                'sample_id': evolution_best_info['sample_id'],
                                'epoch': evolution_best_info['iter_idx'],
                                'round': j
                            }

                        # ===== 在网络更新后立即进行独立评估 =====
                        print("===== Starting independent evaluation of updated network =====")
                        port_eval = find_free_port()
                        eval_env = SUMOTrafficLights(config.SUMO_CONFIG_PATH, port_eval, False, 32)

                        eval_learners = {}
                        eval_replay_buffers = {}
                        for tlID in train_env.get_trafficlights_ID_list():
                            eval_replay_buffers[tlID] = ReplayBuffer(capacity=10000)
                            eval_learner = QLearner(
                                tlID, eval_env, 0, 0, 0.01, 0.01, 'eval',
                                eval_replay_buffers[tlID], 15, 0, 0, 1, 5, 2, 32,
                                None, device
                            )
                            # 复制刚更新的主网络状态到评估网络
                            eval_learner.policy_net.load_state_dict(train_learners[tlID].policy_net.state_dict())
                            eval_learner.target_net.load_state_dict(train_learners[tlID].target_net.state_dict())
                            eval_learner.policy_net.eval()
                            eval_learner.target_net.eval()
                            eval_learners[tlID] = eval_learner

                        eval_env.learners = eval_learners
                        eval_env.replay_buffers = eval_replay_buffers

                        print("Starting evaluation mode simulation")
                        eval_metrics = eval_env.run_episode(config.SIMULATION_DURATION_SEC, None, epoch, 'eval')

                        # 使用单方向最差尾段鲁棒与方向级稳定性
                        robust = eval_metrics.get('avg_tail_robust')
                        p95 = eval_metrics.get('p95_tail_robust')
                        mx = eval_metrics.get('max_tail_robust')
                        tail_dir_avg = eval_metrics.get('avg_tail_robust_per_road') or []
                        tail_dir_p95 = eval_metrics.get('p95_tail_robust_per_road') or []
                        tail_dir_max = eval_metrics.get('max_tail_robust_per_road') or []

                        if tail_dir_avg:
                            worst_tail_dir_avg = max([v for v in tail_dir_avg if v is not None], default=None)
                        else:
                            worst_tail_dir_avg = robust

                        if worst_tail_dir_avg is not None:
                            current_queue = float(worst_tail_dir_avg)
                            print(
                                f"Independent evaluation: worst_tail_dir_avg = {current_queue:.2f}, overall avg_tail={robust}, p95={p95}, max={mx}, tail_steps = {eval_metrics.get('tail_steps')}")
                        else:
                            current_queue = float(eval_metrics.get('average_queue_inner_sum') or 0.0)
                            print(f"Independent evaluation: fallback avg_inner_sum = {current_queue:.2f}")

                        stable = True
                        try:
                            if robust is not None and p95 is not None and mx is not None:
                                if p95 > config.ROBUST_P95_FACTOR * robust or mx > config.ROBUST_MAX_FACTOR * robust:
                                    stable = False
                        except Exception:
                            pass
                        stable_dir = True
                        try:
                            for i in range(min(4, len(tail_dir_avg))):
                                a = tail_dir_avg[i]
                                p = tail_dir_p95[i] if i < len(tail_dir_p95) else None
                                m = tail_dir_max[i] if i < len(tail_dir_max) else None
                                if a is None or p is None or m is None:
                                    continue
                                if p > getattr(config, 'ROBUST_P95_FACTOR', 1.3) * a or m > getattr(config, 'ROBUST_MAX_FACTOR', 1.6) * a:
                                    stable_dir = False
                                    break
                        except Exception:
                            pass
                        if current_queue < global_best_queue and stable and stable_dir:
                            global_best_queue, global_best_epoch = save_best_network(
                                train_learners, current_queue, epoch, global_best_queue, global_best_epoch
                            )
                        else:
                            if not stable or not stable_dir:
                                print(f"[SELECTION] Skip update due to instability: overall(p95={p95}, max={mx}, mean={robust}), per-dir stable={stable_dir}")
                            else:
                                print(
                                    f"[SELECTION] Chosen eval metric ({current_queue:.2f}) >= historical best ({global_best_queue:.2f}), restoring previous main network")
                            for tlID in train_env.get_trafficlights_ID_list():
                                train_learners[tlID].policy_net.load_state_dict(prev_main_state[tlID]["policy"])
                                train_learners[tlID].target_net.load_state_dict(prev_main_state[tlID]["target"])
                                train_learners[tlID].learn_steps = prev_main_state[tlID]["learn_steps"]
                                train_learners[tlID].steps_done = prev_main_state[tlID]["steps_done"]
                                train_learners[tlID].best_loss = prev_main_state[tlID]["best_loss"]
                                optimizer_state = prev_main_state[tlID]["optimizer"]
                                for state in optimizer_state['state'].values():
                                    for k, v in state.items():
                                        if torch.is_tensor(v):
                                            state[k] = v.to(device)
                                train_learners[tlID].optimizer.load_state_dict(optimizer_state)

                        # 5) 保存最佳样本数据到Excel（用于记录）
                        if 'evolution_best_sample' in locals() and evolution_best_sample is not None and chosen_queue == evolution_best_queue:
                            ultimate_best_sample = evolution_best_sample
                        else:
                            ultimate_best_sample = final_best_sample

                        gen_df = pd.DataFrame(ultimate_best_sample.reshape(-1, 4),
                                              columns=["reward0", "reward1", "reward2", "reward3"])
                        gen_path = config.GENERATED_REWARD_PATH
                        gen_df.to_excel(gen_path, index=False, engine='openpyxl')

                        # 清理eval_env
                        try:
                            eval_env.close()
                            del eval_env, eval_learners, eval_replay_buffers
                            # 清理独立评估产生的GPU张量
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except:
                            pass

                        # 新增：用最佳样本奖励值替换CPMData中的0占位符
                        if chosen_sample_result is not None:
                            train_env.write_cpm_data_from_sample_result(chosen_sample_result, epoch, j)
                        else:
                            print("[WARN] Selected sample result not found, skipping CPMData write")

                        # ===== 新增: Episode级最佳模型保存机制 =====
                        # 如果当前Episode的最佳排队长度优于历史基线,保存微调后的模型
                        episode_best_queue = chosen_queue  # 当前Episode最佳排队长度

                        # 尝试加载历史最佳排队长度（如果存在）
                        epoch_baseline_queue = float('inf')
                        if os.path.exists(config.BEST_DDP_MODEL_PATH):
                            # 从上一个Epoch的记录中获取基线延误
                            baseline_record_path = os.path.join(config.CPM_DIR, 'best_diffusion_baseline.txt')
                            if os.path.exists(baseline_record_path):
                                try:
                                    with open(baseline_record_path, 'r') as f:
                                        epoch_baseline_queue = float(f.read().strip())
                                        print(f"[ADAPTIVE] Loaded Epoch baseline queue: {epoch_baseline_queue:.2f}")
                                except:
                                    print("[ADAPTIVE] Failed to load baseline queue, using inf")

                        # 比较Episode最佳与Epoch基线
                        if best_finetuned_model_state is not None and episode_best_queue < epoch_baseline_queue:
                            # Episode模型更优,保存微调后的模型状态
                            episode_model_path = os.path.join(config.CPM_DIR, f'episode_best_epoch{epoch}_round{j}.pth')

                            # 保存最佳微调模型（在reset之前已保存）
                            torch.save(best_finetuned_model_state, episode_model_path)

                            print(f"[ADAPTIVE] Episode best model saved!")
                            print(f"[ADAPTIVE]   - Iteration: {best_finetuned_iter + 1}/{config.EVOLUTION_ITERATIONS}")
                            print(
                                f"[ADAPTIVE]   - Queue: {best_finetuned_queue:.2f} (baseline: {epoch_baseline_queue:.2f})")
                            print(
                                f"[ADAPTIVE]   - Improvement: {epoch_baseline_queue - best_finetuned_queue:.2f} ({((epoch_baseline_queue - best_finetuned_queue) / epoch_baseline_queue * 100):.2f}%)")
                            print(f"[ADAPTIVE]   - Saved to: {episode_model_path}")

                            # 如果显著优于基线(>5%改进),覆盖BEST_DDP_MODEL_PATH
                            improvement_ratio = (epoch_baseline_queue - best_finetuned_queue) / epoch_baseline_queue
                            if improvement_ratio > 0.05:
                                torch.save(best_finetuned_model_state, config.BEST_DDP_MODEL_PATH)
                                # 更新基线记录
                                with open(baseline_record_path, 'w') as f:
                                    f.write(str(best_finetuned_queue))
                                print(
                                    f"[ADAPTIVE] Significant improvement! Updated global best model (>{improvement_ratio * 100:.1f}% better)")
                        else:
                            if best_finetuned_model_state is None:
                                print(f"[ADAPTIVE] WARNING: No fine-tuned model available (no evolution improvement)")
                            else:
                                print(
                                    f"[ADAPTIVE] Episode best queue={episode_best_queue:.2f} >= Epoch baseline={epoch_baseline_queue:.2f}")
                                print(f"[ADAPTIVE] No improvement over baseline, keeping current best model")

                        # 清理扩散模型和相关变量
                        # print(f"[EVOLUTION] Evolution iteration completed, cleaning up diffusion model...")
                        del diffusion_model, expanded_target, sa_tensor
                        if 'best_finetuned_model_state' in locals() and best_finetuned_model_state is not None:
                            del best_finetuned_model_state
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # print(f"[EVOLUTION] Adaptive diffusion model cleaned up")

        # 每个 epoch 都训练条件扩散模型（轮数从config读取）
        input_filepath = config.CPM_DATA_PATH
        SRData = pd.read_csv(input_filepath)

        block_size = 1000

        # 从 state_reward_* 解析 reward，并去掉末尾元素
        for i in range(4):
            reward_col = f'reward_{i}'
            state_reward_col = f'state_reward_{i}'
            SRData[reward_col] = SRData[state_reward_col].apply(lambda x: x.split(',')[-1].strip())

        for i in range(4):
            state_reward_col = f'state_reward_{i}'
            SRData[state_reward_col] = SRData[state_reward_col].apply(lambda x: ','.join(x.split(',')[:-1]).strip())

        # 条件向量 (s+a)
        sa_lists = []
        for col in ['state_reward_0', 'state_reward_1', 'state_reward_2', 'state_reward_3']:
            sa_lists.append(parse_state_column(SRData[col]))

        sa_list = []
        for i in range(len(sa_lists[0])):
            combined_state = sa_lists[0][i] + sa_lists[1][i] + sa_lists[2][i] + sa_lists[3][i]
            sa_list.append(combined_state)

        # 奖励矩阵
        rewards = []
        for col in ['reward_0', 'reward_1', 'reward_2', 'reward_3']:
            rewards.append(parse_state_column(SRData[col]))
        rewards = np.concatenate(rewards, axis=1)

        # 以 1000 步为一个 block
        n_samples = rewards.shape[0]
        num_blocks = n_samples // block_size

        sa_arr = np.array(sa_list, dtype=np.float32)[: num_blocks * block_size, :]
        sa_arr = sa_arr.reshape((num_blocks, block_size, sa_arr.shape[1]))
        sa_tensor = torch.tensor(sa_arr, dtype=torch.float32).to(device)

        rewards = rewards[:num_blocks * block_size, :]
        input_rewards = rewards.reshape((num_blocks, block_size, rewards.shape[1]))

        # 归一化rewards到[-1, 1]，使其与DDPM噪声N(0,1)在同一尺度
        if config.REWARD_NORMALIZE:
            input_rewards = normalize_rewards(input_rewards)

        rewards_tensor = torch.tensor(input_rewards, dtype=torch.float32).to(device)

        # 基于 avg_inner_sum 的权重（排队越少权重越大），先归一到 (0,1)，再乘以 50
        queue_file = config.QUEUE_SUMMARY_PATH
        if os.path.exists(queue_file):
            queue_df = pd.read_excel(queue_file)
            queues = queue_df['avg_inner_sum'].values.astype(
                np.float32) if 'avg_inner_sum' in queue_df.columns else np.array([], dtype=np.float32)
        else:
            queues = np.array([], dtype=np.float32)

        if len(queues) == 0:
            # 情况1: 没有排队数据，使用均匀权重
            block_weights = np.ones((num_blocks,), dtype=np.float32) * 25.0
            print(f"[WEIGHT] No queue data, using uniform weights (25.0)")
        elif len(queues) < num_blocks * 0.5:
            # 情况2: 排队数据严重不足（少于一半），使用均匀权重避免过拟合
            block_weights = np.ones((num_blocks,), dtype=np.float32) * 25.0
            print(f"[WEIGHT] Insufficient queue data ({len(queues)} < {num_blocks * 0.5:.0f}), using uniform weights")
        else:
            # 情况3: 排队数据充足或基本充足
            if len(queues) >= num_blocks:
                # 3a. 数据充足：使用最近的num_blocks个数据
                use_queues = queues[-num_blocks:]
                print(f"[WEIGHT] Using last {num_blocks} queue values for weight calculation")
            else:
                # 3b. 数据不足但超过一半：使用插值填充而非重复填充
                print(f"[WEIGHT] Interpolating {len(queues)} queue values to {num_blocks} blocks")
                # 使用线性插值生成权重
                from scipy.interpolate import interp1d
                x_orig = np.linspace(0, 1, len(queues))
                x_new = np.linspace(0, 1, num_blocks)
                interp_func = interp1d(x_orig, queues, kind='linear', fill_value='extrapolate')
                use_queues = interp_func(x_new).astype(np.float32)

            # 计算基于排队的权重（排队越少权重越高）
            inv = 1.0 / (use_queues + 1e-6)
            inv_min, inv_max = inv.min(), inv.max()
            if inv_max > inv_min:
                scaled = (inv - inv_min) / (inv_max - inv_min + 1e-12)  # 归一化到 (0,1)
            else:
                scaled = np.ones_like(inv)  # 所有排队相等时使用均匀权重
            block_weights = (scaled * 50.0).astype(np.float32)  # 权重范围 [0, 50.0]

            # 打印权重统计信息
            print(f"[WEIGHT] Weight stats: min={block_weights.min():.2f}, max={block_weights.max():.2f}, "
                  f"mean={block_weights.mean():.2f}, unique={len(np.unique(block_weights))}/{len(block_weights)}")

        weights = torch.tensor(block_weights, dtype=torch.float32).to(device)

        input_dim = input_rewards.shape[2]
        condition_dim = sa_tensor.shape[2]
        hidden_dim = config.HIDDEN_DIM
        num_epochs = num_epochs_0
        batch_size = config.DIFFUSION_BATCH_SIZE
        learning_rate = config.DIFFUSION_LEARNING_RATE
        num_samples = 1

        # 进行epoch级别的扩散模型训练
        epoch_model = DiffusionModel(input_dim, condition_dim, hidden_dim, dropout=config.DROPOUT_RATE).to(device)
        optimizer = optim.Adam(epoch_model.parameters(), lr=learning_rate)

        dataset = TrafficDataset(rewards_tensor, sa_tensor, weights)

        # ===== 新增：划分训练集和验证集 (80/20 split) =====
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

        print(f"[TRAINING] Epoch {epoch} - Training diffusion model for {num_epochs} epochs")
        print(f"[TRAINING] Dataset split: {train_size} train samples, {val_size} val samples")

        # 使用验证集进行训练
        train_diffusion_model(epoch_model, train_loader, optimizer, num_epochs, device, val_dataloader=val_loader)

        print(f"[TRAINING] Epoch {epoch} - Diffusion model training completed")

        # ===== 新增：保存Epoch训练后的基线排队长度 =====
        # 从queue_summary中获取最近的最佳排队长度作为下一轮的基线（基于avg_inner_sum）
        if os.path.exists(config.QUEUE_SUMMARY_PATH):
            try:
                queue_df = pd.read_excel(config.QUEUE_SUMMARY_PATH)
                if len(queue_df) > 0 and 'avg_inner_sum' in queue_df.columns:
                    # 取最近10个episode的最小排队长度作为基线（内圈四边之和）
                    recent_queues = queue_df['avg_inner_sum'].tail(10).values
                    baseline_queue = float(np.min(recent_queues))
                    baseline_record_path = os.path.join(config.CPM_DIR, 'best_diffusion_baseline.txt')
                    os.makedirs(os.path.dirname(baseline_record_path), exist_ok=True)
                    with open(baseline_record_path, 'w') as f:
                        f.write(str(baseline_queue))
                    print(f"[TRAINING] Updated baseline queue: {baseline_queue:.2f} (from recent 10 episodes)")
            except Exception as e:
                print(f"[TRAINING] Failed to update baseline queue: {e}")

        endtime = datetime.now().time()
        # 计算秒数差
        end_seconds = endtime.hour * 3600 + endtime.minute * 60 + endtime.second
        spendtime_seconds = end_seconds - start_seconds
        print(f'Duration: {spendtime_seconds} seconds')

        # 写入下一次应从哪个 epoch 开始
        try:
            epoch_marker_file = config.EPOCH_MARKER_PATH
            os.makedirs(os.path.dirname(epoch_marker_file), exist_ok=True)
            with open(epoch_marker_file, 'w', encoding='utf-8') as f:
                f.write(str(epoch + 1))
        except Exception as e:
            print(f"Failed to write checkpoint: {e}")

        # ===== Epoch结束后强制资源清理 =====
        if config.SHOW_RESOURCE_CLEANUP_LOGS:
            print(f"[EPOCH {epoch}] Starting forced resource cleanup...")

        # 1. 强制终止所有SUMO进程
        try:
            if os.name == 'nt':  # Windows
                subprocess.call(['taskkill', '/f', '/im', 'sumo.exe'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.call(['taskkill', '/f', '/im', 'sumo-gui.exe'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"[EPOCH {epoch}] Windows SUMO process forced cleanup completed")
            else:  # Linux/Unix
                subprocess.call(['pkill', '-f', 'sumo'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if config.SHOW_RESOURCE_CLEANUP_LOGS:
                    print(f"[EPOCH {epoch}] Linux SUMO process forced cleanup completed")
        except Exception as e:
            if config.SHOW_RESOURCE_CLEANUP_LOGS:
                print(f"[EPOCH {epoch}] SUMO process cleanup failed: {e}")

        # 2. 强化垃圾回收
        import gc
        for i in range(5):
            gc.collect()
        if config.SHOW_RESOURCE_CLEANUP_LOGS:
            print(f"[EPOCH {epoch}] Enhanced garbage collection completed")

        # 3. 显示当前系统资源使用情况
        if config.SHOW_RESOURCE_CLEANUP_LOGS:
            try:
                current_process = psutil.Process(os.getpid())
                memory_info = current_process.memory_info()
                num_threads = current_process.num_threads()

                # 获取系统内存信息
                system_memory = psutil.virtual_memory()

                print(f"[EPOCH {epoch}] System resource status:")
                print(
                    f"  - Current process memory: RSS={memory_info.rss / 1024 / 1024:.1f}MB, VMS={memory_info.vms / 1024 / 1024:.1f}MB")
                print(f"  - Current process thread count: {num_threads}")
                print(
                    f"  - System memory usage: {system_memory.percent:.1f}% ({system_memory.used / 1024 / 1024 / 1024:.1f}GB/{system_memory.total / 1024 / 1024 / 1024:.1f}GB)")
            except Exception as e:
                print(f"[EPOCH {epoch}] Resource status display failed: {e}")

        # 4. 资源清理完成
        if config.SHOW_RESOURCE_CLEANUP_LOGS:
            print(f"[EPOCH {epoch}] Resource cleanup completed")

    # ===== 程序结束前清理主进程资源 =====
    try:
        if 'train_env' in locals():
            # print("[CLEANUP] Closing main process train_env...")
            train_env.close()
            del train_env
    except Exception as e:
        print(f"Failed to cleanup main process train_env: {e}")

    # print("[CLEANUP] Main process resource cleanup completed")

    # 关闭Ray
    if ray.is_initialized():
        ray.shutdown()


if __name__ == '__main__':
    main()
