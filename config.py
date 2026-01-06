"""
Simple configuration file for traffic signal optimization
"""

import torch

# ==================== 基础设置 ====================
# 是否强制使用 CPU（True 时忽略 GPU）
FORCE_CPU = False

# 允许使用的 CUDA 设备索引（明确禁用 0 号设备）
GPU_CANDIDATES = [0]

if FORCE_CPU:
    DEVICE = torch.device("cpu")
else:
    if torch.cuda.is_available():
        try:
            count = torch.cuda.device_count()
        except Exception:
            count = 0
        chosen = None
        for idx in GPU_CANDIDATES:
            if isinstance(idx, int) and 0 <= idx < count:
                chosen = idx
                break
        # 如果没有可用的 1/2/3 号设备，则回退 CPU（不使用 0 号设备）
        DEVICE = torch.device(f"cuda:{chosen}") if chosen is not None else torch.device("cpu")
    else:
        DEVICE = torch.device("cpu")
SUMO_CONFIG_PATH = 'nets_2/22grid_fuben_e1_1012.sumocfg'

# ==================== 训练参数 ====================
EPOCHS = 50
ROUNDS_PER_EPOCH = 3
START_EPOCH = 0

# ==================== 强化学习参数 ====================
RL_LEARNING_RATE = 0.001
GAMMA = 0.95
EPSILON_START = 0.9
EPSILON_END = 0.01
EPSILON_DECAY = 0.999
RL_BATCH_SIZE = 32
MEMORY_SIZE = 50000

# ==================== 扩散模型参数 ====================
DIFFUSION_LEARNING_RATE = 0.0001
DIFFUSION_BATCH_SIZE = 32  # Reduced from 32 to save memory
DIFFUSION_EPOCHS_MAIN = 500  # 第一次训练轮数（与 main.py 一致）
DIFFUSION_EPOCHS_REGULAR = 100  # 后续训练轮数（与 main.py 一致）
DIFFUSION_FINETUNE_EPOCHS = 20  # 微调轮数（从5增加到20）
FINETUNE_SAMPLES = 100  # 微调样本数（从10增加到100）
ENABLE_FAST_FINETUNE = True  # 是否启用快速微调模式

# ==================== 模型结构参数 ====================
INPUT_DIM = 4  # 奖励维度
CONDITION_DIM = 24  # 状态+动作维度
HIDDEN_DIM = 256
NUM_HEADS = 8
NUM_LSTM_LAYERS = 1
DROPOUT_RATE = 0.3

# ==================== 扩散调度器参数 ====================
NUM_TRAIN_TIMESTEPS = 1000
BETA_START = 0.0001
BETA_END = 0.02
BETA_SCHEDULE = "linear"
NUM_INFERENCE_STEPS = 15

# ==================== 进化算法参数 ====================
EVOLUTION_ITERATIONS = 10  # 从20降到10，减少计算开销，配合早停机制
POPULATION_SIZE = 500
SELECTION_SIZE = 300  # 从500降到300，增加选择压力，保留60%精英
MUTATION_STEPS = 30  # 变异扩散步数（从100降到30，平衡变异强度）

# ==================== 多目标优化设置 ====================
# 目标：最内圈四条路（top,right,bottom,left）的平均排队长度，越小越好
# 固定采用 Pareto 非支配排序 + 拥挤距离 概率映射
# 帕累托概率映射超参
PARETO_ALPHA = 1.0   # rank 衰减强度（越大越偏好低rank）
PARETO_BETA = 1e-3   # crowding 距离的平滑项，避免全零

# ==================== 并发控制参数 ====================
EVALUATION_BATCH_SIZE = 200  # 每批评估的样本数（分批处理500个样本，避免资源耗尽）
INTER_BATCH_DELAY = 2.0  # 批次间延迟（秒），确保进程清理完毕

# ==================== 文件路径 ====================
DATA_DIR = "data"
WEIGHTS_DIR = "weights"
CPM_DIR = "diffusion"

# 具体文件路径
SA_SEQ_PATH = f"{DATA_DIR}/SA_seq_ep.xlsx"
GENERATED_REWARD_PATH = f"{DATA_DIR}/generated_reward.xlsx"
TRAINING_LOSSES_PATH = f"{DATA_DIR}/training_losses.xlsx"
EPOCH_MARKER_PATH = f"{DATA_DIR}/epoch.txt"
POPULATION_STATS_PATH = f"{DATA_DIR}/population.csv"
ALL_SAMPLES_PATH = f"{DATA_DIR}/samples.npy"
BEST_DDP_MODEL_PATH = f"{CPM_DIR}/BestDDPMode.pth"
DELAY_SUMMARY_PATH = f"{DATA_DIR}/avg_delay.xlsx"  # 辅助指标（保留）
QUEUE_SUMMARY_PATH = f"{DATA_DIR}/avg_queue.xlsx"  # 主要指标：平均排队长度

# 目标收集的 CPMData 行数（控制初始 DQN 需要跑的 episode 数）
CPM_TARGET_ROWS = 300000

# 强化学习模型路径
RL_MODEL_PATH = f"{WEIGHTS_DIR}/best_mode_dqn.pth"
BEST_DELAY_MODEL_PATH = f"{WEIGHTS_DIR}/best_mode_dqn_eva.pth"
BEST_DELAY_LOG_PATH = f"{WEIGHTS_DIR}/avg_delay.txt"

# 数据和样本路径
CPM_DATA_PATH = f"{DATA_DIR}/DiffData.csv"
SAMPLES_DIR = f"{DATA_DIR}/samples"
ITERATION_STATS_PATH = f"{DATA_DIR}/iteration_stats.csv"

# 动态生成的文件路径（函数形式）
def get_sample_path(epoch, round_idx):
    return f"{SAMPLES_DIR}/generated_reward_ep{epoch}_round{round_idx}.npy"

def get_sample_eval_path(epoch, round_idx):
    return f"{SAMPLES_DIR}/sample_eval_ep{epoch}_round{round_idx}.csv"

def get_base_state_path(epoch, round_idx):
    return f"{WEIGHTS_DIR}/base_state_ep{epoch}_round{round_idx}.pth"

def get_final_base_state_path(epoch, round_idx):
    return f"{WEIGHTS_DIR}/final_base_state_ep{epoch}_round{round_idx}.pth"

def get_global_best_network_path(epoch, round_idx):
    return f"{WEIGHTS_DIR}/global_best_network_ep{epoch}_round{round_idx}.pth"

# ==================== 系统参数 ====================
MAX_STATS_RECORDS = 100000
LOG_INTERVAL = 10
SAVE_INTERVAL = 100
DEBUG_MODE = True  # 是否启用调试输出和验证

# 仿真时长（秒），用于 main.py 调用 run_episode 时长
SIMULATION_DURATION_SEC = 3610

# 是否显示epoch结束时的资源清理日志
SHOW_RESOURCE_CLEANUP_LOGS = False

# ==================== 奖励归一化参数 ====================
# DDPM要求数据与噪声N(0,1)在同一尺度，必须归一化训练数据
REWARD_NORMALIZE = True
# 动态计算归一化边界：使用百分位数避免极端值影响
REWARD_PERCENTILE_LOW = 1.0   # 下界百分位数（P1）
REWARD_PERCENTILE_HIGH = 99.0  # 上界百分位数（P99）
# 归一化统计量会在训练时从DiffData.csv动态计算并缓存

# ==================== 评估稳健性参数 ====================
# 评估预热秒数（仅在评估中，前 warmup 秒不计入主指标）
EVAL_WARMUP_SEC = 600
# 评估尾窗长度（仅在评估中，计算末尾 tail 秒的鲁棒指标）
EVAL_TAIL_SEC = 1000
# 评估随机种子（None 表示不指定）
SUMO_SEED = 42
# 鲁棒性阈值：用于判定是否允许刷新最佳网络
ROBUST_P95_FACTOR = 1.3
ROBUST_MAX_FACTOR = 1.6
