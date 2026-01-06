#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch

from traffic_simulator import SUMOTrafficLights
from neural_agent import QLearner
from replay import ReplayBuffer
import config


def find_model_path(default_path: str) -> str:
    # Accept both 'weight' and 'weights' directories; default to provided path
    if os.path.isfile(default_path):
        return default_path
    # Try alternate folder naming
    alt = default_path.replace("weight/", "weights/") if "weight/" in default_path else default_path.replace("weights/", "weight/")
    if os.path.isfile(alt):
        return alt
    # Fall back to config.BEST_DELAY_MODEL_PATH
    if os.path.isfile(getattr(config, "BEST_DELAY_MODEL_PATH", "")):
        return getattr(config, "BEST_DELAY_MODEL_PATH")
    return default_path


def build_eval_agents(env: SUMOTrafficLights, model_path: str, device: torch.device):
    learners = {}
    replays = {}
    # DQN input/output dims in this project are 5 and 2
    input_dim, output_dim = 5, 2
    # 读取合并权重：{tlID(str): state_dict}
    combined = torch.load(model_path, map_location=device)
    if not isinstance(combined, dict):
        raise RuntimeError(f"Model file is not a combined dict: {model_path}")
    for tlID in env.get_trafficlights_ID_list():
        key = str(tlID)
        if key not in combined:
            raise RuntimeError(f"Missing key '{key}' in model file: {model_path}")
        replays[tlID] = ReplayBuffer(capacity=10000)
        learner = QLearner(
            tlID, env,
            starting_state=0, goal_state=0,
            alpha=0.01, gamma=0.01,
            model='eval',  # 禁止探索/学习，纯贪心
            replay=replays[tlID],
            target_update=15,
            eps_start=0.0, eps_end=0.0, eps_decay=1,
            input_dim=input_dim, output_dim=output_dim,
            batch_size=32,
            network_file=None,  # 手动加载
            device_override=device,
        )
        learner.policy_net.load_state_dict(combined[key], strict=False)
        learner.target_net.load_state_dict(learner.policy_net.state_dict(), strict=False)
        # 明确设置为评估模式
        learner.policy_net.eval()
        learner.target_net.eval()
        learners[tlID] = learner
    return learners, replays


def main():
    parser = argparse.ArgumentParser(description="Evaluate DQN policy in SUMO GUI")
    parser.add_argument("--model", default="weight/best_mode_dqn_eva.pth", help="Path to model .pth file")
    parser.add_argument("--duration", type=int, default=getattr(config, "SIMULATION_DURATION_SEC", 3010), help="Simulation duration in seconds")
    parser.add_argument("--seed", type=int, default=42, help="SUMO random seed (overrides config.SUMO_SEED)")
    parser.add_argument("--sumo-config", default=config.SUMO_CONFIG_PATH, help="Path to .sumocfg")
    args = parser.parse_args()

    model_path = find_model_path(args.model)
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(1)

    # Ensure output directories exist
    for d in [getattr(config, "DATA_DIR", "data")]:
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass

    # Set SUMO seed for reproducible evaluation
    try:
        config.SUMO_SEED = int(args.seed)
    except Exception:
        config.SUMO_SEED = 42

    # Force CPU for deterministic eval unless config allows GPU explicitly
    device = torch.device("cpu")

    # Build environment with SUMO GUI
    try:
        env = SUMOTrafficLights(args.sumo_config, port=8813, use_gui=True, batch_size=32)
    except Exception as e:
        print(f"[ERROR] Failed to initialize SUMO environment: {e}")
        sys.exit(1)

    # Build agents and attach to env
    learners, replays = build_eval_agents(env, model_path, device)
    env.learners = learners
    env.replay_buffers = replays

    print(f"[INFO] Starting GUI simulation for {args.duration}s using '{model_path}'")
    try:
        metrics = env.run_episode(max_steps=args.duration, exp=None, epoch_idx=0, mode='eval', sample_id=None, save_outputs=True)
    except Exception as e:
        print(f"[ERROR] Evaluation run failed: {e}")
        try:
            env.close()
        except Exception:
            pass
        sys.exit(2)

    # Print summary metrics
    if metrics is not None:
        inner = metrics.get("average_queue_inner_per_road")
        total = metrics.get("average_queue_inner_sum")
        print("[RESULT] average_queue_inner_per_road:", inner)
        print("[RESULT] average_queue_inner_sum:", total)
        if metrics.get("avg_tail_robust") is not None:
            print("[RESULT] avg_tail_robust:", metrics.get("avg_tail_robust"))
            print("[RESULT] p95_tail_robust:", metrics.get("p95_tail_robust"))
            print("[RESULT] max_tail_robust:", metrics.get("max_tail_robust"))
            print("[RESULT] tail_steps:", metrics.get("tail_steps"))

        # Outer ring metrics
        outer = metrics.get("average_queue_outer_per_road")
        total_outer = metrics.get("average_queue_outer_sum")
        if outer is not None:
            print("[RESULT] average_queue_outer_per_road:", outer)
        if total_outer is not None:
            print("[RESULT] average_queue_outer_sum:", total_outer)
        if metrics.get("avg_tail_robust_outer") is not None:
            print("[RESULT] avg_tail_robust_outer:", metrics.get("avg_tail_robust_outer"))
            print("[RESULT] p95_tail_robust_outer:", metrics.get("p95_tail_robust_outer"))
            print("[RESULT] max_tail_robust_outer:", metrics.get("max_tail_robust_outer"))
            print("[RESULT] tail_steps_outer:", metrics.get("tail_steps_outer"))

    # Ensure proper shutdown
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
