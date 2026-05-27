import time
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
from tqdm import tqdm

from lifelong_env import LifelongMAPFEnv, LifelongConfig
from modles import MAPF_ResUNet


Position = Tuple[int, int]


@dataclass
class LifelongRHCRConfig:
    H: int = 32
    W: int = 32
    N_AGENTS: int = 32
    TOTAL_STEPS: int = 500

    WINDOW_SIZE: int = 10
    REPLAN_PERIOD: int = 5

    MIN_WINDOW_SIZE: int = 5
    MID_WINDOW_SIZE: int = 10
    MAX_WINDOW_SIZE: int = 15

    LOW_CONGESTION_TH: float = 0.005
    HIGH_CONGESTION_TH: float = 0.015

    W_GOAL: float = 3.5
    W_WAIT: float = 1.0
    W_CONFLICT: float = 12.0

    MODEL_PATH: str = "./checkpoints_multi/best_model_multi.pth"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    SEED: int = 42


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_neighbors(pos, obs):
    y, x = pos
    H, W = obs.shape

    candidates = [
        (y, x),
        (y - 1, x),
        (y + 1, x),
        (y, x - 1),
        (y, x + 1),
    ]

    return [
        (ny, nx)
        for ny, nx in candidates
        if 0 <= ny < H and 0 <= nx < W and obs[ny, nx] < 0.5
    ]


def get_bfs_distance_map(obs, goal):
    H, W = obs.shape
    gy, gx = goal

    dist = torch.full((H, W), 1e9, dtype=torch.float32)
    dist[gy, gx] = 0.0

    q = [(gy, gx)]
    head = 0

    while head < len(q):
        y, x = q[head]
        head += 1

        for ny, nx in get_neighbors((y, x), obs):
            if dist[ny, nx] > dist[y, x] + 1:
                dist[ny, nx] = dist[y, x] + 1
                q.append((ny, nx))

    return dist


def count_step_collisions(current, next_pos):
    collisions = len(next_pos) - len(set(next_pos))

    n = len(current)
    for i in range(n):
        for j in range(i + 1, n):
            if current[i] == next_pos[j] and current[j] == next_pos[i]:
                collisions += 1

    return collisions


def repair_collisions(current, next_pos):
    repaired = list(next_pos)
    n = len(current)

    changed = True
    max_iter = 10
    it = 0

    while changed and it < max_iter:
        changed = False
        it += 1

        pos_to_agents = {}
        for i, p in enumerate(repaired):
            pos_to_agents.setdefault(p, []).append(i)

        for pos, agents in pos_to_agents.items():
            if len(agents) > 1:
                for agent_id in agents:
                    if repaired[agent_id] != current[agent_id]:
                        repaired[agent_id] = current[agent_id]
                        changed = True

        for i in range(n):
            for j in range(i + 1, n):
                if current[i] == repaired[j] and current[j] == repaired[i]:
                    if repaired[i] != current[i]:
                        repaired[i] = current[i]
                        changed = True
                    if repaired[j] != current[j]:
                        repaired[j] = current[j]
                        changed = True

    return repaired


def plan_one_step_rhcr(obs, current, goals, dist_maps, cfg):
    n = len(current)

    priorities = list(range(n))
    priorities.sort(
        key=lambda i: float(dist_maps[i][current[i][0], current[i][1]]),
        reverse=True,
    )

    next_pos: List[Optional[Position]] = [None for _ in range(n)]
    reserved: Dict[Position, int] = {}

    def choose(agent_id, visiting):
        if agent_id in visiting:
            return False

        visiting.add(agent_id)

        cur = current[agent_id]
        candidates = get_neighbors(cur, obs)

        scored = []

        for cand in candidates:
            cy, cx = cand

            goal_dist = float(dist_maps[agent_id][cy, cx])
            wait_penalty = 1.0 if cand == cur else 0.0
            conflict_penalty = 1.0 if cand in reserved else 0.0

            score = (
                cfg.W_GOAL * goal_dist
                + cfg.W_WAIT * wait_penalty
                + cfg.W_CONFLICT * conflict_penalty
            )

            scored.append((score, random.random(), cand))

        scored.sort(key=lambda x: (x[0], x[1]))

        for _, _, cand in scored:
            if cand in reserved:
                other = reserved[cand]

                if next_pos[other] is None:
                    ok = choose(other, visiting)
                    if not ok:
                        continue

                if cand in reserved:
                    continue

            swap_conflict = False

            for other in range(n):
                if other == agent_id:
                    continue
                if next_pos[other] is None:
                    continue
                if current[other] == cand and next_pos[other] == cur:
                    swap_conflict = True
                    break

            if swap_conflict:
                continue

            next_pos[agent_id] = cand
            reserved[cand] = agent_id
            visiting.remove(agent_id)
            return True

        if cur not in reserved:
            next_pos[agent_id] = cur
            reserved[cur] = agent_id
            visiting.remove(agent_id)
            return True

        visiting.remove(agent_id)
        return False

    for i in priorities:
        if next_pos[i] is None:
            choose(i, set())

    for i in range(n):
        if next_pos[i] is None:
            next_pos[i] = current[i]

    return list(next_pos)


def load_unet_model(cfg):
    device = torch.device(cfg.DEVICE)

    model = MAPF_ResUNet(
        num_actions=5,
        use_aux_head=True,
        dropout_p=0.10,
    ).to(device)

    checkpoint = torch.load(cfg.MODEL_PATH, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded model epoch:", checkpoint.get("epoch", "unknown"))
        print("Loaded model val loss:", checkpoint.get("val_loss", "unknown"))
        print("Loaded model val acc:", checkpoint.get("val_acc", "unknown"))
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def build_unet_features(obs, current, goals, t, cfg):
    H, W = obs.shape

    f_map = obs.clone().float()
    f_cur = torch.zeros((H, W), dtype=torch.float32)
    f_goal = torch.zeros((H, W), dtype=torch.float32)
    f_cg = torch.zeros((H, W), dtype=torch.float32)
    f_grad_x = torch.zeros((H, W), dtype=torch.float32)
    f_grad_y = torch.zeros((H, W), dtype=torch.float32)

    f_capacity = torch.ones((H, W), dtype=torch.float32)
    f_capacity[obs >= 0.5] = 0.0

    f_time = torch.full(
        (H, W),
        float(t % cfg.REPLAN_PERIOD) / float(cfg.REPLAN_PERIOD),
        dtype=torch.float32,
    )

    f_flow = torch.zeros((H, W), dtype=torch.float32)

    for i, (cy, cx) in enumerate(current):
        gy, gx = goals[i]

        f_cur[cy, cx] = 1.0
        f_goal[gy, gx] = 1.0
        f_flow[cy, cx] += 1.0

        dist_map = get_bfs_distance_map(obs, (gy, gx))
        f_cg[cy, cx] = dist_map[cy, cx]

        if cx > 0 and dist_map[cy, cx - 1] < dist_map[cy, cx]:
            f_grad_x[cy, cx] = -1.0
        elif cx < W - 1 and dist_map[cy, cx + 1] < dist_map[cy, cx]:
            f_grad_x[cy, cx] = 1.0

        if cy > 0 and dist_map[cy - 1, cx] < dist_map[cy, cx]:
            f_grad_y[cy, cx] = -1.0
        elif cy < H - 1 and dist_map[cy + 1, cx] < dist_map[cy, cx]:
            f_grad_y[cy, cx] = 1.0

    f_flow = f_flow / max(1, len(current))

    map_x = torch.stack(
        [
            f_map,
            f_cur,
            f_goal,
            f_cg,
            f_grad_x,
            f_grad_y,
            f_capacity,
            f_time,
            f_flow,
        ],
        dim=0,
    )

    map_feat = map_x[[0, 6], :, :].unsqueeze(0)
    agent_feat = map_x[[1, 2, 3, 4, 5], :, :].unsqueeze(0)
    res_feat = map_x[[7, 8], :, :].unsqueeze(0)

    return map_feat, agent_feat, res_feat


@torch.no_grad()
def predict_neural_congestion(model, obs, current, goals, t, cfg):
    device = torch.device(cfg.DEVICE)

    map_feat, agent_feat, res_feat = build_unet_features(
        obs=obs,
        current=current,
        goals=goals,
        t=t,
        cfg=cfg,
    )

    map_feat = map_feat.to(device)
    agent_feat = agent_feat.to(device)
    res_feat = res_feat.to(device)

    _, heatmap_logits = model(
        map_feat,
        agent_feat,
        res_feat,
        return_aux=True,
    )

    heatmap = torch.sigmoid(heatmap_logits)[0, 0].detach().cpu()
    return heatmap


def choose_dynamic_window(heatmap, cfg):
    congestion_score = heatmap.mean().item()

    if congestion_score < cfg.LOW_CONGESTION_TH:
        return cfg.MIN_WINDOW_SIZE
    elif congestion_score < cfg.HIGH_CONGESTION_TH:
        return cfg.MID_WINDOW_SIZE
    else:
        return cfg.MAX_WINDOW_SIZE


def window_to_dynamic_h(current_window, cfg):
    if current_window == cfg.MAX_WINDOW_SIZE:
        return 3
    elif current_window == cfg.MID_WINDOW_SIZE:
        return 5
    else:
        return 8


def run_lifelong_rhcr_method(cfg, use_neural, model=None):
    set_seed(cfg.SEED)

    env_cfg = LifelongConfig(
        H=cfg.H,
        W=cfg.W,
        N_AGENTS=cfg.N_AGENTS,
        SEED=cfg.SEED,
    )

    env = LifelongMAPFEnv(env_cfg)

    total_collisions = 0
    total_replans = 0
    used_windows = []
    used_replan_intervals = []

    start_time = time.time()

    dist_maps = None
    next_replan_t = 0

    method_name = "Neural Dynamic-Replan Lifelong RHCR" if use_neural else "Vanilla Lifelong RHCR"
    pbar = tqdm(range(cfg.TOTAL_STEPS), desc=method_name, leave=False)

    for t in pbar:
        should_replan = False

        if dist_maps is None:
            should_replan = True
        elif use_neural:
            if t >= next_replan_t:
                should_replan = True
        else:
            if t % cfg.REPLAN_PERIOD == 0:
                should_replan = True

        MAX_REPLANS = 100

        if should_replan and total_replans < MAX_REPLANS:

            total_replans += 1

            if use_neural:

                heatmap = predict_neural_congestion(
                    model=model,
                    obs=env.obs,
                    current=env.current_positions,
                    goals=env.goals,
                    t=t,
                    cfg=cfg,
                )

                current_window = choose_dynamic_window(
                    heatmap,
                    cfg
                )

                dynamic_h = window_to_dynamic_h(
                    current_window,
                    cfg
                )

                next_replan_t = t + dynamic_h

            else:

                current_window = cfg.WINDOW_SIZE
                dynamic_h = cfg.REPLAN_PERIOD


            used_windows.append(current_window)
            used_replan_intervals.append(dynamic_h)

            dist_maps = [
                get_bfs_distance_map(env.obs, env.goals[i])
                for i in range(cfg.N_AGENTS)
            ]

        current = env.current_positions

        next_positions = plan_one_step_rhcr(
            obs=env.obs,
            current=current,
            goals=env.goals,
            dist_maps=dist_maps,
            cfg=cfg,
        )

        next_positions = repair_collisions(current, next_positions)

        step_collisions = count_step_collisions(current, next_positions)
        total_collisions += step_collisions

        _, newly_completed = env.step(next_positions)

        throughput = env.completed_tasks / max(1, env.timestep)

        pbar.set_postfix(
            {
                "tasks": env.completed_tasks,
                "new": newly_completed,
                "coll": total_collisions,
                "thr": f"{throughput:.3f}",
            }
        )

    runtime = time.time() - start_time

    return {
        "completed_tasks": env.completed_tasks,
        "throughput": env.completed_tasks / cfg.TOTAL_STEPS,
        "collisions": total_collisions,
        "runtime": runtime,
        "runtime_per_step": runtime / cfg.TOTAL_STEPS,
        "replans": total_replans,
        "avg_window": sum(used_windows) / max(1, len(used_windows)),
        "avg_replan_interval": sum(used_replan_intervals) / max(1, len(used_replan_intervals)),
    }


def summarize_results(name, results):
    keys = results[0].keys()
    summary = {}

    for k in keys:
        vals = np.array([r[k] for r in results], dtype=np.float64)
        summary[k + "_mean"] = vals.mean()
        summary[k + "_std"] = vals.std()

    print(f"\n==============================")
    print(name)
    print("==============================")

    for k, v in summary.items():
        print(f"{k}: {v:.6f}")

    return summary


def run_multi_seed():
    SEEDS = [1, 2, 3, 4, 5]

    base_cfg = LifelongRHCRConfig(
        H=32,
        W=32,
        N_AGENTS=32,
        TOTAL_STEPS=500,
        WINDOW_SIZE=10,
        REPLAN_PERIOD=5,
        MIN_WINDOW_SIZE=5,
        MID_WINDOW_SIZE=10,
        MAX_WINDOW_SIZE=15,
        LOW_CONGESTION_TH=0.005,
        HIGH_CONGESTION_TH=0.015,
        W_GOAL=3.5,
        W_WAIT=1.0,
        W_CONFLICT=12.0,
        SEED=42,
    )

    print("=== Lifelong RHCR Multi-Seed Experiment ===")
    print("Seeds:", SEEDS)
    print(f"Agents: {base_cfg.N_AGENTS}")
    print(f"Total steps: {base_cfg.TOTAL_STEPS}")
    print(f"Vanilla h: {base_cfg.REPLAN_PERIOD}")
    print(f"Dynamic windows: {base_cfg.MIN_WINDOW_SIZE}, {base_cfg.MID_WINDOW_SIZE}, {base_cfg.MAX_WINDOW_SIZE}")
    print(f"Thresholds: {base_cfg.LOW_CONGESTION_TH}, {base_cfg.HIGH_CONGESTION_TH}")
    print(f"Device: {base_cfg.DEVICE}")

    model = load_unet_model(base_cfg)

    all_vanilla = []
    all_neural = []

    for seed in SEEDS:
        print("\n==============================")
        print(f"Running seed {seed}")
        print("==============================")

        cfg = LifelongRHCRConfig(
            H=base_cfg.H,
            W=base_cfg.W,
            N_AGENTS=base_cfg.N_AGENTS,
            TOTAL_STEPS=base_cfg.TOTAL_STEPS,
            WINDOW_SIZE=base_cfg.WINDOW_SIZE,
            REPLAN_PERIOD=base_cfg.REPLAN_PERIOD,
            MIN_WINDOW_SIZE=base_cfg.MIN_WINDOW_SIZE,
            MID_WINDOW_SIZE=base_cfg.MID_WINDOW_SIZE,
            MAX_WINDOW_SIZE=base_cfg.MAX_WINDOW_SIZE,
            LOW_CONGESTION_TH=base_cfg.LOW_CONGESTION_TH,
            HIGH_CONGESTION_TH=base_cfg.HIGH_CONGESTION_TH,
            W_GOAL=base_cfg.W_GOAL,
            W_WAIT=base_cfg.W_WAIT,
            W_CONFLICT=base_cfg.W_CONFLICT,
            MODEL_PATH=base_cfg.MODEL_PATH,
            DEVICE=base_cfg.DEVICE,
            SEED=seed,
        )

        vanilla = run_lifelong_rhcr_method(
            cfg=cfg,
            use_neural=False,
            model=None,
        )

        neural = run_lifelong_rhcr_method(
            cfg=cfg,
            use_neural=True,
            model=model,
        )

        all_vanilla.append(vanilla)
        all_neural.append(neural)

        print(f"\nSeed {seed} results:")
        print(f"Vanilla tasks={vanilla['completed_tasks']}, throughput={vanilla['throughput']:.6f}, collisions={vanilla['collisions']}, runtime={vanilla['runtime']:.2f}")
        print(f"Neural  tasks={neural['completed_tasks']}, throughput={neural['throughput']:.6f}, collisions={neural['collisions']}, runtime={neural['runtime']:.2f}")
        print(f"Avg h: vanilla={vanilla['avg_replan_interval']:.2f}, neural={neural['avg_replan_interval']:.2f}")
        print(f"Avg w: vanilla={vanilla['avg_window']:.2f}, neural={neural['avg_window']:.2f}")

    vanilla_summary = summarize_results("Vanilla Lifelong RHCR Summary", all_vanilla)
    neural_summary = summarize_results("Neural Dynamic-Replan Lifelong RHCR Summary", all_neural)

    print("\n==============================")
    print("Final Comparison")
    print("==============================")

    print(
        f"Completed tasks: "
        f"vanilla={vanilla_summary['completed_tasks_mean']:.2f} ± {vanilla_summary['completed_tasks_std']:.2f} | "
        f"neural={neural_summary['completed_tasks_mean']:.2f} ± {neural_summary['completed_tasks_std']:.2f}"
    )

    print(
        f"Throughput: "
        f"vanilla={vanilla_summary['throughput_mean']:.6f} ± {vanilla_summary['throughput_std']:.6f} | "
        f"neural={neural_summary['throughput_mean']:.6f} ± {neural_summary['throughput_std']:.6f}"
    )

    print(
        f"Collisions: "
        f"vanilla={vanilla_summary['collisions_mean']:.2f} ± {vanilla_summary['collisions_std']:.2f} | "
        f"neural={neural_summary['collisions_mean']:.2f} ± {neural_summary['collisions_std']:.2f}"
    )

    print(
        f"Runtime: "
        f"vanilla={vanilla_summary['runtime_mean']:.2f} ± {vanilla_summary['runtime_std']:.2f} | "
        f"neural={neural_summary['runtime_mean']:.2f} ± {neural_summary['runtime_std']:.2f}"
    )

    print(
        f"Avg replan interval: "
        f"vanilla={vanilla_summary['avg_replan_interval_mean']:.2f} | "
        f"neural={neural_summary['avg_replan_interval_mean']:.2f}"
    )

    print(
        f"Avg window: "
        f"vanilla={vanilla_summary['avg_window_mean']:.2f} | "
        f"neural={neural_summary['avg_window_mean']:.2f}"
    )

    if neural_summary["throughput_mean"] > vanilla_summary["throughput_mean"]:
        improvement = (
            neural_summary["throughput_mean"]
            - vanilla_summary["throughput_mean"]
        ) / max(1e-8, vanilla_summary["throughput_mean"]) * 100

        print(f"\n✅ Neural improves throughput by {improvement:.2f}% on average.")
    else:
        print("\n⚠️ Neural does not improve average throughput.")

    print("==============================")


if __name__ == "__main__":
    run_multi_seed()