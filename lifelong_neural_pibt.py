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
class LifelongPIBTConfig:
    H: int = 32
    W: int = 32
    N_AGENTS: int = 40
    TOTAL_STEPS: int = 500

    # PIBT / scoring weights
    W_GOAL: float = 3.5
    W_WAIT: float = 1.0
    W_CONFLICT: float = 12.0

    # Neural heatmap guidance
    W_CONGESTION: float = 0.5

    # Priority-only main method:
    # False = heatmap only changes priority ordering, not candidate movement score
    USE_HEATMAP_REWARD: bool = False

    # Neural update frequency
    NEURAL_UPDATE_PERIOD: int = 5

    # Stuck metric: consecutive no-progress steps
    STUCK_THRESHOLD: int = 3

    # U-Net model
    MODEL_PATH: str = "./checkpoints_multi/best_model_multi.pth"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # For feature construction
    REPLAN_PERIOD: int = 5

    SEED: int = 42


# =====================================================
# 1. Basic utilities
# =====================================================
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_neighbors(pos: Position, obs: torch.Tensor):
    y, x = pos
    H, W = obs.shape

    candidates = [
        (y, x),        # wait
        (y - 1, x),    # up
        (y + 1, x),    # down
        (y, x - 1),    # left
        (y, x + 1),    # right
    ]

    valid = []
    for ny, nx in candidates:
        if 0 <= ny < H and 0 <= nx < W and obs[ny, nx] < 0.5:
            valid.append((ny, nx))

    return valid


def get_bfs_distance_map(obs: torch.Tensor, goal: Position):
    H, W = obs.shape
    gy, gx = goal

    dist = torch.full((H, W), 1e9, dtype=torch.float32)

    if obs[gy, gx] >= 0.5:
        return dist

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


# =====================================================
# 2. Collision check, repair, and waiting/stuck metrics
# =====================================================
def count_step_collisions(current: List[Position], next_pos: List[Position]):
    collisions = 0

    # vertex collision
    collisions += len(next_pos) - len(set(next_pos))

    # edge-swap collision
    n = len(current)
    for i in range(n):
        for j in range(i + 1, n):
            if current[i] == next_pos[j] and current[j] == next_pos[i]:
                collisions += 1

    return collisions


def repair_collisions(current: List[Position], next_pos: List[Position]):
    """
    Strict safety repair:
    If a vertex or edge-swap collision remains, conflicted agents wait.
    """
    repaired = list(next_pos)
    n = len(current)

    changed = True
    max_iter = 10
    it = 0

    while changed and it < max_iter:
        changed = False
        it += 1

        # vertex collision
        pos_to_agents = {}
        for i, p in enumerate(repaired):
            pos_to_agents.setdefault(p, []).append(i)

        for _, agents in pos_to_agents.items():
            if len(agents) > 1:
                for agent_id in agents:
                    if repaired[agent_id] != current[agent_id]:
                        repaired[agent_id] = current[agent_id]
                        changed = True

        # edge-swap collision
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


def compute_waiting_and_stuck_metrics(
    current: List[Position],
    next_pos: List[Position],
    dist_maps: List[torch.Tensor],
    no_progress_streak: List[int],
    cfg: LifelongPIBTConfig,
):
    """
    waiting_steps:
        agent stays in the same cell.

    no_progress_steps:
        agent does not get closer to its current goal.

    stuck_steps:
        agent has no progress for at least STUCK_THRESHOLD consecutive steps.
    """
    wait_steps = 0
    no_progress_steps = 0
    stuck_steps = 0

    n = len(current)

    for i in range(n):
        cy, cx = current[i]
        ny, nx = next_pos[i]

        old_dist = float(dist_maps[i][cy, cx])
        new_dist = float(dist_maps[i][ny, nx])

        # agent did not move
        if current[i] == next_pos[i]:
            wait_steps += 1

        # agent did not get closer to goal
        if new_dist >= old_dist:
            no_progress_steps += 1
            no_progress_streak[i] += 1
        else:
            no_progress_streak[i] = 0

        # consecutive no-progress means stuck
        if no_progress_streak[i] >= cfg.STUCK_THRESHOLD:
            stuck_steps += 1

    return wait_steps, no_progress_steps, stuck_steps, no_progress_streak


# =====================================================
# 3. Load U-Net model
# =====================================================
def load_unet_model(cfg: LifelongPIBTConfig):
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


# =====================================================
# 4. Build U-Net features from lifelong state
# =====================================================
def build_unet_features(
    obs: torch.Tensor,
    current: List[Position],
    goals: List[Position],
    t: int,
    cfg: LifelongPIBTConfig,
):
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
def predict_neural_congestion(
    model,
    obs: torch.Tensor,
    current: List[Position],
    goals: List[Position],
    t: int,
    cfg: LifelongPIBTConfig,
):
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


def build_zero_congestion(obs: torch.Tensor):
    H, W = obs.shape
    return torch.zeros((H, W), dtype=torch.float32)


# =====================================================
# 5. PIBT one-step planner
# =====================================================
def plan_one_step_pibt(
    obs: torch.Tensor,
    current: List[Position],
    goals: List[Position],
    dist_maps: List[torch.Tensor],
    congestion_map: torch.Tensor,
    use_neural: bool,
    cfg: LifelongPIBTConfig,
):
    """
    PIBT-like priority inheritance and backtracking.

    Vanilla:
        priority = distance-to-goal

    Neural-Priority:
        priority = predicted pressure at current agent position,
                   then distance-to-goal

    Candidate heatmap reward is optional.
    Main method currently uses USE_HEATMAP_REWARD=False.
    """
    n = len(current)

    priorities = list(range(n))

    if use_neural:
        priorities.sort(
            key=lambda i: (
                float(congestion_map[current[i][0], current[i][1]]),
                float(dist_maps[i][current[i][0], current[i][1]]),
            ),
            reverse=True,
        )
    else:
        priorities.sort(
            key=lambda i: float(dist_maps[i][current[i][0], current[i][1]]),
            reverse=True,
        )

    next_pos: List[Optional[Position]] = [None for _ in range(n)]
    reserved: Dict[Position, int] = {}

    def choose(agent_id: int, visiting: set):
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
            heatmap_reward = float(congestion_map[cy, cx])

            score = (
                cfg.W_GOAL * goal_dist
                + cfg.W_WAIT * wait_penalty
                + cfg.W_CONFLICT * conflict_penalty
            )

            if use_neural and cfg.USE_HEATMAP_REWARD:
                score -= cfg.W_CONGESTION * heatmap_reward

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

            # avoid edge swap
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

        # fallback: wait
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


# =====================================================
# 6. Run one lifelong PIBT method
# =====================================================
def run_lifelong_pibt_method(
    cfg: LifelongPIBTConfig,
    use_neural: bool,
    model=None,
):
    set_seed(cfg.SEED)

    env_cfg = LifelongConfig(
        H=cfg.H,
        W=cfg.W,
        N_AGENTS=cfg.N_AGENTS,
        SEED=cfg.SEED,
    )

    env = LifelongMAPFEnv(env_cfg)

    total_collisions = 0
    neural_calls = 0

    # New metrics
    total_wait_steps = 0
    total_no_progress_steps = 0
    total_stuck_steps = 0
    no_progress_streak = [0 for _ in range(cfg.N_AGENTS)]

    start_time = time.time()

    congestion_map = build_zero_congestion(env.obs)

    method_name = "Neural Priority Lifelong PIBT" if use_neural else "Vanilla Lifelong PIBT"
    pbar = tqdm(range(cfg.TOTAL_STEPS), desc=method_name, leave=False)

    for t in pbar:
        # Update distance maps every step because lifelong goals may change.
        dist_maps = [
            get_bfs_distance_map(env.obs, env.goals[i])
            for i in range(cfg.N_AGENTS)
        ]

        if use_neural and (t % cfg.NEURAL_UPDATE_PERIOD == 0):
            congestion_map = predict_neural_congestion(
                model=model,
                obs=env.obs,
                current=env.current_positions,
                goals=env.goals,
                t=t,
                cfg=cfg,
            )
            neural_calls += 1

        if not use_neural:
            congestion_map = build_zero_congestion(env.obs)

        current = env.current_positions

        next_positions = plan_one_step_pibt(
            obs=env.obs,
            current=current,
            goals=env.goals,
            dist_maps=dist_maps,
            congestion_map=congestion_map,
            use_neural=use_neural,
            cfg=cfg,
        )

        next_positions = repair_collisions(current, next_positions)

        # Collision metric
        step_collisions = count_step_collisions(current, next_positions)
        total_collisions += step_collisions

        # Waiting / stuck metrics
        wait_steps, no_progress_steps, stuck_steps, no_progress_streak = (
            compute_waiting_and_stuck_metrics(
                current=current,
                next_pos=next_positions,
                dist_maps=dist_maps,
                no_progress_streak=no_progress_streak,
                cfg=cfg,
            )
        )

        total_wait_steps += wait_steps
        total_no_progress_steps += no_progress_steps
        total_stuck_steps += stuck_steps

        _, newly_completed = env.step(next_positions)

        throughput = env.completed_tasks / max(1, env.timestep)

        pbar.set_postfix(
            {
                "tasks": env.completed_tasks,
                "new": newly_completed,
                "coll": total_collisions,
                "wait": total_wait_steps,
                "stuck": total_stuck_steps,
                "thr": f"{throughput:.3f}",
            }
        )

    runtime = time.time() - start_time

    total_agent_steps = cfg.TOTAL_STEPS * cfg.N_AGENTS

    return {
        "completed_tasks": env.completed_tasks,
        "throughput": env.completed_tasks / cfg.TOTAL_STEPS,
        "collisions": total_collisions,
        "runtime": runtime,
        "runtime_per_step": runtime / cfg.TOTAL_STEPS,
        "neural_calls": neural_calls,

        # Waiting metrics
        "total_wait_steps": total_wait_steps,
        "wait_ratio": total_wait_steps / max(1, total_agent_steps),
        "avg_wait_steps_per_agent": total_wait_steps / max(1, cfg.N_AGENTS),

        # No-progress metrics
        "total_no_progress_steps": total_no_progress_steps,
        "no_progress_ratio": total_no_progress_steps / max(1, total_agent_steps),

        # Stuck metrics
        "total_stuck_steps": total_stuck_steps,
        "stuck_ratio": total_stuck_steps / max(1, total_agent_steps),
        "avg_stuck_steps_per_agent": total_stuck_steps / max(1, cfg.N_AGENTS),
    }


# =====================================================
# 7. Statistics
# =====================================================
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


# =====================================================
# 8. Main multi-seed experiment
# =====================================================
def run_multi_seed():
    SEEDS = [1, 2, 3, 4, 5]

    base_cfg = LifelongPIBTConfig(
        H=32,
        W=32,
        N_AGENTS=40,
        TOTAL_STEPS=500,
        W_GOAL=3.5,
        W_WAIT=1.0,
        W_CONFLICT=12.0,
        W_CONGESTION=0.5,
        USE_HEATMAP_REWARD=False,
        NEURAL_UPDATE_PERIOD=5,
        STUCK_THRESHOLD=3,
        SEED=42,
    )

    print("=== Lifelong Neural-Priority PIBT Multi-Seed Experiment ===")
    print("Seeds:", SEEDS)
    print(f"Agents: {base_cfg.N_AGENTS}")
    print(f"Total steps: {base_cfg.TOTAL_STEPS}")
    print(f"W_CONGESTION: {base_cfg.W_CONGESTION}")
    print(f"Use heatmap reward: {base_cfg.USE_HEATMAP_REWARD}")
    print(f"Neural update period: {base_cfg.NEURAL_UPDATE_PERIOD}")
    print(f"Stuck threshold: {base_cfg.STUCK_THRESHOLD}")
    print(f"Device: {base_cfg.DEVICE}")

    model = load_unet_model(base_cfg)

    all_vanilla = []
    all_neural = []

    for seed in SEEDS:
        print("\n==============================")
        print(f"Running seed {seed}")
        print("==============================")

        cfg = LifelongPIBTConfig(
            H=base_cfg.H,
            W=base_cfg.W,
            N_AGENTS=base_cfg.N_AGENTS,
            TOTAL_STEPS=base_cfg.TOTAL_STEPS,
            W_GOAL=base_cfg.W_GOAL,
            W_WAIT=base_cfg.W_WAIT,
            W_CONFLICT=base_cfg.W_CONFLICT,
            W_CONGESTION=base_cfg.W_CONGESTION,
            USE_HEATMAP_REWARD=base_cfg.USE_HEATMAP_REWARD,
            NEURAL_UPDATE_PERIOD=base_cfg.NEURAL_UPDATE_PERIOD,
            STUCK_THRESHOLD=base_cfg.STUCK_THRESHOLD,
            MODEL_PATH=base_cfg.MODEL_PATH,
            DEVICE=base_cfg.DEVICE,
            SEED=seed,
        )

        vanilla = run_lifelong_pibt_method(
            cfg=cfg,
            use_neural=False,
            model=None,
        )

        neural = run_lifelong_pibt_method(
            cfg=cfg,
            use_neural=True,
            model=model,
        )

        all_vanilla.append(vanilla)
        all_neural.append(neural)

        print(f"\nSeed {seed} results:")
        print(
            f"Vanilla tasks={vanilla['completed_tasks']}, "
            f"throughput={vanilla['throughput']:.6f}, "
            f"collisions={vanilla['collisions']}, "
            f"wait_ratio={vanilla['wait_ratio']:.6f}, "
            f"stuck_ratio={vanilla['stuck_ratio']:.6f}, "
            f"runtime={vanilla['runtime']:.2f}"
        )
        print(
            f"Neural  tasks={neural['completed_tasks']}, "
            f"throughput={neural['throughput']:.6f}, "
            f"collisions={neural['collisions']}, "
            f"wait_ratio={neural['wait_ratio']:.6f}, "
            f"stuck_ratio={neural['stuck_ratio']:.6f}, "
            f"runtime={neural['runtime']:.2f}, "
            f"neural_calls={neural['neural_calls']}"
        )

    vanilla_summary = summarize_results(
        "Vanilla Lifelong PIBT Summary",
        all_vanilla,
    )

    neural_summary = summarize_results(
        "Neural-Priority Lifelong PIBT Summary",
        all_neural,
    )

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
        f"Wait ratio: "
        f"vanilla={vanilla_summary['wait_ratio_mean']:.6f} ± {vanilla_summary['wait_ratio_std']:.6f} | "
        f"neural={neural_summary['wait_ratio_mean']:.6f} ± {neural_summary['wait_ratio_std']:.6f}"
    )

    print(
        f"No-progress ratio: "
        f"vanilla={vanilla_summary['no_progress_ratio_mean']:.6f} ± {vanilla_summary['no_progress_ratio_std']:.6f} | "
        f"neural={neural_summary['no_progress_ratio_mean']:.6f} ± {neural_summary['no_progress_ratio_std']:.6f}"
    )

    print(
        f"Stuck ratio: "
        f"vanilla={vanilla_summary['stuck_ratio_mean']:.6f} ± {vanilla_summary['stuck_ratio_std']:.6f} | "
        f"neural={neural_summary['stuck_ratio_mean']:.6f} ± {neural_summary['stuck_ratio_std']:.6f}"
    )

    print(
        f"Runtime: "
        f"vanilla={vanilla_summary['runtime_mean']:.2f} ± {vanilla_summary['runtime_std']:.2f} | "
        f"neural={neural_summary['runtime_mean']:.2f} ± {neural_summary['runtime_std']:.2f}"
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