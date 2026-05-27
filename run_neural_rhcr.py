import os
import re
import time
import random
import subprocess
from dataclasses import dataclass
from collections import deque
from typing import List, Tuple, Dict, Optional

import torch
from tqdm import tqdm

from modles import MAPF_ResUNet


Position = Tuple[int, int]


# ============================================================
# 1. Config
# ============================================================
@dataclass
class Config:
    H: int = 32
    W: int = 32
    N_AGENTS: int = 16

    NUM_CASES: int = 50

    # RHCR parameters
    WINDOW_SIZE: int = 10       # default w
    REPLAN_PERIOD: int = 5      # h
    TOTAL_STEPS: int = 80

    # Dynamic neural horizon
    MIN_WINDOW_SIZE: int = 5
    MID_WINDOW_SIZE: int = 10
    MAX_WINDOW_SIZE: int = 15

    LOW_CONGESTION_TH: float = 0.005
    HIGH_CONGESTION_TH: float = 0.015

    # LaCAM path
    WSL_LACAM_EXE: str = "/home/wlf/lacam/build/main"
    TEMP_DIR: str = "./rhcr_temp"

    LACAM_TIME_LIMIT_SEC: int = 20
    SUBPROCESS_TIMEOUT_SEC: int = 60

    # U-Net model
    MODEL_PATH: str = "./checkpoints_multi/best_model_multi.pth"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # scoring weights
    W_GOAL: float = 3.5
    W_WAIT: float = 1.5
    W_CONFLICT: float = 12.0

    SEED: int = 42


# ============================================================
# 2. Utilities
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def windows_to_wsl_path(win_path: str) -> str:
    win_path = os.path.abspath(win_path)
    drive = win_path[0].lower()
    rest = win_path[2:].replace("\\", "/")
    return f"/mnt/{drive}{rest}"


def get_neighbors(pos: Position, obs: torch.Tensor) -> List[Position]:
    y, x = pos
    H, W = obs.shape

    candidates = [
        (y, x),
        (y - 1, x),
        (y + 1, x),
        (y, x - 1),
        (y, x + 1),
    ]

    valid = []
    for ny, nx in candidates:
        if 0 <= ny < H and 0 <= nx < W and obs[ny, nx] < 0.5:
            valid.append((ny, nx))

    return valid


# ============================================================
# 3. Map generation
# ============================================================
def generate_random_map_and_agents(cfg: Config):
    obs = torch.zeros((cfg.H, cfg.W), dtype=torch.float32)

    for y in range(4, cfg.H - 4, 3):
        for x in range(4, cfg.W - 4):
            if x % 4 != 0:
                obs[y, x] = 1.0

    for y in range(cfg.H):
        for x in range(cfg.W):
            if obs[y, x] == 1.0 and random.random() < 0.25:
                obs[y, x] = 0.0

    obs[0, :] = 1.0
    obs[-1, :] = 1.0
    obs[:, 0] = 1.0
    obs[:, -1] = 1.0

    free_cells = [
        (y, x)
        for y in range(cfg.H)
        for x in range(cfg.W)
        if obs[y, x] < 0.5
    ]

    if len(free_cells) < cfg.N_AGENTS * 3:
        return None, None

    random.shuffle(free_cells)

    starts = free_cells[:cfg.N_AGENTS]
    goals = free_cells[cfg.N_AGENTS:2 * cfg.N_AGENTS]

    agents = []
    for i in range(cfg.N_AGENTS):
        agents.append({
            "sy": starts[i][0],
            "sx": starts[i][1],
            "gy": goals[i][0],
            "gx": goals[i][1],
        })

    return obs, agents


def bfs_reachable(obs, sy, sx, gy, gx):
    if obs[sy, sx] >= 0.5 or obs[gy, gx] >= 0.5:
        return False

    q = deque([(sy, sx)])
    visited = {(sy, sx)}

    while q:
        y, x = q.popleft()

        if (y, x) == (gy, gx):
            return True

        for ny, nx in get_neighbors((y, x), obs):
            if (ny, nx) not in visited:
                visited.add((ny, nx))
                q.append((ny, nx))

    return False


def all_agents_reachable(obs, agents):
    for ag in agents:
        if not bfs_reachable(obs, ag["sy"], ag["sx"], ag["gy"], ag["gx"]):
            return False
    return True


# ============================================================
# 4. BFS distance maps
# ============================================================
def get_bfs_distance_map(obs: torch.Tensor, goal: Position):
    H, W = obs.shape
    gy, gx = goal

    dist = torch.full((H, W), 1e9, dtype=torch.float32)

    if obs[gy, gx] >= 0.5:
        return dist

    dist[gy, gx] = 0.0
    q = deque([(gy, gx)])

    while q:
        y, x = q.popleft()

        for ny, nx in get_neighbors((y, x), obs):
            if dist[ny, nx] > dist[y, x] + 1:
                dist[ny, nx] = dist[y, x] + 1
                q.append((ny, nx))

    return dist


# ============================================================
# 5. LaCAM for reference
# ============================================================
def export_to_movingai(obs, agents, map_path, scen_path, map_name="temp.map"):
    H, W = obs.shape

    with open(map_path, "w", encoding="ascii", newline="\n") as f:
        f.write("type octile\n")
        f.write(f"height {H}\n")
        f.write(f"width {W}\n")
        f.write("map\n")

        for y in range(H):
            row = "".join("@" if obs[y, x] >= 0.5 else "." for x in range(W))
            f.write(row + "\n")

    with open(scen_path, "w", encoding="ascii", newline="\n") as f:
        f.write("version 1\n")

        for ag in agents:
            line = (
                f"0\t{map_name}\t{W}\t{H}\t"
                f"{ag['sx']}\t{ag['sy']}\t{ag['gx']}\t{ag['gy']}\t0\n"
            )
            f.write(line)


def parse_lacam_output(out_file, num_agents):
    if not os.path.exists(out_file):
        return None

    text = open(out_file, "r", encoding="utf-8", errors="ignore").read().strip()
    if not text:
        return None

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("solution"):
            start_idx = i + 1
            break

    if start_idx is None:
        return None

    parsed_lines = []

    for line in lines[start_idx:]:
        m = re.match(r"^(\d+)\s*:\s*(.*)$", line)
        if not m:
            continue

        idx = int(m.group(1))
        coords_str = m.group(2)
        matches = re.findall(r"\((\d+)\s*,\s*(\d+)\)", coords_str)

        coords = [(int(y), int(x)) for x, y in matches]
        parsed_lines.append((idx, coords))

    if len(parsed_lines) == 0:
        return None

    if len(parsed_lines) == num_agents:
        paths = [[] for _ in range(num_agents)]
        for idx, coords in parsed_lines:
            if 0 <= idx < num_agents:
                paths[idx] = coords
        if all(len(p) > 0 for p in paths):
            return paths

    if all(len(coords) == num_agents for _, coords in parsed_lines):
        parsed_lines.sort(key=lambda x: x[0])
        time_steps = [coords for _, coords in parsed_lines]

        paths = [[] for _ in range(num_agents)]
        for coords_at_t in time_steps:
            for agent_id in range(num_agents):
                paths[agent_id].append(coords_at_t[agent_id])

        if all(len(p) > 0 for p in paths):
            return paths

    return None


def run_lacam_expert_wsl(obs, agents, cfg: Config):
    os.makedirs(cfg.TEMP_DIR, exist_ok=True)

    win_map_file = os.path.abspath(os.path.join(cfg.TEMP_DIR, "temp.map"))
    win_scen_file = os.path.abspath(os.path.join(cfg.TEMP_DIR, "temp.scen"))
    win_out_file = os.path.abspath(os.path.join(cfg.TEMP_DIR, "out.txt"))

    if os.path.exists(win_out_file):
        os.remove(win_out_file)

    export_to_movingai(obs, agents, win_map_file, win_scen_file)

    cmd = [
        "wsl",
        cfg.WSL_LACAM_EXE,
        "-m", windows_to_wsl_path(win_map_file),
        "-i", windows_to_wsl_path(win_scen_file),
        "-N", str(len(agents)),
        "-t", str(cfg.LACAM_TIME_LIMIT_SEC),
        "-o", windows_to_wsl_path(win_out_file),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=cfg.SUBPROCESS_TIMEOUT_SEC,
        )

        if result.returncode != 0:
            return None

        return parse_lacam_output(win_out_file, len(agents))

    except Exception:
        return None


# ============================================================
# 6. Load U-Net
# ============================================================
def load_unet_model(cfg: Config):
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


# ============================================================
# 7. Build U-Net features
# ============================================================
def build_unet_features(
    obs: torch.Tensor,
    current: List[Position],
    goals: List[Position],
    t: int,
    cfg: Config,
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

    if len(current) > 0:
        f_flow = f_flow / float(len(current))

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
    cfg: Config,
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


def choose_dynamic_window(heatmap: torch.Tensor, cfg: Config) -> int:
    congestion_score = heatmap.mean().item()

    if congestion_score < cfg.LOW_CONGESTION_TH:
        return cfg.MIN_WINDOW_SIZE
    elif congestion_score < cfg.HIGH_CONGESTION_TH:
        return cfg.MID_WINDOW_SIZE
    else:
        return cfg.MAX_WINDOW_SIZE


def build_zero_congestion(obs: torch.Tensor):
    H, W = obs.shape
    return torch.zeros((H, W), dtype=torch.float32)


# ============================================================
# 8. One-step planner
# ============================================================
def plan_one_step_with_scores(
    obs: torch.Tensor,
    current: List[Position],
    goals: List[Position],
    dist_maps: List[torch.Tensor],
    congestion_map: torch.Tensor,
    cfg: Config,
):
    n = len(current)

    priorities = list(range(n))
    priorities.sort(
        key=lambda i: float(dist_maps[i][current[i][0], current[i][1]]),
        reverse=True,
    )

    next_pos: List[Optional[Position]] = [None for _ in range(n)]
    reserved: Dict[Position, int] = {}

    def choose(agent_id: int, visiting: set) -> bool:
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


# ============================================================
# 9. RHCR simulator
# ============================================================
def run_rhcr(
    obs: torch.Tensor,
    agents: List[Dict],
    cfg: Config,
    use_neural: bool = False,
    model=None,
):
    n = len(agents)

    current = [(ag["sy"], ag["sx"]) for ag in agents]
    goals = [(ag["gy"], ag["gx"]) for ag in agents]

    paths = [[current[i]] for i in range(n)]

    planning_calls = 0
    start_time = time.time()

    dist_maps = [get_bfs_distance_map(obs, goals[i]) for i in range(n)]

    used_windows = []

    for global_t in range(0, cfg.TOTAL_STEPS, cfg.REPLAN_PERIOD):
        planning_calls += 1

        if all(current[i] == goals[i] for i in range(n)):
            break

        if use_neural:
            heatmap = predict_neural_congestion(
                model=model,
                obs=obs,
                current=current,
                goals=goals,
                t=global_t,
                cfg=cfg,
            )
            current_window = choose_dynamic_window(heatmap, cfg)
        else:
            current_window = cfg.WINDOW_SIZE

        used_windows.append(current_window)

        # heatmap only controls horizon, not movement score
        congestion_map = build_zero_congestion(obs)

        # simplified RHCR execution:
        # execute h steps after each replanning decision
        execute_steps = min(cfg.REPLAN_PERIOD, current_window)

        for _ in range(execute_steps):
            if all(current[i] == goals[i] for i in range(n)):
                break

            next_pos = plan_one_step_with_scores(
                obs=obs,
                current=current,
                goals=goals,
                dist_maps=dist_maps,
                congestion_map=congestion_map,
                cfg=cfg,
            )

            current = next_pos

            for i in range(n):
                paths[i].append(current[i])

    runtime = time.time() - start_time

    avg_window = sum(used_windows) / max(1, len(used_windows))

    return paths, runtime, planning_calls, avg_window


# ============================================================
# 10. Metrics
# ============================================================
def count_collisions(paths: List[List[Position]]) -> int:
    max_len = max(len(p) for p in paths)
    n = len(paths)
    collisions = 0

    for t in range(max_len):
        positions = []
        for i in range(n):
            pos = paths[i][t] if t < len(paths[i]) else paths[i][-1]
            positions.append(pos)

        collisions += len(positions) - len(set(positions))

        if t > 0:
            prev_positions = []
            for i in range(n):
                prev = paths[i][t - 1] if t - 1 < len(paths[i]) else paths[i][-1]
                prev_positions.append(prev)

            for i in range(n):
                for j in range(i + 1, n):
                    if prev_positions[i] == positions[j] and prev_positions[j] == positions[i]:
                        collisions += 1

    return collisions


def success_rate(paths: List[List[Position]], goals: List[Position]) -> float:
    success = 0
    for i, g in enumerate(goals):
        if paths[i][-1] == g:
            success += 1
    return success / len(paths)


def total_path_length(paths: List[List[Position]]) -> int:
    return sum(len(p) for p in paths)


def evaluate_paths(paths, goals):
    return {
        "success": success_rate(paths, goals),
        "collisions": count_collisions(paths),
        "total_path_length": total_path_length(paths),
    }


# ============================================================
# 11. Main experiment
# ============================================================
def main():
    cfg = Config()
    set_seed(cfg.SEED)

    if cfg.WINDOW_SIZE < cfg.REPLAN_PERIOD:
        raise ValueError("RHCR requires WINDOW_SIZE >= REPLAN_PERIOD")

    print("=== Dynamic Neural-Horizon RHCR Experiment ===")
    print("Device:", cfg.DEVICE)
    print("N_AGENTS:", cfg.N_AGENTS)
    print("NUM_CASES:", cfg.NUM_CASES)
    print("DEFAULT WINDOW_SIZE w:", cfg.WINDOW_SIZE)
    print("REPLAN_PERIOD h:", cfg.REPLAN_PERIOD)
    print("DYNAMIC WINDOWS:", cfg.MIN_WINDOW_SIZE, cfg.MID_WINDOW_SIZE, cfg.MAX_WINDOW_SIZE)
    print("THRESHOLDS:", cfg.LOW_CONGESTION_TH, cfg.HIGH_CONGESTION_TH)
    print("TOTAL_STEPS:", cfg.TOTAL_STEPS)
    print("MODEL:", cfg.MODEL_PATH)

    model = load_unet_model(cfg)

    results = []

    attempts = 0
    pbar = tqdm(total=cfg.NUM_CASES, desc="Running RHCR cases")

    while len(results) < cfg.NUM_CASES and attempts < cfg.NUM_CASES * 100:
        attempts += 1

        obs, agents = generate_random_map_and_agents(cfg)
        if obs is None:
            continue

        if not all_agents_reachable(obs, agents):
            continue

        goals = [(ag["gy"], ag["gx"]) for ag in agents]

        lacam_paths = run_lacam_expert_wsl(obs, agents, cfg)
        if lacam_paths is None:
            continue

        lacam_metrics = evaluate_paths(lacam_paths, goals)

        vanilla_paths, vanilla_runtime, vanilla_calls, vanilla_avg_window = run_rhcr(
            obs=obs,
            agents=agents,
            cfg=cfg,
            use_neural=False,
            model=None,
        )

        neural_paths, neural_runtime, neural_calls, neural_avg_window = run_rhcr(
            obs=obs,
            agents=agents,
            cfg=cfg,
            use_neural=True,
            model=model,
        )

        vanilla_metrics = evaluate_paths(vanilla_paths, goals)
        neural_metrics = evaluate_paths(neural_paths, goals)

        row = {
            "lacam_success": lacam_metrics["success"],
            "lacam_collisions": lacam_metrics["collisions"],
            "lacam_total_path_length": lacam_metrics["total_path_length"],

            "vanilla_success": vanilla_metrics["success"],
            "vanilla_collisions": vanilla_metrics["collisions"],
            "vanilla_total_path_length": vanilla_metrics["total_path_length"],
            "vanilla_runtime": vanilla_runtime,
            "vanilla_calls": vanilla_calls,
            "vanilla_avg_window": vanilla_avg_window,

            "neural_success": neural_metrics["success"],
            "neural_collisions": neural_metrics["collisions"],
            "neural_total_path_length": neural_metrics["total_path_length"],
            "neural_runtime": neural_runtime,
            "neural_calls": neural_calls,
            "neural_avg_window": neural_avg_window,
        }

        results.append(row)
        pbar.update(1)

    pbar.close()

    if len(results) == 0:
        print("No valid cases.")
        return

    keys = results[0].keys()
    avg = {k: sum(r[k] for r in results) / len(results) for k in keys}

    print("\n==============================")
    print("Averaged Results")
    print("==============================")

    for k, v in avg.items():
        print(f"{k}: {v:.4f}")

    print("\n==============================")
    print("Comparison")
    print("==============================")
    print(f"Vanilla RHCR success: {avg['vanilla_success']:.4f}")
    print(f"Neural  RHCR success: {avg['neural_success']:.4f}")

    print(f"Vanilla RHCR collisions: {avg['vanilla_collisions']:.4f}")
    print(f"Neural  RHCR collisions: {avg['neural_collisions']:.4f}")

    print(f"Vanilla RHCR path length: {avg['vanilla_total_path_length']:.2f}")
    print(f"Neural  RHCR path length: {avg['neural_total_path_length']:.2f}")

    print(f"Vanilla RHCR runtime: {avg['vanilla_runtime']:.4f}")
    print(f"Neural  RHCR runtime: {avg['neural_runtime']:.4f}")

    print(f"Vanilla RHCR avg window: {avg['vanilla_avg_window']:.2f}")
    print(f"Neural  RHCR avg window: {avg['neural_avg_window']:.2f}")

    print("==============================")


if __name__ == "__main__":
    main()