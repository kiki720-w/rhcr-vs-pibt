import os
import re
import glob
import random
import shutil
import subprocess
from collections import deque, Counter

import torch
import torch.nn.functional as F
from tqdm import tqdm


# 1. 配置
class DataConfig:
    H = 32
    W = 32
    N_AGENTS = 16

    NUM_TRAIN = 5000
    NUM_VAL = 500
    NUM_TEST = 500

    SAVE_DIR = "./dataset_v2_random"
    TEMP_DIR = "./dataset_v2_random/temp"

    WSL_LACAM_EXE = "/home/wlf/lacam/build/main"

    LACAM_TIME_LIMIT_SEC = 20
    SUBPROCESS_TIMEOUT_SEC = 60

    MAX_TOTAL_ATTEMPTS_MULTIPLIER = 300
    LOG_EVERY_N_FAILS = 20

    # RHCR-style rolling window
    # WINDOW_SIZE 对应论文里的 time horizon w
    # REPLAN_PERIOD 对应论文里的 replanning period h
    # 要求 WINDOW_SIZE >= REPLAN_PERIOD
    WINDOW_SIZE = 10
    REPLAN_PERIOD = 5

    HEATMAP_SIGMA = 1.0
    HEATMAP_KERNEL_SIZE = 5

    RESET_DATASET = True


# 2. 工具函数
def reset_split_dir(split_dir):
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)
    os.makedirs(split_dir, exist_ok=True)


def windows_to_wsl_path(win_path: str) -> str:
    win_path = os.path.abspath(win_path)
    drive = win_path[0].lower()
    rest = win_path[2:].replace("\\", "/")
    return f"/mnt/{drive}{rest}"


def gaussian_smooth_heatmap(heatmap, sigma=1.0, kernel_size=5):
    assert kernel_size % 2 == 1

    dtype = heatmap.dtype
    device = heatmap.device

    coords = torch.arange(kernel_size, dtype=dtype, device=device) - kernel_size // 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")

    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)

    x = heatmap.view(1, 1, heatmap.shape[0], heatmap.shape[1])
    x = F.conv2d(x, kernel, padding=kernel_size // 2)

    return x.squeeze(0).squeeze(0)


# 3. 随机地图生成
def generate_random_map_and_agents(cfg):
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


# 4. MovingAI 格式导出
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


# 5. 可达性检查
def bfs_reachable(obs, sy, sx, gy, gx):
    if obs[sy, sx] >= 0.5 or obs[gy, gx] >= 0.5:
        return False

    H, W = obs.shape
    q = deque([(sy, sx)])
    visited = {(sy, sx)}

    while q:
        y, x = q.popleft()

        if (y, x) == (gy, gx):
            return True

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx

            if (
                0 <= ny < H
                and 0 <= nx < W
                and obs[ny, nx] < 0.5
                and (ny, nx) not in visited
            ):
                visited.add((ny, nx))
                q.append((ny, nx))

    return False


def all_agents_reachable(obs, agents):
    for ag in agents:
        if not bfs_reachable(obs, ag["sy"], ag["sx"], ag["gy"], ag["gx"]):
            return False
    return True


# 6. 解析 LaCAM 输出
def parse_official_lacam_output(out_file, num_agents, verbose=False):
    if not os.path.exists(out_file):
        if verbose:
            print("[WARN] out.txt not found")
        return None

    text = open(out_file, "r", encoding="utf-8", errors="ignore").read().strip()

    if not text:
        if verbose:
            print("[WARN] out.txt empty")
        return None

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("solution"):
            start_idx = i + 1
            break

    if start_idx is None:
        if verbose:
            print("[WARN] no solution= found")
            print(text[:1000])
        return None

    sol_lines = lines[start_idx:]
    parsed_lines = []

    for line in sol_lines:
        m = re.match(r"^(\d+)\s*:\s*(.*)$", line)
        if not m:
            continue

        idx = int(m.group(1))
        coords_str = m.group(2)
        matches = re.findall(r"\((\d+)\s*,\s*(\d+)\)", coords_str)

        coords = [(int(y), int(x)) for x, y in matches]
        parsed_lines.append((idx, coords))

    if len(parsed_lines) == 0:
        if verbose:
            print("[WARN] no valid solution lines")
            print(text[:1000])
        return None

    # agent-major
    if len(parsed_lines) == num_agents:
        paths = [[] for _ in range(num_agents)]

        for idx, coords in parsed_lines:
            if 0 <= idx < num_agents:
                paths[idx] = coords

        if all(len(p) > 0 for p in paths):
            if verbose:
                print("[INFO] Parsed as agent-major format")
                print("path lens:", [len(p) for p in paths])
            return paths

    # time-major
    if all(len(coords) == num_agents for _, coords in parsed_lines):
        parsed_lines.sort(key=lambda x: x[0])
        time_steps = [coords for _, coords in parsed_lines]

        paths = [[] for _ in range(num_agents)]

        for coords_at_t in time_steps:
            for agent_id in range(num_agents):
                paths[agent_id].append(coords_at_t[agent_id])

        if all(len(p) > 0 for p in paths):
            if verbose:
                print("[INFO] Parsed as time-major format")
                print("path lens:", [len(p) for p in paths])
                print("agent0 first 10:", paths[0][:10])
            return paths

    if verbose:
        print("[WARN] unable to determine solution layout")
        print("num parsed lines:", len(parsed_lines))
        print("first 3 parsed lines:", parsed_lines[:3])

    return None


# 7. 调用 WSL LaCAM
def run_lacam_expert_wsl(obs, agents, cfg, verbose=False):
    os.makedirs(cfg.TEMP_DIR, exist_ok=True)

    win_map_file = os.path.abspath(os.path.join(cfg.TEMP_DIR, "temp.map"))
    win_scen_file = os.path.abspath(os.path.join(cfg.TEMP_DIR, "temp.scen"))
    win_out_file = os.path.abspath(os.path.join(cfg.TEMP_DIR, "out.txt"))

    if os.path.exists(win_out_file):
        os.remove(win_out_file)

    export_to_movingai(obs, agents, win_map_file, win_scen_file)

    wsl_map_file = windows_to_wsl_path(win_map_file)
    wsl_scen_file = windows_to_wsl_path(win_scen_file)
    wsl_out_file = windows_to_wsl_path(win_out_file)

    cmd = [
        "wsl",
        cfg.WSL_LACAM_EXE,
        "-m", wsl_map_file,
        "-i", wsl_scen_file,
        "-N", str(len(agents)),
        "-t", str(cfg.LACAM_TIME_LIMIT_SEC),
        "-o", wsl_out_file,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=cfg.SUBPROCESS_TIMEOUT_SEC,
        )

        if verbose:
            print("\n[WSL LaCAM CMD]")
            print(" ".join(cmd))
            print("[returncode]", result.returncode)
            print("[stdout]")
            print(result.stdout if result.stdout.strip() else "(empty)")
            print("[stderr]")
            print(result.stderr if result.stderr.strip() else "(empty)")
            print("[out exists]", os.path.exists(win_out_file))
            if os.path.exists(win_out_file):
                print("[out size]", os.path.getsize(win_out_file))

        if result.returncode != 0:
            if verbose:
                print("[WARN] WSL LaCAM returned non-zero")
            return None

        return parse_official_lacam_output(
            win_out_file,
            len(agents),
            verbose=verbose,
        )

    except subprocess.TimeoutExpired as e:
        if verbose:
            print("[TIMEOUT]", repr(e))
        return None

    except Exception as e:
        if verbose:
            print("[EXCEPTION]", repr(e))
        return None


# 8. BFS 距离图
def get_bfs_distance_map(obs, gy, gx):
    H, W = obs.shape
    dist = torch.full((H, W), 1e9, dtype=torch.float32)

    if obs[gy, gx] >= 0.5:
        return dist

    dist[gy, gx] = 0.0
    q = deque([(gy, gx)])

    while q:
        y, x = q.popleft()

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx

            if 0 <= ny < H and 0 <= nx < W and obs[ny, nx] < 0.5:
                if dist[ny, nx] > dist[y, x] + 1:
                    dist[ny, nx] = dist[y, x] + 1
                    q.append((ny, nx))

    return dist


# 9. 动作编码
def get_action_index(cy, cx, ny, nx):
    if cy == ny and cx == nx:
        return 0
    elif ny == cy - 1 and nx == cx:
        return 1
    elif ny == cy + 1 and nx == cx:
        return 2
    elif ny == cy and nx == cx - 1:
        return 3
    elif ny == cy and nx == cx + 1:
        return 4

    return 0


# 10. 构造 9 通道 + RHCR rolling-window heatmap 多任务样本
def build_training_sample(obs, agents, expert_paths, cfg):
    H, W = obs.shape

    max_len = max(len(p) for p in expert_paths)
    if max_len <= 1:
        return None

    # 只选择至少一个 agent 在移动的时间步
    candidate_ts = []

    for t in range(max_len - 1):
        moved = False

        for p in expert_paths:
            cur = p[t] if t < len(p) else p[-1]
            nxt = p[t + 1] if t + 1 < len(p) else p[-1]

            if cur != nxt:
                moved = True
                break

        if moved:
            candidate_ts.append(t)

    if len(candidate_ts) == 0:
        return None

    t = random.choice(candidate_ts)

    f_map = obs.clone()
    f_cur = torch.zeros((H, W), dtype=torch.float32)
    f_goal = torch.zeros((H, W), dtype=torch.float32)
    f_cg = torch.zeros((H, W), dtype=torch.float32)
    f_grad_x = torch.zeros((H, W), dtype=torch.float32)
    f_grad_y = torch.zeros((H, W), dtype=torch.float32)

    f_capacity = torch.ones((H, W), dtype=torch.float32)
    f_capacity[obs >= 0.5] = 0.0

    f_time = torch.full(
        (H, W),
        float(t) / float(max_len),
        dtype=torch.float32,
    )

    f_flow = torch.zeros((H, W), dtype=torch.float32)
    label = torch.full((H, W), -1, dtype=torch.long)

    heatmap_target = torch.zeros((H, W), dtype=torch.float32)

    # 输入特征：全路径访问频率
    for p in expert_paths:
        for y, x in p:
            f_flow[y, x] += 1.0

    f_flow = f_flow / float(len(expert_paths))
    f_flow[obs >= 0.5] = 0.0

    # RHCR-style rolling-window heatmap:
    # 统计未来 WINDOW_SIZE 步窗口内的 occupancy / congestion
    for p in expert_paths:
        end_t = min(t + cfg.WINDOW_SIZE, len(p) - 1)

        for future_t in range(t, end_t + 1):
            y, x = p[future_t]
            heatmap_target[y, x] += 1.0

    heatmap_target = heatmap_target / float(len(expert_paths))

    # 高斯扩散，让 rolling-window heatmap 更平滑、更适合学习
    heatmap_target = gaussian_smooth_heatmap(
        heatmap_target,
        sigma=cfg.HEATMAP_SIGMA,
        kernel_size=cfg.HEATMAP_KERNEL_SIZE,
    )

    heatmap_target[obs >= 0.5] = 0.0

    max_val = heatmap_target.max()
    if max_val > 0:
        heatmap_target = heatmap_target / max_val

    heatmap_target = torch.clamp(heatmap_target, 0.0, 1.0)

    # 当前 agent 状态 + 当前一步动作标签
    for i, ag in enumerate(agents):
        p = expert_paths[i]

        cy, cx = p[t] if t < len(p) else p[-1]
        ny, nx = p[t + 1] if t + 1 < len(p) else p[-1]
        gy, gx = ag["gy"], ag["gx"]

        f_cur[cy, cx] = 1.0
        f_goal[gy, gx] = 1.0

        dist_map = get_bfs_distance_map(obs, gy, gx)
        f_cg[cy, cx] = dist_map[cy, cx]

        if cx > 0 and dist_map[cy, cx - 1] < dist_map[cy, cx]:
            f_grad_x[cy, cx] = -1.0
        elif cx < W - 1 and dist_map[cy, cx + 1] < dist_map[cy, cx]:
            f_grad_x[cy, cx] = 1.0

        if cy > 0 and dist_map[cy - 1, cx] < dist_map[cy, cx]:
            f_grad_y[cy, cx] = -1.0
        elif cy < H - 1 and dist_map[cy + 1, cx] < dist_map[cy, cx]:
            f_grad_y[cy, cx] = 1.0

        label[cy, cx] = get_action_index(cy, cx, ny, nx)

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

    assert map_x.shape == (9, H, W), f"map_x shape error: {map_x.shape}"
    assert label.shape == (H, W), f"label shape error: {label.shape}"
    assert heatmap_target.shape == (H, W), f"heatmap shape error: {heatmap_target.shape}"

    return {
        "map_x": map_x,
        "label": label,
        "heatmap_target": heatmap_target,
        "window_size": cfg.WINDOW_SIZE,
        "replan_period": cfg.REPLAN_PERIOD,
    }


# 11. 生成 split
def generate_split_data(mode, num_samples):
    cfg = DataConfig()
    save_path = os.path.join(cfg.SAVE_DIR, mode)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(cfg.TEMP_DIR, exist_ok=True)

    print(f"Generating {num_samples} samples for [{mode}] ...")

    success_count = 0
    fail_count = 0
    attempt_count = 0
    max_attempts = num_samples * cfg.MAX_TOTAL_ATTEMPTS_MULTIPLIER

    pbar = tqdm(total=num_samples, desc=f"[{mode.upper()}]", unit="sample")

    while success_count < num_samples and attempt_count < max_attempts:
        attempt_count += 1

        obs, agents = generate_random_map_and_agents(cfg)
        if obs is None:
            fail_count += 1
            continue

        if not all_agents_reachable(obs, agents):
            fail_count += 1
            continue

        verbose = fail_count < 2

        expert_paths = run_lacam_expert_wsl(obs, agents, cfg, verbose=verbose)
        if expert_paths is None:
            fail_count += 1

            if fail_count % cfg.LOG_EVERY_N_FAILS == 0:
                print(
                    f"[{mode}] fail={fail_count}, "
                    f"success={success_count}, attempts={attempt_count}"
                )
            continue

        sample = build_training_sample(obs, agents, expert_paths, cfg)
        if sample is None:
            fail_count += 1
            continue

        assert sample["map_x"].shape[0] == 9
        assert "heatmap_target" in sample
        assert "window_size" in sample
        assert "replan_period" in sample

        out_file = os.path.join(save_path, f"sample_{success_count:05d}.pt")
        torch.save(sample, out_file)

        success_count += 1
        pbar.update(1)

    pbar.close()

    print(f"\n[{mode}] done.")
    print(f"success_count = {success_count}")
    print(f"fail_count    = {fail_count}")
    print(f"attempt_count = {attempt_count}")

    if success_count < num_samples:
        print(
            f"[WARN] only generated {success_count}/{num_samples} samples.\n"
            f"Try:\n"
            f"1) reduce N_AGENTS\n"
            f"2) increase LACAM_TIME_LIMIT_SEC\n"
            f"3) increase SUBPROCESS_TIMEOUT_SEC"
        )


# 12. 检查数据
def inspect_dataset(root="./dataset_v2_random"):
    for split in ["train", "val", "test"]:
        data_dir = os.path.join(root, split)
        files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))

        channel_counter = Counter()
        label_counter = Counter()
        heatmap_min = []
        heatmap_max = []
        heatmap_mean = []
        heatmap_nonzero = []
        window_counter = Counter()
        replan_counter = Counter()

        for f in files:
            data = torch.load(f, map_location="cpu", weights_only=False)

            channel_counter[data["map_x"].shape[0]] += 1

            valid = data["label"][data["label"] != -1]
            label_counter.update(valid.tolist())

            if "heatmap_target" in data:
                hm = data["heatmap_target"]
                heatmap_min.append(hm.min().item())
                heatmap_max.append(hm.max().item())
                heatmap_mean.append(hm.mean().item())
                heatmap_nonzero.append((hm > 0).sum().item())

            if "window_size" in data:
                window_counter[int(data["window_size"])] += 1

            if "replan_period" in data:
                replan_counter[int(data["replan_period"])] += 1

        print(f"\n[{split}]")
        print(f"samples: {len(files)}")
        print(f"channels: {channel_counter}")
        print(f"labels: {label_counter}")
        print(f"window_size: {window_counter}")
        print(f"replan_period: {replan_counter}")

        if heatmap_max:
            print(f"heatmap min range: {min(heatmap_min):.4f} ~ {max(heatmap_min):.4f}")
            print(f"heatmap max range: {min(heatmap_max):.4f} ~ {max(heatmap_max):.4f}")
            print(f"heatmap mean range: {min(heatmap_mean):.6f} ~ {max(heatmap_mean):.6f}")
            print(f"heatmap nonzero range: {min(heatmap_nonzero)} ~ {max(heatmap_nonzero)}")
        else:
            print("heatmap_target missing")


# 13. 主函数
if __name__ == "__main__":
    cfg = DataConfig()

    print("=== Starting RHCR-style Rolling-Window Dataset Generation via WSL LaCAM ===")
    print("WSL_LACAM_EXE =", cfg.WSL_LACAM_EXE)
    print("SAVE_DIR      =", os.path.abspath(cfg.SAVE_DIR))
    print("TEMP_DIR      =", os.path.abspath(cfg.TEMP_DIR))
    print("WINDOW_SIZE w =", cfg.WINDOW_SIZE)
    print("REPLAN_PERIOD h =", cfg.REPLAN_PERIOD)
    print("HEATMAP_SIGMA =", cfg.HEATMAP_SIGMA)

    if cfg.WINDOW_SIZE < cfg.REPLAN_PERIOD:
        raise ValueError("RHCR requires WINDOW_SIZE >= REPLAN_PERIOD")

    if cfg.RESET_DATASET:
        reset_split_dir(os.path.join(cfg.SAVE_DIR, "train"))
        reset_split_dir(os.path.join(cfg.SAVE_DIR, "val"))
        reset_split_dir(os.path.join(cfg.SAVE_DIR, "test"))
        os.makedirs(cfg.TEMP_DIR, exist_ok=True)

    generate_split_data("val", cfg.NUM_VAL)
    generate_split_data("test", cfg.NUM_TEST)
    generate_split_data("train", cfg.NUM_TRAIN)

    print("Inspecting generated dataset...")
    inspect_dataset(cfg.SAVE_DIR)

    print("Finished.")