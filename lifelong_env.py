import random
from dataclasses import dataclass
from collections import deque
from typing import List, Tuple, Dict, Optional

import torch


Position = Tuple[int, int]


@dataclass
class LifelongConfig:
    H: int = 32
    W: int = 32
    N_AGENTS: int = 16
    SEED: int = 42


class LifelongMAPFEnv:
    def __init__(self, cfg: LifelongConfig):
        self.cfg = cfg
        self.obs = None
        self.agents = []
        self.current_positions = []
        self.goals = []
        self.completed_tasks = 0
        self.timestep = 0

        random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)

        self.reset()

    # =====================================================
    # 1. 地图生成
    # =====================================================
    def generate_random_map(self):
        cfg = self.cfg
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

        return obs

    def get_free_cells(self):
        cfg = self.cfg
        return [
            (y, x)
            for y in range(cfg.H)
            for x in range(cfg.W)
            if self.obs[y, x] < 0.5
        ]

    # =====================================================
    # 2. BFS 可达性
    # =====================================================
    def get_neighbors(self, pos: Position):
        y, x = pos
        H, W = self.obs.shape

        candidates = [
            (y, x),
            (y - 1, x),
            (y + 1, x),
            (y, x - 1),
            (y, x + 1),
        ]

        valid = []
        for ny, nx in candidates:
            if 0 <= ny < H and 0 <= nx < W and self.obs[ny, nx] < 0.5:
                valid.append((ny, nx))

        return valid

    def bfs_reachable(self, start: Position, goal: Position):
        if self.obs[start[0], start[1]] >= 0.5:
            return False

        if self.obs[goal[0], goal[1]] >= 0.5:
            return False

        q = deque([start])
        visited = {start}

        while q:
            y, x = q.popleft()

            if (y, x) == goal:
                return True

            for nxt in self.get_neighbors((y, x)):
                if nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)

        return False

    def sample_reachable_goal(self, start: Position, forbidden: Optional[set] = None):
        if forbidden is None:
            forbidden = set()

        free_cells = self.get_free_cells()
        random.shuffle(free_cells)

        for goal in free_cells:
            if goal == start:
                continue
            if goal in forbidden:
                continue
            if self.bfs_reachable(start, goal):
                return goal

        return None

    # =====================================================
    # 3. 初始化 lifelong task
    # =====================================================
    def reset(self):
        cfg = self.cfg

        while True:
            self.obs = self.generate_random_map()
            free_cells = self.get_free_cells()

            if len(free_cells) < cfg.N_AGENTS * 3:
                continue

            random.shuffle(free_cells)

            starts = free_cells[:cfg.N_AGENTS]

            current_positions = []
            goals = []
            ok = True

            used = set(starts)

            for i in range(cfg.N_AGENTS):
                start = starts[i]
                goal = self.sample_reachable_goal(start, forbidden=used)

                if goal is None:
                    ok = False
                    break

                current_positions.append(start)
                goals.append(goal)
                used.add(goal)

            if ok:
                break

        self.current_positions = current_positions
        self.goals = goals

        self.agents = []
        for i in range(cfg.N_AGENTS):
            self.agents.append({
                "id": i,
                "pos": self.current_positions[i],
                "goal": self.goals[i],
                "tasks_completed": 0,
            })

        self.completed_tasks = 0
        self.timestep = 0

        return self.get_state()

    # =====================================================
    # 4. 到达 goal 后分配新 goal
    # =====================================================
    def assign_new_goal_if_arrived(self, agent_id: int):
        pos = self.current_positions[agent_id]
        goal = self.goals[agent_id]

        if pos != goal:
            return False

        forbidden = set(self.current_positions)
        forbidden.update(self.goals)

        new_goal = self.sample_reachable_goal(pos, forbidden=forbidden)

        if new_goal is None:
            return False

        self.goals[agent_id] = new_goal
        self.agents[agent_id]["goal"] = new_goal
        self.agents[agent_id]["tasks_completed"] += 1
        self.completed_tasks += 1

        return True

    def assign_new_goals_for_arrived_agents(self):
        assigned_count = 0

        for i in range(self.cfg.N_AGENTS):
            assigned = self.assign_new_goal_if_arrived(i)
            if assigned:
                assigned_count += 1

        return assigned_count

    # =====================================================
    # 5. step: 外部 planner 给 next_positions
    # =====================================================
    def step(self, next_positions: List[Position]):
        assert len(next_positions) == self.cfg.N_AGENTS

        # 更新位置
        self.current_positions = list(next_positions)

        for i in range(self.cfg.N_AGENTS):
            self.agents[i]["pos"] = self.current_positions[i]

        self.timestep += 1

        # 到达目标后自动派新任务
        newly_completed = self.assign_new_goals_for_arrived_agents()

        return self.get_state(), newly_completed

    def get_state(self):
        return {
            "obs": self.obs,
            "current_positions": list(self.current_positions),
            "goals": list(self.goals),
            "agents": self.agents,
            "completed_tasks": self.completed_tasks,
            "timestep": self.timestep,
        }

    # =====================================================
    # 6. Debug 打印
    # =====================================================
    def print_summary(self):
        print("====== Lifelong MAPF Env Summary ======")
        print(f"Grid: {self.cfg.H} x {self.cfg.W}")
        print(f"Agents: {self.cfg.N_AGENTS}")
        print(f"Timestep: {self.timestep}")
        print(f"Completed tasks: {self.completed_tasks}")

        for i in range(min(5, self.cfg.N_AGENTS)):
            print(
                f"Agent {i}: "
                f"pos={self.current_positions[i]}, "
                f"goal={self.goals[i]}, "
                f"tasks={self.agents[i]['tasks_completed']}"
            )

        print("=======================================")


# =====================================================
# 7. 简单测试：用 greedy move 跑几步
# =====================================================
def greedy_next_positions(env: LifelongMAPFEnv):
    next_positions = []

    occupied = set()

    for i in range(env.cfg.N_AGENTS):
        cur = env.current_positions[i]
        goal = env.goals[i]

        candidates = env.get_neighbors(cur)

        candidates.sort(
            key=lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1])
        )

        chosen = cur

        for cand in candidates:
            if cand not in occupied:
                chosen = cand
                break

        occupied.add(chosen)
        next_positions.append(chosen)

    return next_positions


if __name__ == "__main__":
    cfg = LifelongConfig(
        H=32,
        W=32,
        N_AGENTS=16,
        SEED=42,
    )

    env = LifelongMAPFEnv(cfg)

    env.print_summary()

    TOTAL_STEPS = 100

    print("\nRunning simple greedy lifelong test...")

    for t in range(TOTAL_STEPS):
        next_pos = greedy_next_positions(env)
        state, newly_completed = env.step(next_pos)

        if newly_completed > 0:
            print(
                f"t={env.timestep}: "
                f"newly completed={newly_completed}, "
                f"total={env.completed_tasks}"
            )

    env.print_summary()

    print("\nFinished lifelong_env.py test.")