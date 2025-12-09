import random, datetime
from pathlib import Path
import torch

from metrics import MetricLogger
from agent import Mario
from wrappers import create_mario_vec_env   # ✅ 병렬 환경 생성 함수

# -----------------------------
# 1. 환경 / 에이전트 초기화
# -----------------------------

NUM_ENVS = 4          # 병렬로 돌릴 env 개수
SKIP_FRAME = 4

env = create_mario_vec_env(num_envs=NUM_ENVS, skip_frame=SKIP_FRAME)

save_dir = Path("checkpoints") / datetime.datetime.now().strftime(
    "%Y-%m-%dT%H-%M-%S"
)
save_dir.mkdir(parents=True, exist_ok=True)

mario = Mario(
    state_dim=env.single_observation_space.shape,   # (4, 84, 84)
    action_dim=env.single_action_space.n,           # Discrete(n)
    save_dir=save_dir,
)

logger = MetricLogger(save_dir)

# 학습할 총 에피소드 수 (모든 env 합산 기준)
target_episodes = 28000

print(f"\n🎯 학습 목표: {target_episodes:,} 에피소드 (병렬 env: {NUM_ENVS})")
print("-" * 80)

# -----------------------------
# 2. 병렬 학습 루프
# -----------------------------

# 각 env별 에피소드 리워드, 완료 횟수 추적
episode_rewards = [0.0 for _ in range(NUM_ENVS)]
episode_counts = [0 for _ in range(NUM_ENVS)]
total_episodes = 0

# 벡터 환경 초기 reset
state = env.reset()  # shape: (NUM_ENVS, 4, 84, 84)

last_logged_episode = 0

while total_episodes < target_episodes:
    # 1) 모든 env에 대해 액션 선택 (배치)
    actions = mario.act_batch(state)  # (NUM_ENVS,)

    # 2) 벡터 env step
    next_state, rewards, dones, infos = env.step(actions)
    # rewards, dones: shape (NUM_ENVS,)
    # infos: 길이가 NUM_ENVS인 리스트(dict)

    # 3) 각 env별로 transition 저장 / 학습 / 로깅
    for i in range(NUM_ENVS):
        s = state[i]
        ns = next_state[i]
        a = int(actions[i])
        r = float(rewards[i])
        d = bool(dones[i])
        info = infos[i]

        mario.cache(s, ns, a, r, d)
        q, loss = mario.learn()
        logger.log_step(r, loss, q)
        episode_rewards[i] += r

        # 해당 env의 에피소드가 끝난 경우
        if d or info.get("flag_get", False):
            total_episodes += 1
            episode_counts[i] += 1

            logger.log_episode()
            mario.episode_finished()
            episode_rewards[i] = 0.0

            # AsyncVectorEnv는 done=True인 env를 자동 reset해서
            # 다음 obs를 next_state[i]로 넣어주는 구현이 많다.
            # 별도 reset이 필요하면 여기서 env.reset_at(i) 등을 호출.

    # 4) 다음 step 준비
    state = next_state

    # 5) 주기적으로 기록
    if total_episodes >= last_logged_episode + 20:
        logger.record(
            episode=total_episodes,
            epsilon=mario.exploration_rate,
            step=mario.curr_step,
        )
        last_logged_episode = total_episodes

env.close()
print(f"\n✅ 총 {total_episodes:,} 에피소드 학습 완료! (병렬 env: {NUM_ENVS})")
