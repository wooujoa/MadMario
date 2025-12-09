import random, datetime
from pathlib import Path
import torch

from metrics import MetricLogger
from agent import Mario  # agent_fast_learning.py를 agent.py로 복사
from wrappers import create_mario_env

env = create_mario_env(skip_frame=4)
save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

mario = Mario(
    state_dim=(4, 84, 84), 
    action_dim=env.action_space.n,
    save_dir=save_dir
)

logger = MetricLogger(save_dir)

episodes = 28000  # 빠른 검증

print(f"\n🎯 학습 목표: {episodes:,} 에피소드")
if episodes == 20000:
    print(f"   예상 시간: ~7시간")
    print(f"   예상 리워드: ~1900")
elif episodes == 40000:
    print(f"   예상 시간: ~13-14시간")
    print(f"   예상 리워드: ~2200")
print("-" * 80)

for e in range(episodes):
    state = env.reset()
    
    while True:
        action = mario.act(state)
        next_state, reward, done, info = env.step(action)
        mario.cache(state, next_state, action, reward, done)
        q, loss = mario.learn()
        logger.log_step(reward, loss, q)
        state = next_state
        
        if done or info['flag_get']:
            break
    
    logger.log_episode()
    mario.episode_finished()
    
    if e % 20 == 0:
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

env.close()
print(f"\n✅ {episodes:,} 에피소드 학습 완료!")