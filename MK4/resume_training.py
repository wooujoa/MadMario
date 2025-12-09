import random, datetime
from pathlib import Path
import torch

from metrics import MetricLogger
from agent import Mario
from wrappers import create_mario_env

# 1. 환경 생성
env = create_mario_env(skip_frame=4)

# 2. 새로운 저장 경로 생성 (이어하기 기록을 따로 저장)
# 기존 폴더에 섞이지 않게 현재 시간으로 새 폴더를 만듭니다.
save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S_resume')
save_dir.mkdir(parents=True)

# ⭐⭐⭐ 3. 체크포인트 경로 설정 (사용자가 제공한 경로) ⭐⭐⭐
# 이 파일에서 가중치(Weights)와 탐험률(Epsilon)을 복구합니다.
checkpoint_path = Path("/home/jwg/MadMario/checkpoints/2025-12-07T11-26-29/mario_net_2.chkpt")

print(f"🔄 학습 이어하기 모드 시작...")
print(f"📂 로드할 체크포인트: {checkpoint_path}")

# 4. 마리오 에이전트 생성 (체크포인트 로드)
mario = Mario(
    state_dim=(4, 84, 84), 
    action_dim=env.action_space.n,
    save_dir=save_dir,
    checkpoint=checkpoint_path  # 👈 여기에 경로를 넣어줍니다!
)

# ⭐⭐⭐ 5. Burn-in 제거 (핵심!) ⭐⭐⭐
# 이미 똑똑한 모델이므로 10만 스텝을 기다릴 필요가 없습니다.
# 버퍼에 배치 사이즈(32개)만 차면 바로 학습을 시작합니다.
mario.burnin = 0  
print(f"🔥 Burn-in 강제 해제: 0 (즉시 학습 시작)")
print(f"📊 현재 탐험률(Epsilon): {mario.exploration_rate:.4f}")

logger = MetricLogger(save_dir)

# 6. 추가 학습 목표 설정
# 이미 많이 학습했으므로 필요한 만큼 추가로 돌립니다.
episodes = 20000 

print(f"\n🎯 추가 학습 목표: {episodes:,} 에피소드")
print("-" * 80)

# 7. 학습 루프 (기존과 동일)
for e in range(episodes):
    state = env.reset()
    
    while True:
        action = mario.act(state)
        next_state, reward, done, info = env.step(action)
        
        mario.cache(state, next_state, action, reward, done)
        
        # burnin을 0으로 했기 때문에, 메모리가 32개 차는 순간부터 바로 learn()이 작동합니다.
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
print(f"\n✅ {episodes:,} 에피소드 추가 학습 완료!")