"""
Best Model 평가 스크립트

사용법:
1. 학습 완료 후 실행
2. best_model.chkpt를 로드하여 평가
3. 10 에피소드 실행 후 통계 출력
"""

from pathlib import Path
import torch
import numpy as np
from agent import Mario
from wrappers import create_mario_env

print("="*80)
print("🏆 Best Model 평가")
print("="*80)

# 체크포인트 디렉토리 선택
checkpoint_dir = input("체크포인트 디렉토리 경로를 입력하세요 (예: checkpoints/2025-12-04T12-30-00): ")
save_dir = Path(checkpoint_dir)

if not save_dir.exists():
    print(f"❌ 디렉토리를 찾을 수 없습니다: {save_dir}")
    exit(1)

# Best model 경로
best_model_path = save_dir / 'best_model.chkpt'

if not best_model_path.exists():
    print(f"❌ best_model.chkpt를 찾을 수 없습니다: {best_model_path}")
    print("학습이 완료되지 않았거나 Episode 100 이상 학습하지 않았을 수 있습니다.")
    exit(1)

print(f"\n✅ Best model 발견: {best_model_path}")

# 환경 생성
env = create_mario_env(skip_frame=4)

# Agent 생성 및 best model 로드
mario = Mario(
    state_dim=(4, 84, 84),
    action_dim=env.action_space.n,
    save_dir=save_dir,
    checkpoint=best_model_path
)

print(f"\n📊 Best model 정보:")
print(f"   Best mean reward: {mario.best_mean_reward:.1f}")
print(f"   Exploration rate: {mario.exploration_rate:.4f}")

# Epsilon을 0으로 설정 (순수 exploitation)
mario.exploration_rate = 0.0
print(f"   평가 모드: Epsilon = 0.0 (exploitation only)")

# 평가 에피소드 수
num_episodes = 10
print(f"\n🎮 {num_episodes} 에피소드 평가 시작...")
print("-" * 80)

# 평가 실행
episode_rewards = []
episode_steps = []
flag_gets = 0

for ep in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    steps = 0
    
    while True:
        # Action 선택 (exploitation only)
        action = mario.act(state)
        
        # Step
        next_state, reward, done, info = env.step(action)
        
        episode_reward += reward
        steps += 1
        state = next_state
        
        # 종료 조건
        if done or info['flag_get']:
            if info['flag_get']:
                flag_gets += 1
            break
    
    episode_rewards.append(episode_reward)
    episode_steps.append(steps)
    
    flag_icon = "🚩" if info.get('flag_get', False) else "  "
    print(f"Episode {ep+1:2d}: Reward = {episode_reward:6.1f}, Steps = {steps:4d} {flag_icon}")

# 통계 계산
mean_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)
max_reward = np.max(episode_rewards)
min_reward = np.min(episode_rewards)
mean_steps = np.mean(episode_steps)

print("-" * 80)
print(f"\n📊 평가 결과:")
print(f"   평균 보상: {mean_reward:.1f} ± {std_reward:.1f}")
print(f"   최고 보상: {max_reward:.1f}")
print(f"   최저 보상: {min_reward:.1f}")
print(f"   평균 스텝: {mean_steps:.1f}")
print(f"   깃발 도달: {flag_gets}/{num_episodes} ({flag_gets/num_episodes*100:.1f}%)")

# 학습 시 best mean reward와 비교
print(f"\n🏆 비교:")
print(f"   학습 시 best mean reward: {mario.best_mean_reward:.1f}")
print(f"   평가 평균 보상:           {mean_reward:.1f}")
diff = mean_reward - mario.best_mean_reward
print(f"   차이:                     {diff:+.1f}")

if abs(diff) < 50:
    print("   ✅ 학습 시 성능과 유사합니다.")
elif diff > 0:
    print("   🎉 평가 성능이 더 좋습니다! (운이 좋았을 수 있음)")
else:
    print("   ⚠️  평가 성능이 낮습니다. (샘플 수가 적어서 그럴 수 있음)")

print("\n" + "="*80)
print("✅ 평가 완료!")
print("="*80)

env.close()