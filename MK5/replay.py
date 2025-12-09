import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation
from nes_py.wrappers import JoypadSpace
from pathlib import Path
from agent import Mario
from wrappers import ResizeObservation, SkipFrame

ACTION_SPACE = [
    ["right"],           # 0: 걷기
    ["right", "A"],      # 1: 점프하며 걷기
    ["right", "B"],      # 2: 달리기
    ["right", "A", "B"], # 3: 달리며 점프
]

# 1. 환경 생성 (학습 때와 똑같이 맞춰야 함!)
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, ACTION_SPACE)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)

# ⭐⭐⭐ [핵심 수정] TransformObservation 삭제! ⭐⭐⭐
# env = TransformObservation(env, f=lambda x: x / 255.)  <-- 이거 지워야 마리오가 앞을 봅니다.

env = FrameStack(env, num_stack=4)
env.reset()

# 2. 체크포인트 로드
# 경로가 맞는지 확인하세요 (파일이 실제로 존재하는지)
checkpoint_path = Path('/home/jwg/MadMario/MK5/checkpoints/2025-12-09T07-23-50/best_model.chkpt')

# 저장할 필요 없으니 save_dir은 아무거나
save_dir = Path('checkpoints') 

mario = Mario(
    state_dim=(4, 84, 84), 
    action_dim=env.action_space.n, 
    save_dir=save_dir, 
    checkpoint=checkpoint_path
)

# ⭐⭐⭐ [핵심 수정] 탐험률 0% 설정 (랜덤 행동 금지) ⭐⭐⭐
mario.exploration_rate = 0.0
# 학습 모드가 아니므로 burnin도 해제
mario.burnin = 0 

print("🎮 Test Drive Start!")

episodes = 5  # 5판만 구경

for e in range(episodes):
    state = env.reset()
    total_reward = 0
    
    while True:
        # 화면 출력 (속도가 너무 빠르면 time.sleep을 import해서 조절 가능)
        env.render()

        # 행동 결정 (탐험 없이 100% 실력으로)
        action = mario.act(state)

        next_state, reward, done, info = env.step(action)
        
        # 테스트 때는 cache(저장)나 learn(학습)을 하지 않습니다.
        # mario.cache(...) -> 삭제
        
        total_reward += reward
        state = next_state

        if done or info['flag_get']:
            break

    print(f"Episode {e+1} - Total Reward: {total_reward}")

env.close()