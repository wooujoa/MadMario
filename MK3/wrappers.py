"""
환경 생성 함수 - uint8 출력 (Agent에서 정규화)

핵심 수정:
TransformObservation(/ 255.0) 제거!
→ 환경은 uint8 (0~255) 출력
→ Agent가 필요할 때 / 255.0 수행
"""

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation
from nes_py.wrappers import JoypadSpace
import numpy as np

# Custom wrappers
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=self.observation_space.dtype)

    def observation(self, observation):
        from PIL import Image
        observation = Image.fromarray(observation)
        observation = observation.resize(self.shape)
        observation = np.array(observation)
        return observation


def create_mario_env(skip_frame=4):
    """
    Mario 환경 생성 함수
    
    중요: 환경은 uint8 (0~255) 출력!
    Agent가 필요할 때 / 255.0 수행!
    
    Args:
        skip_frame: 프레임 스킵 수 (기본 4 - 원래 프로젝트 설정)
    """
    # 환경 생성
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    
    # 액션 간소화
    env = JoypadSpace(env, [['right'], ['right', 'A']])
    
    # Frame skip
    env = SkipFrame(env, skip=skip_frame)
    
    # Grayscale
    env = GrayScaleObservation(env, keep_dim=False)
    
    # Resize
    env = ResizeObservation(env, shape=84)
    
    # Frame stack
    env = FrameStack(env, num_stack=4)
    
    print(f"✅ 환경 생성 완료 (SkipFrame={skip_frame})")
    print(f"   출력 타입: uint8 (0~255) ⭐ Agent에서 정규화!")
    
    return env