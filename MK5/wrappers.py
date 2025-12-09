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
from gym.vector import AsyncVectorEnv   # 병렬 환경용
import numpy as np

ACTION_SPACE = [
    ["right"],           # 0: 걷기
    ["right", "A"],      # 1: 점프하며 걷기
    ["right", "B"],      # 2: 달리기
    ["right", "A", "B"], # 3: 달리며 점프
]

# ---------------------------------------------------------------------
# Custom wrappers
# ---------------------------------------------------------------------

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

        old_shape = self.observation_space.shape
        # (H, W) 또는 (H, W, C) 모두 지원
        if len(old_shape) == 2:
            obs_shape = self.shape
        else:
            obs_shape = self.shape + old_shape[2:]

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        from PIL import Image

        img = Image.fromarray(observation)
        img = img.resize(self.shape)
        obs = np.array(img)
        return obs


# ---------------------------------------------------------------------
#  병렬 환경 생성용 함수들
# ---------------------------------------------------------------------

def make_single_env(skip_frame=4):
    """
    AsyncVectorEnv에 넘길, env 하나를 만드는 thunk.
    병렬/단일 둘 다에서 동일한 전처리 파이프라인을 재사용.
    """
    def _thunk():
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
        env = JoypadSpace(env, ACTION_SPACE)
        env = SkipFrame(env, skip=skip_frame)
        env = GrayScaleObservation(env, keep_dim=False)
        env = ResizeObservation(env, shape=84)
        env = FrameStack(env, num_stack=4)
        return env
    return _thunk


def create_mario_vec_env(num_envs=4, skip_frame=4):
    """
    병렬 Mario 환경 생성 함수 (AsyncVectorEnv 사용)

    Args:
        num_envs: 병렬로 돌릴 env 개수
        skip_frame: 프레임 스킵 수
    """
    env_fns = [make_single_env(skip_frame) for _ in range(num_envs)]
    vec_env = AsyncVectorEnv(env_fns)

    print(f"✅ 벡터 환경 생성 완료 (num_envs={num_envs}, SkipFrame={skip_frame})")
    print("   출력 타입: uint8 (0~255) ⭐ Agent에서 정규화!")
    print(f"   single_observation_space: {vec_env.single_observation_space.shape}")
    print(f"   single_action_space: {vec_env.single_action_space}")

    return vec_env
