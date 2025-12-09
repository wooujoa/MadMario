import torch
import random, numpy as np
from pathlib import Path
import gc

from neural import MarioNet
from collections import deque


class Mario:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # ⭐ 원본 레포 설정 (yfeng997/MadMario)
        self.memory = deque(maxlen=100000)
        self.batch_size = 32

        self.exploration_rate = 1.0
        
        # ⭐⭐⭐ [수정 B] 탐험률 감소 속도 더 늦춤
        # 기존: 0.99999 (약 100만 스텝에 0.1 도달)
        # 수정: 0.999995 (약 200만 스텝에 0.1 도달)
        self.exploration_rate_decay = 0.999995
        self.exploration_rate_min = 0.1
        self.gamma = 0.9

        self.curr_step = 0
        self.burnin = 1e5    
        self.learn_every = 3
        
        # ⭐⭐⭐ [수정 C] Target Network 동기화 주기 단축
        # 기존: 10000 (Q값 폭발 위험)
        # 수정: 5000 (더 안정적인 학습)
        self.sync_every = 5000

        self.save_every = 5e5
        self.save_dir = save_dir
        
        # Best checkpoint tracking
        self.episode_rewards = deque(maxlen=100)
        self.best_mean_reward = -float('inf')
        self.current_episode_reward = 0

        self.use_cuda = torch.cuda.is_available()
        
        if self.use_cuda:
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("❌ CUDA not available, using CPU")

        # 네트워크 초기화
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device='cuda')
            
        print(f"\n⚙️  Updated Settings (안정화 최적화):")
        print(f"   Replay buffer: {self.memory.maxlen:,}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Burnin: {int(self.burnin):,}")
        print(f"   Learn every: {self.learn_every} steps")
        print(f"   Sync every: {int(self.sync_every):,} ⭐ 5000으로 단축!")
        print(f"\n🎯 Exploration (핵심 수정!):")
        print(f"   Initial rate: {self.exploration_rate}")
        print(f"   Decay: {self.exploration_rate_decay} ⭐ 더 느리게!")
        print(f"   Min rate: {self.exploration_rate_min}")
        print(f"   Burn-in 동안: Epsilon 동결 ⭐⭐⭐")
        print(f"   예상: Episode 3,000~4,000에 0.1 도달")
        print(f"\n🔧 안정화 기법:")
        print(f"   Gradient Clipping: max_norm=10.0 ⭐⭐⭐")
        print(f"   메모리: uint8 (CPU) → float/255.0 (GPU)")
        print(f"\n🏆 Best checkpoint tracking: ON")
        print(f"📁 Checkpoints: {save_dir}")
        
        if checkpoint:
            self.load(checkpoint)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()


    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.
        
        ⭐⭐⭐ [수정 1] Burn-in 기간 동안 Epsilon 동결!
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = torch.FloatTensor(state)
            if self.use_cuda:
                state = state.cuda()
            
            state = state.unsqueeze(0)
            state = state / 255.0
            
            action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values, axis=1).item()

        # ⭐⭐⭐ [핵심 수정] Burn-in이 끝난 후에만 Epsilon 감소
        # 이유: 학습 전에는 탐험률을 유지해야 다양한 경험 수집 가능
        if self.curr_step >= self.burnin:
            self.exploration_rate *= self.exploration_rate_decay
            self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action_idx


    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)
        
        ⭐ 메모리 최적화: CPU에서 uint8로 저장 (255배 절약!)
        """
        state = np.array(state, dtype=np.uint8)
        next_state = np.array(next_state, dtype=np.uint8)
        
        self.memory.append((state, next_state, action, reward, done))
        self.current_episode_reward += reward


    def recall(self):
        """
        Retrieve a batch of experiences from memory
        
        ⭐⭐⭐ 핵심 최적화: CPU에서 uint8 유지, GPU에서 float 변환
        ⭐⭐⭐ state / 255.0으로 0~1 범위로 정규화!
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = zip(*batch)
        
        # CPU에서 uint8로 유지
        state = torch.ByteTensor(np.array(state))
        next_state = torch.ByteTensor(np.array(next_state))
        
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.BoolTensor(done)
        
        # GPU로 이동
        if self.use_cuda:
            state = state.cuda()
            next_state = next_state.cuda()
            action = action.cuda()
            reward = reward.cuda()
            done = done.cuda()
        
        # ⭐⭐⭐ float 변환 + 255로 나누기 (0~1 범위)
        return state.float() / 255.0, next_state.float() / 255.0, action, reward, done


    def td_estimate(self, state, action):
        """TD Estimate: Q(s,a)"""
        current_Q = self.net(state, model='online')[np.arange(0, self.batch_size), action]
        return current_Q


    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        """TD Target using Double DQN"""
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()


    def update_Q_online(self, td_estimate, td_target):
        """
        Backpropagate loss through Q_online
        
        ⭐⭐⭐ [수정 D] Gradient Clipping 추가!
        """
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        
        # ⭐⭐⭐ [핵심 추가] Gradient Clipping
        # Q값 폭발(0→24) 방지를 위한 가장 강력한 안전장치
        # max_norm=10: 기울기 L2 norm이 10을 넘으면 자동으로 스케일 조정
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        return loss.item()


    def sync_Q_target(self):
        """Copy weights from online network to target network"""
        self.net.target.load_state_dict(self.net.online.state_dict())


    def learn(self):
        """
        Update the Q-network with a batch of experiences
        """
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        if len(self.memory) < self.batch_size:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)
        
        q_value = float(td_est.mean().item())
        
        # 메모리 정리
        del state, next_state, action, reward, done
        del td_est, td_tgt
        
        if self.use_cuda and self.curr_step % 100 == 0:
            torch.cuda.empty_cache()
        
        if self.curr_step % 500 == 0:
            gc.collect()

        return (q_value, loss)


    def episode_finished(self):
        """에피소드 종료 시 호출: best checkpoint 체크"""
        self.episode_rewards.append(self.current_episode_reward)
        
        if len(self.episode_rewards) >= 100:
            mean_reward = np.mean(self.episode_rewards)
            
            if mean_reward > self.best_mean_reward:
                old_best = self.best_mean_reward
                self.best_mean_reward = mean_reward
                self.save_best()
                print(f"\n🏆 NEW BEST! Mean Reward: {mean_reward:.1f} (이전: {old_best:.1f})")
                self.current_episode_reward = 0
                return True
        
        self.current_episode_reward = 0
        
        if self.use_cuda:
            torch.cuda.empty_cache()
        gc.collect()
        
        return False


    def save(self):
        """주기적 체크포인트 저장"""
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate,
                best_mean_reward=self.best_mean_reward
            ),
            save_path
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")


    def save_best(self):
        """최고 성능 모델 저장"""
        save_path = self.save_dir / "best_model.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate,
                best_mean_reward=self.best_mean_reward,
                step=self.curr_step
            ),
            save_path
        )
        print(f"   ✅ Best model saved to {save_path}")


    def load(self, load_path):
        """Load a saved checkpoint"""
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')
        best_mean_reward = ckp.get('best_mean_reward', -float('inf'))

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        if best_mean_reward > -float('inf'):
            print(f"   Best mean reward from checkpoint: {best_mean_reward:.1f}")
        
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate
        self.best_mean_reward = best_mean_reward