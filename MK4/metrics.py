import numpy as np
import time, datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 사용 안 함 (메모리 절약)
import matplotlib.pyplot as plt


class MetricLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.save_log = save_dir / "log"
        
        # Episode 관련
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []
        
        # Step 관련 (현재 에피소드)
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = []
        self.curr_ep_q = []
        
        # 시간
        self.record_time = time.time()
        
        self.plot_every = 100  # 100 에피소드마다만 그래프 그리기
        
        # 로그 파일 초기화
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )

    def log_step(self, reward, loss, q):
        """매 스텝 호출"""
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss.append(loss)
        if q:
            self.curr_ep_q.append(q)

    def log_episode(self):
        """에피소드 종료 시 호출"""
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        
        if self.curr_ep_loss:
            self.ep_avg_losses.append(np.mean(self.curr_ep_loss))
        else:
            self.ep_avg_losses.append(0)
            
        if self.curr_ep_q:
            self.ep_avg_qs.append(np.mean(self.curr_ep_q))
        else:
            self.ep_avg_qs.append(0)
        
        # 초기화
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = []
        self.curr_ep_q = []

    def record(self, episode, epsilon, step):
        """주기적으로 통계 출력 및 그래프 저장"""
        mean_ep_reward = np.mean(self.ep_rewards[-100:])
        mean_ep_length = np.mean(self.ep_lengths[-100:])
        mean_ep_loss = np.mean(self.ep_avg_losses[-100:])
        mean_ep_q = np.mean(self.ep_avg_qs[-100:])
        
        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = self.record_time - last_record_time
        
        print(
            f"Episode {episode:>5} - "
            f"Step {step:>8} - "
            f"Epsilon {epsilon:>6.4f} - "
            f"Mean Reward {mean_ep_reward:>7.1f} - "
            f"Mean Length {mean_ep_length:>5.1f} - "
            f"Mean Loss {mean_ep_loss:>8.5f} - "
            f"Mean Q {mean_ep_q:>7.3f} - "
            f"Time {time_since_last_record:>5.1f}s"
        )
        
        # 로그 파일에 기록
        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.4f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}"
                f"{mean_ep_loss:15.5f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )
        
        if episode % self.plot_every == 0 and episode > 0:
            self._plot_and_save()

    def _plot_and_save(self):
        """
        그래프 그리기 및 저장
        
        Matplotlib 메모리 누수 방지:
        - plt.close('all') 반드시 호출
        - Figure 명시적으로 삭제
        """
        # 기존 그래프 모두 닫기
        plt.close('all')
        
        # Reward plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.ep_rewards, alpha=0.3, color='blue', label='Episode Reward')
        
        if len(self.ep_rewards) >= 100:
            moving_avg = np.convolve(self.ep_rewards, np.ones(100)/100, mode='valid')
            ax.plot(range(99, len(self.ep_rewards)), moving_avg, 
                   color='red', linewidth=2, label='Moving Average (100 ep)')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Training Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'reward_plot.jpg', dpi=100)
        
        plt.close(fig)
        del fig, ax
        
        # Loss plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.ep_avg_losses, alpha=0.3, color='green', label='Episode Avg Loss')
        
        if len(self.ep_avg_losses) >= 100:
            moving_avg = np.convolve(self.ep_avg_losses, np.ones(100)/100, mode='valid')
            ax.plot(range(99, len(self.ep_avg_losses)), moving_avg,
                   color='red', linewidth=2, label='Moving Average (100 ep)')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'loss_plot.jpg', dpi=100)
        
        plt.close(fig)
        del fig, ax
        
        plt.close('all')
        
        import gc
        gc.collect()