# 🍄 Super Mario Bros RL Agent

<div align="center">

**강화학습(Reinforcement Learning)을 통해 슈퍼 마리오 브라더스 스테이지를 클리어하는 AI 에이전트**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📋 목차

- [프로젝트 소개](#-프로젝트-소개)
- [주요 특징](#-주요-특징)
- [환경 설정](#-환경-설정)
- [모델 버전](#-모델-버전)
- [사용 방법](#-사용-방법)
- [프로젝트 구조](#-프로젝트-구조)

---

## 🎯 프로젝트 소개

이 프로젝트는 Deep Q-Network(DQN) 알고리즘을 기반으로 슈퍼 마리오 브라더스 게임을 학습하는 AI 에이전트를 구현합니다. 

### 🔬 연구 목표
- **Double DQN (DDQN)** 기본 구현
- **Dueling DQN** 및 다양한 DQN 변형 실험
- 알고리즘별 성능 비교 및 분석

### 🎓 핵심 기술
- Experience Replay를 활용한 효율적 학습
- Target Network를 통한 학습 안정화
- CNN 기반 상태 인식 및 Q-value 추정

---

## ✨ 주요 특징

- 🧠 **Deep Q-Learning**: DDQN 알고리즘 기반 강화학습
- 🎮 **실시간 학습**: 게임 플레이를 통한 자동 학습
- 📊 **성능 추적**: 학습 진행 상황 시각화 및 모니터링
- 💾 **체크포인트 저장**: 학습된 모델 자동 저장 및 복원
- 🔄 **다양한 버전**: MK3, MK4, MK5 실험 버전 관리

---

## 🛠 환경 설정

### 필수 요구사항
- Python 3.8+
- CUDA 지원 GPU (권장)
- Conda 또는 Miniconda

### 설치 방법

1. **레포지토리 클론**
```bash
   git clone https://github.com/wooujoa/MadMario.git
   cd MadMario
```

2. **Conda 환경 생성**
```bash
   conda env create -f environment.yml
```

3. **가상환경 활성화**
```bash
   conda activate mario_env
```

---

## 📦 모델 버전

프로젝트는 실험 단계에 따라 3가지 버전으로 구성되어 있습니다.

### 📂 MK3 - Basic Agent
초기 기본 모델입니다.

**특징:**
- 기본적인 액션 스페이스 구현
- 기초 리워드 구조 테스트

**실행:**
```bash
cd MK3
```

### 📂 MK4 - Improved Agent
개선된 중급 모델입니다.

**특징:**
- 하이퍼파라미터 튜닝 적용
- 학습 안정성 개선
- MK3 대비 향상된 성능

**실행:**
```bash
cd MK4
```

### 📂 MK5 - Advanced Agent
최신 고급 모델입니다.

**특징:**
- 확장된 액션 스페이스 (달리기+점프 조합)
- 최적화된 네트워크 구조
- 복잡한 구간 돌파 가능

**실행:**
```bash
cd MK5
```

---

## 🚀 사용 방법

### 1️⃣ 새로운 학습 시작

원하는 모델 버전 디렉토리로 이동 후 실행:
```bash
# 예시: MK5 모델 학습
cd MK5
python main.py
```

학습 과정에서 생성되는 파일:
- 📊 학습 로그 (에피소드별 리워드, 손실 등)
- 💾 체크포인트 파일 (`checkpoints/` 폴더)
- 📈 성능 그래프

### 2️⃣ 학습된 모델로 플레이 보기

학습된 가중치를 불러와 에이전트의 플레이를 확인합니다.

사전 학습이 완료된 MK3,4,5 가중치 파일
https://drive.google.com/drive/folders/1_iWydIyRkmOEnvTrQZ3GDtvRONfUd-gA?usp=drive_link

**설정 방법:**
1. `replay.py` 파일 열기
2. 체크포인트 경로 수정:
```python
   # replay.py 내부
   checkpoint_path = Path('checkpoints/mario_net_1600.chkpt')  # 원하는 체크포인트 경로
```

**실행:**
```bash
python replay.py
```

---

## 📁 프로젝트 구조
```
📦 Super-Mario-RL
├── 📂 MK3/                    # 기본 모델
│   ├── main.py
│   ├── replay.py
│   ├── agent.py
│   ├── neural.py
│   └── 📂 checkpoints/
│
├── 📂 MK4/                    # 개선 모델
│   ├── main.py
│   ├── replay.py
│   ├── agent.py
│   ├── neural.py
│   └── 📂 checkpoints/
│
├── 📂 MK5/                    # 최신 모델
│   ├── main.py
│   ├── replay.py
│   ├── agent.py
│   ├── neural.py
│   └── 📂 checkpoints/
│
├── 📂 tutorial/               # 학습 튜토리얼
│   ├── tutorial_v2.ipynb
│   └── tutorial_v2.py
│
├── environment.yml            # Conda 환경 설정
└── README.md
```

---

## 📊 성능 지표

학습 중 추적되는 주요 지표:
- 📈 Episode Reward
- 📉 Loss Value
- 🎯 Q-value 평균

---

</div>
