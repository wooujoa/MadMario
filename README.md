🍄 Super Mario Bros RL Agent
강화학습(Reinforcement Learning)을 통해 슈퍼 마리오 브라더스(Super Mario Bros) 스테이지를 클리어하는 AI 에이전트 프로젝트입니다.

1. 환경 설정 (Installation)
이 프로젝트는 Conda 가상환경을 기반으로 실행됩니다. environment.yml 파일을 통해 필요한 패키지를 설치해 주세요.

Bash

# 레포지토리 클론 (필요 시)
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>

# Conda 환경 생성
conda env create -f environment.yml

# 가상환경 활성화
conda activate mario
2. 모델 버전 선택 (Select Model Version)
프로젝트는 실험 단계에 따라 MK3, MK4, MK5로 나뉘어 있습니다. 원하시는 모델 디렉토리로 이동하여 작업을 진행하세요.

📂 MK3 (Basic Agent)
초기 모델입니다. 기본적인 액션 스페이스와 리워드 구조를 테스트한 버전입니다.

Bash

cd MK3
📂 MK4 (Improved Agent)
MK3의 학습 결과를 바탕으로 하이퍼파라미터를 튜닝하고, 학습 안정성을 개선한 버전입니다.

Bash

cd MK4
📂 MK5 (Advanced Agent)
최신 모델입니다. 확장된 액션 스페이스(예: 달리기+점프)와 최적화된 네트워크 구조를 적용하여 더 복잡한 구간 돌파를 목표로 합니다.

Bash

cd MK5
3. 학습 진행 (Training)
새로운 학습을 시작하려면 해당 모델 디렉토리에서 아래 명령어를 실행하세요.

Bash

python main.py
학습 로그와 모델 체크포인트(checkpoint)는 지정된 checkpoints 폴더에 자동 저장됩니다.

4. 추론 및 시각화 (Inference / Replay)
학습이 완료된 가중치(checkpoint)를 로드하여 에이전트의 플레이를 직접 확인합니다.

replay.py 파일을 열어 checkpoint_path를 수정합니다.

Python

# replay.py 내부 수정 예시
checkpoint_path = Path('checkpoints/mario_net_1600.chkpt') # 불러올 파일 경로 지정
아래 명령어로 실행합니다.

Bash

python replay.py