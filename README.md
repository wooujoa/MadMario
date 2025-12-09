# 🍄 Super Mario Bros RL Agent

강화학습(Reinforcement Learning)을 통해  
**슈퍼 마리오 브라더스(Super Mario Bros)** 스테이지를 클리어하는 AI 에이전트 프로젝트입니다.

MK3 → MK4 → MK5로 갈수록 모델 구조와 학습 안정성이 개선된 실험 버전을 포함하고 있습니다.

---

## 1. 환경 설정 (Installation)

이 프로젝트는 **Conda 가상환경**을 기반으로 실행됩니다.  
`environment.yml` 파일을 통해 필요한 패키지를 설치해 주세요.

```bash
# 레포지토리 클론 (필요 시)
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>

# Conda 환경 생성
conda env create -f environment.yml

# 가상환경 활성화
conda activate mario_env

