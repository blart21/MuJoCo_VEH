# Capstone Design

0. 현재 진행중인 작업 
    - interface, perception, vehicle, utils 계층 분리 
    - interface : 차량 제어를 위한 입력 코드 작성 필요 
    - perception : lidar 모듈 정상 작동 확인 필요 
    - vehicle : 차량 주행 코드 이식 및 고도화 필요 
    - utils : 기존 뷰어 이식 필요 

1. 실행 방법
```bash
cd script
python -m main
```

2. 파일 구조 
```bash
root/
│
├── models/                 # XML 모델 관련
│   ├── obstacle/           # 장애물 정의의
│   ├── scene/              # 환경 정의
│   └── vehicle/            # 차량 정의
│
├── scripts/                 # 파이썬 코드
│   │
│   ├── main.py              # 실행 진입점
│   │
│   ├── interface/           # 입력 관련 모듈
│   │   ├── __init__.py
│   │   ├── input_manager.py
│   │   ├── keyboard.py
│   │   ├── joystick.py
│   │   └── wheel.py
│   │
│   ├── perception/          # 센서 모듈
│   │   ├── __init__.py
│   │   ├── control.py
│   │   ├── ebrake.py
│   │   └── lidar.py
│   │
│   ├── utils/               # 공용 유틸리티
│   │   ├── __init__.py
│   │   └── viewer.py
│   │
│   └── vehicle/             # Vehicle 환경 정의
│       ├── __init__.py
│       └── vehicleEnv.py
│
├── requirements.txt
│
└── README.md
```

3. 구현 완료 사안 

4. TEST CASE 및 요구사항 
- 개발 후 기능 정상 작동 여부 테스트 
