# Deep-Daiv

- 기간 : 2021.12.15 ~ 2022.03.24
- Personal Goal : 
  1. How to Explain Simply 
  2. Survive

# week 1 : OT
- False Positive (X)
- True Positivie (O)
- Goal: 10k
- Slack & Notion
- 10h per week

# week 2
- What is GA?
- LaTeX Writing
- Jekyll: Static Generator -> 컨텐츠를 html(static websites)로 변환시키기, github 내부에도 설치됨
  - markdown파일로 글 작성, git push로 업로드, Github pages 내부 Jekyll이 이를 인식한 후 html로 변환후 웹 호스팅  
- Static Website: 어떤 웹사이트 주소를 접속한다면 모든 사람들이 모든 결과물(html)이 동일

# week 3
- [paper] Character-level CNN
- https://arxiv.org/abs/1509.01626
- CNN=Vision이라는 편견 깨기 
- CNN을 NLP에 적용시키기 with code

# week 4
- [paper] Going Deeper with Convolutions
- https://arxiv.org/abs/1409.4842
- GoogleNet이라고도 불리는 모델 -> ILSVRC에서 SOTA 달성했지만, 레이어가 너무 깊어 공동 1위한 VGGNet에 좀 더 집중함.
- 하지만 그 다음 챌린지인 2015 ILSVRC에서 SOTA를 달성한 모델이 152개의 레이어를 갖는걸 보면, layer 22인 GoogleNet이 영향을 줬을 거라고 생각한다...!
- Deep Learning Architecture 자세히 샅샅이 뜯어봄! 
- 코드도 돌려봤지만, 이전 CNN 공부 방향과는 다르게 '코드 구현'을 집중적으로 하지 않고 Architecture에 집중하며, Parameter에 집중함.
- 1\*1 Conv layer
- Inception

# week 5
- Reinforcement Learning foundation
- not paper
- 통계학의 시선으로 강화학습 바라보기

# week 6
- [paper] Play Atari with deep reinforcement learning
- https://arxiv.org/abs/1312.5602
- RL with CNN
- Deep Q Network (DQN)
- 즉 컴퓨터가 실제로 이미지 데이터를 보면서 스스로 학습해 게임 실행 
- Experience Replay memory
- Input data: pixel -> CNN 사용



# week 7
- [paper] Human-level control through deep reinforcement learning
- https://www.nature.com/articles/nature14236
- Week 6에서 읽은 Atari 게임이랑 강화학습이 비슷한 느낌이어서 찾아보니 같은 모델 DQN이었다.
- 두 논문 모두 딥마인드에서 작성한 paper였고, 2013년의 DQN과 다른점은 다음과 같다. (DQN의 개정판이라고 생각했다)
- 2013년의 DQN은 target vlaue 를 상수취급하여 학습 진행했지만, 2015년의 DQNdms target value를 네트워크로 구성해 performance를 향상시킴
  - 이 과정에서 새로운 파라미터 theta^- 나타남
  - 이터레이션 C 시간동안 Q_hat과 Q를 동일하게 취급함



