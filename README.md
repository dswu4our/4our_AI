# 4our_AI

4학년 졸업 프로젝트
백엔드 flask 서버
1. 이미지 데이터 전처리: numpy를 이용해 64*64 사이즈로 압축
2. 모델 구성: tensorflow keras를 import 해 5개층의 relu 함수 적용
3. 모델 학습: cpu로 돌리기에는 시간이 오래걸려서 GPU 를 사용해야했기에 Flask를 이용해 서버를 분리 한 후 AWS EC2를 활용.
             테스트데이터와 훈련데이터를 3:7 정도로 나누어서 batch 사이즈를 조절해가면서 학습.
4. 예측하기: 정확도 약 80%의 15가지 식재료를 구분할수 있음.

![image](https://github.com/dswu4our/4our_AI/assets/43868373/38761ed1-d6ec-4a1f-82e3-2a02379023fa)
