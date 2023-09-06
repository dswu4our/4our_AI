# 4OUR_AI
# [4OUR팀 플라스크 서버]
1. 프로젝트 제목: 냉장고를 부탁해 - 딥러닝을 이용한 레시피 추천 및 냉장고 관리 애플리케이션

2. Demo 링크: https://www.youtube.com/watch?v=w-Q_QPlQyB0

3. 제작 기간 & 참여 인원 : 약 6개월, 프론트 2명(김유경,양세민), 백엔드 및 AI 개발 2명(유지원,황혜원)
 
4. 사용한 기술 (기술 스택): Node.js / MongoDB/ Python / AWS EC2 / AWS S3 / putty
5. 식재료 인식
   
![9_28발표ppt](https://github.com/dswu4our/4our_AI/assets/43868373/8640e88b-974b-4631-8c9d-71610b6e13ae)
   1. 이미지 데이터 전처리: numpy를 이용해 64*64 사이즈로 압축
   2. 모델 구성: tensorflow keras를 import 해 5개층의 relu 함수 적용
   3. 모델 학습: cpu로 돌리기에는 시간이 오래걸려서 GPU 를 사용해야했기에 Flask를 이용해 서버를 분리 한 후 AWS EC2를 활용해서 백엔드서버와 연동. 테스트데이터와 훈련데이터를 3:7 정도로 나누어서 batch 사이즈를 조절해가면서 학습.
   4. 예측하기: 랜덤으로 테스트 사진들을 정해 정확도 약 80%의 15가지 식재료를 구분할수 있음.
   5. 데이터 전달: 백엔드 서버로 인식한 데이터를 json 형식으로 전달.
  
![image](https://github.com/dswu4our/4our_AI/assets/43868373/38761ed1-d6ec-4a1f-82e3-2a02379023fa)

  > 본인이 맡은 역할: 인공지능 수정 및 학습 / 식재료 이미지 데이터 베이스 관리 / 플라스크와 백엔드 서버 연동.
