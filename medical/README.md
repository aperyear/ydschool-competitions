### 제2차 이어드림스쿨 모의경진대회 [AI CONNECT](https://www.aiconnect.kr/main/competition/list)
- 기간: 22. 2월 9일(수) ~ 2월 15일(화) 12:00PM
- 과제: 흉부 CT 코로나 감염 여부 분류 모델
- 목표: 코로나19 감염 여부 이진 분류
- 데이터: 흉부 CT 이미지 (646 images)
- 평가지표: Accuracy
- 팀구성: 4명

### 실험 과정
1. 개인전과는 다르게 이번 대회는 3개의 GPU 서버와 4명의 팀전으로 진행됐다.
2. 대회 초반 3일은 각자의 방식대로 EDA와 학습을 진행하며 의견을 나누었다.
3. 흉부 CT 이미지를 같이 보면서 문제가 될 수 있는 요소와 해결법을 주로 정리했고,
4. 관련된 논문과 회의 아이디어를 융합하면서 학습 계획을 세웠다.
5. timm library를 사용해 CNN 모델을 불러와 실험했고,
6. 기본적으로 대회 dataset에 성능이 좋은 모델을 모아서 augmentation 실험을 진행했다.
7. 모델이 예측하는 부분을 Grad-CAM으로 분석했다.
8. 모든 과정은 wandb와 디스코드로 팀원과 공유했다.

### 실험 성과
- transformer보다 CNN 성능이 좋았다.
- 총 20개 이상의 모델을 실험했고, 그중 accuracy가 90% 이상 되는 모델을 선별했다.
- horizontal flip을 기본값으로, ShiftScaleRotate, RandomBrightnessContrast 3가지가 가장 큰 폭의 성능 향상이 있었고,
- 그 이후에 추가한 augmentation 실험에서는 성능에 큰 변화가 없었는데,  
- OpticalDistortion, GridDistortion, GaussNoise 이 3가지를 전부 포함한 총 6개의 augmentation 조합에서 성능이 향상했다.
- 성능이 낮은 모델과 높은 모델을 함께 앙상블 하면 성능이 하락했다.
- 비슷한 성능을 가진 모델 중 가장 cv 점수가 높은 모델 3가지 앙상블에서 public 점수 개선이 있었고, 최종 제출했다.

### 대회 결과
- 15팀 중 1위로 마무리했다.
- accuracy: public 0.9667 final 0.8429

### 차별성
- wandb로 각 팀원의 실험 과정을 모니터링할 수 있었고, 중복되는 실험을 방지하면서 의견을 나눌 수 있었다.
- 주어진 GPU 서버에서 자동화 코드로 24시간 실험을 했다.

### 개선 가능성
- 특정 dataset에서 적용할 수 있는 augmentation 실험을 더 많이하고,
- inference 단계에서 TTA를 사용해 볼 수 있을 것 같다.

### 결론
- 업무도 게임도 축구도 그렇고 팀전은 정말 매력적이다. 팀원 모두가 한가지 목표를 위해 몰입하고 의견을 나누고 성과를 내는 일은 언제나 감동적이고 즐거운 것 같다.
- public 마지막 끝에서 시작해서 꾸준히 회의하며 실험하고, 한칸 한칸 올라와 마지막 제출 일에 public과 final 1위로 대회를 마쳤다.
- 처음 해보는 AI team project에서 훌륭한 팀원분들과 함께 좋은 경험을 했다.