### 제1차 이어드림스쿨 모의경진대회 [AI CONNECT](https://www.aiconnect.kr/main/competition/list)
- 기간: 22. 1월 26일(수) ~ 2월 8일(화) 12:00PM
- 과제: 대출자 채무 불이행 예측
- 목표: 불이행 여부 이진 분류
- 데이터: 이자율 등 대부업체 고객 데이터 (100,000 rows)
- 평가지표: Macro F1 Score
- 팀구성: 개인전

### 실험 과정
1. 베이스라인에서 GBM을 사용한 모델링과 최적화 그리고 앙상블이 주어졌다.
2. 대회 초기 2~3일은 베이스라인을 따라가면서 feature engineering과 grid search에 주력했으나 그 과정에서 성과가 크지 않았고, 무엇보다 LGBM 최적화가 눈으로 보이지 않았다.
3. 따라서 deep learning으로 처음부터 다시 모델링을 시도했고, 처음 테스트에서 학습 가능성을 찾았다.
4. 알맞은 network를 선택한 이후에는 data scaler, learning rate, batch size, optimizer, loss function, scheduler, weight decay 등의 실험을 진행했다.
5. 그 이후 데이터를 kfold 방식으로 나누어 충분한 epoch로 5~20폴드까지 실험했다.
6. 저장한 모델을 inference하면서 threshold를 주는 방식을 실험하고, F1 score를 최대화했다.

### 실험 성과
- data scale 방법을 변경하면서 유의미한 성능 변화가 있었다.
- batch size를 실험하면서 학습의 안정성이 향상됐다.
- optimizer를 실험하면서 Adam이 가장 빠르고 더 낮은 loss로 수렴했다.
- loss function은 크게 차이가 없었다.
- scheduler의 차이가 크지 않아서 최종적으로 사용하지 않았다.
- weight decay도 효과가 없었다.
- fold는 수가 많으면 많을수록 일반화 성능이 향상했다.
- threshold를 주는 방법은 최대 2% 정도 점수가 올랐다.

### 대회 결과
- 1위로 마무리했다.
- F1 score: public 0.7237 final 0.7188

### 차별성
- 참여자 중 유일하게 deep learning 모델을 사용했다.
- 따라서 어떤 feature engineering 없이 모든 데이터를 그대로 사용했다.
- F1 점수 최대화 방법론 실험을 많이 했다.
- cloud instance에 지속적인 실험과 학습을 할 수 있는 환경을 구축했다.

### 개선 가능성
- 편향된 dataset으로 StratifiedKFold를 사용하면 개선이 기대된다.

### 결론
- deep learning 모델은 tabular data에서 잘 작동하지 않는다고 한다.
- 이번에 모델을 만들어서 GBM보다 성능이 좋았던 것은 내가 이런 것에 대해 모르고 경험도 없었기 때문에 특이한 아이디어를 시도할 수 있었던 것 같다.