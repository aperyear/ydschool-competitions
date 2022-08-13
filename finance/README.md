### 제1차 이어드림스쿨 모의경진대회 [AI CONNECT](https://www.aiconnect.kr/main/competition/list)
- 기간: 22. 1월 26일(수) ~ 2월 8일(화) 12:00PM
- 과제: 대출자 채무 불이행 예측
- 목표: 불이행 여부 이진 분류
- 데이터: 이자율 등 대부업체 고객 데이터 (100,000 rows)
- 평가지표: Macro F1 Score
- 팀구성: 개인전

### 실험 과정
1. 베이스라인에서 GBM을 사용한 모델링과 최적화 그리고 앙상블이 주어짐
2. 대회 초기 3일은 베이스라인을 따라가면서 feature engineering과 grid search에 주력
3. 이후 Deep learning 모델링을 시도 후 성공
4. 알맞은 network를 선택한 이후에는 data scaler, learning rate, batch size, optimizer, loss function, scheduler, weight decay 등의 실험 진행
5. 그 이후 데이터를 kfold 방식으로 나누어 충분한 5~20폴드까지 실험
6. 저장한 모델을 inference하면서 threshold를 주는 방식으로 F1 score를 최대화

### 실험 성과
- data scale 방법을 변경하면서 유의미한 성능 변화 확인
- batch size 학습 안정성 향상
- optimizer Adam이 가장 빠르고 더 낮은 loss로 수렴
- loss function은 크게 차이가 없음
- scheduler의 차이가 크지 않아서 최종적으로 사용 제외
- weight decay도 큰 효과 없음
- fold는 수가 증가하면서 일반화 성능 향상
- threshold 조정으로 최대 ~2% 점수 상승

### 대회 결과
- 1위
- F1 score: public 0.7237 final 0.7188

### 차별성
- 참여자 중 유일하게 deep learning 모델 사용
- 따라서 feature engineering 없이 모든 데이터를 그대로 사용
- cloud instance에 지속적인 실험과 학습을 할 수 있는 환경을 구축

### 개선 가능성
- 편향된 dataset으로 StratifiedKFold 사용

### 결론
- deep learning 모델은 tabular data에서 잘 작동하지 않는데 바닥부터 특이한 아이디어로 모델링한 결과가 좋았음.
