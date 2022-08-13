### 제4차 이어드림스쿨 모의경진대회 [AI CONNECT](https://www.aiconnect.kr/main/competition/list)
- 기간: 22. 2월 23일(수) ~ 3월 4일(금) 12:00PM
- 과제: 멀티 라벨 자연어 분류
- 목표: 뉴스 기사의 편견과 혐오 표현의 댓글 식별 
- 데이터: 뉴스 기사의 제목 및 댓글 (8367 rows)
- 평가지표: Macro F1 Score
- 팀구성: 4명

### 실험 과정
1. 3일간 각 팀원 데이터를 분석 및 모델링 결과 공유
2. 데이터 tokenize / transformer 학습 결과 스코어가 베이스라인 보다 낮음  
3. 대회에서 베이스라인은 Kc-ELECTRA fine-tunning으로 이미 높은 성능의 베이스라인이 주어짐
4. 뉴스와 댓글로 구성된 데이터는 위키피디아 등을 사전학습 model보다 댓글로 사전학습한 Kc-BERT/Kc-ELECTRA가 더 잘 분류할 것으로 예상
5. 다만, hugging face에서 한국어가 포함된 BERT 모델을 대부분 실험 결과 Kc-ELECTRA가 가장 성능이 높음
6. 결국 베이스라인과 같은 모델을 선택하고 fine-tunning 파라미터에 집중
7. input data로 '댓글'과 '뉴스 기사 제목' 두 가지가 있는데 여러 조합을 실험
8. 250번이 넘는 wandb를 찍으면서 팀원과 함께 개선 방안을 논의
9. 1~2번의 epoch로 overfitting에 가깝게 되는 문제를 해결하기 위해 다양한 실험
10. data augmentation 실험

### 실험 성과
- 댓글 + 제목을 함께 input으로 줄 때 성능이 가장 높음
- learning rate 1e-5 안정적
- scheduler가 없을 때 학습이 더 안정적
- 특정 step size에서 validation metrics 확인 및 모델 저장
- dropout 큰 차이 없음
- classifier head를 제외한 다른 layer를 freeze 큰 차이 없음
- 모델을 작게 하기 위해 중간 layer를 제거 후 성능이 낮아짐
- 마지막 hidden layer 4개를 concat 후 성능이 낮아짐
- AdamW의 weight decay도 유의미한 성능 향상이 없음
- 전처리 과정에서 특수 문자나 불필요해 보이는 문자, stop words 등을 제거하거나 바꾸는 작업은 효과가 없음
- comment와 title을 Kc-ELECTRA와 Ko-BERT/Kc-ELECTRA에 각각 input으로 주고 마지막 logits을 concat 하는 방법도 효과 없음
- layer마다 learning rate를 다르게 주는 학습 효과 없음
- augmentation 측면에서 input data를 50% 확률로 random하게 shuffle하는 방법 성능 향상.

### 대회 결과
- 1위로 마무리.
- F1 score: public 0.7768 final 0.7504

### 차별성
- bias와 hate를 나누어서 학습
- augmentation 성능 향상

### 개선 가능성
- 성능을 향상할 수 있는 전처리 방법을 찾아보고 적용해봤으나 성능이 오르지 않음
- 댓글 데이터의 특수성 때문에 그렇다면, 다음에는 가장 먼저 전처리 방법을 다시 고민해야함
