### 제4차 이어드림스쿨 모의경진대회 [AI CONNECT](https://www.aiconnect.kr/main/competition/list)
- 기간: 22. 2월 23일(수) ~ 3월 4일(금) 12:00PM
- 과제: 멀티 라벨 자연어 분류
- 목표: 뉴스 기사의 편견과 혐오 표현의 댓글 식별 
- 데이터: 뉴스 기사의 제목 및 댓글 (8367 rows)
- 평가지표: Macro F1 Score
- 팀구성: 4명

### 실험 과정
1. 언제나처럼 처음 3일은 팀원 각자 데이터를 분석하고 모델을 만들어 보고 의견을 공유했다.
2. 나는 데이터를 처음부터 tokenize를 하고 transformer로 학습시켜봤지만, 점수가 베이스라인보다 낮았다.  
3. 이번 대회는 이미 높은 성능을 낼 수 있는 베이스라인이 주어졌다. Kc-ELECTRA를 사용해 fine-tunning 했다.
4. data가 뉴스와 댓글이기 때문에 위키피디아나 사전 등을 학습한 pre-trained model보다 댓글로 사전학습시킨 Kc-BERT/Kc-ELECTRA가 가장 성능이 좋을 것으로 예상됐다.
5. 우리 팀은 hugging face에서 한국어가 포함된 BERT 모델을 대부분 실험해봤고, 예상대로 이번 task에서는 Kc-ELECTRA가 가장 성능이 좋았다.
6. 그래서 베이스라인과 같은 모델을 선택하고 hyper parameter tunning에 집중했다.
7. input data로 '댓글'과 '뉴스 기사 제목' 두 가지가 있는데 여러 조합을 실험했다.
8. 공식적으로 250번이 넘는 wandb를 찍으면서 결과를 제출하고 팀원과 같이 피드백을 했다.
9. 단 1~2번의 epoch로 overfitting에 가깝게 되는 문제를 해결하기 위해 다양한 실험을 했다.
10. data augmentation을 실험했다.

### 실험 성과
- 댓글 + 제목을 함께 input으로 줄 때 성능이 가장 높았다.
- learning rate는 1e-5에서 학습이 됐다.
- scheduler가 없을 때 학습 결과가 안정적이었다.
- 특정 step size에서 validation metrics를 확인하고 가장 성능이 좋은 모델을 저장해 cv를 높일 수 있었다.
- 같은 환경에서 실험한 bias(3가지 예측)와 hate(2가지 예측)의 결과 차이가 분명히 있었다.
- 한 팀원이 학습한 모델의 bias 성능은 다른 팀원의 모델보다 낮았지만, hate는 높았다.
- dropout은 큰 차이가 없었다.
- classifier head를 제외한 다른 layers를 freeze 하는 실험도 오히려 성능이 낮아지거나 비슷했다.
- 모델을 작게 하기 위해 중간 layer를 제거하는 실험도 성능이 낮아졌다.
- 마지막 hidden layer 4개를 concat 하는 방법도 성능이 낮아졌다.
- AdamW의 weight decay도 유의미한 성능 향상이 없었다.
- 전처리 과정에서 특수 문자나 불필요해 보이는 문자 stop words 등을 제거나 바꾸는 작업은 효과가 없었다.
- comment와 title을 Kc-ELECTRA와 Ko-BERT/Kc-ELECTRA에 각각 input으로 주고 마지막 두 logits을 concat 하는 방법도 효과가 없었다.
- layer마다 learning rate를 다르게 주는 방법도 효과가 없었다.
- 모델의 bias accuracy와 F1의 차이가 8% 이상 차이가 나기 때문에,
- 모델의 logit 값에 threshold [0.9, 1, 0.8] 등으로 조정해 주는 방법이 F1 점수를 조금 향상시켰지만, 일반화에 부작용이 우려된다.
- 마지막으로 augmentation 측면에서 input data를 50% 확률로 random하게 shuffle하는 방법에서 성능이 향상됐다.

### 대회 결과
- 15팀 중 1위로 마무리했다.
- F1 score: public 0.7768 final 0.7504

### 차별성
- bias와 hate를 둘로 나누어서 따로 학습시켰다.
- augmentation으로 성능 향상이 있었다.

### 개선 가능성
- 성능을 향상할 수 있는 전처리 방법을 찾아보고 적용해봤으나 성능이 오르지 않았다.
- 댓글 데이터의 특수성 때문에 그렇다면, 다음에는 가장 먼저 전처리 방법을 다시 고민해볼 수 있을 것 같다.

### 결론
- 이번 자연어 BERT task에서 token은 중요한 의미를 가지는 것 같다. 
- 사람으로 치자면 pre-train에 사용된 token이 다르면 다른 언어를 사용하는 사람과 같다.
- 그래서 맞춤법, 띄어쓰기 등 전처리를 강하게 했던 모델에서 오히려 더 낮은 점수가 나온 것 같다.
- 이미지 대회도 그렇지만 AI의 성능을 감히 예단할 수 없는 것 같다. 굉장히 예측을 잘한다.
- 이번 대회는 이미지에서 사용했던 augmentation을 자연어에도 적용해보는 등 많은 실험을 했다.
- 이렇게 모든 대회를 마치게 되었다. 너무 집중해서 잠도 잘 안 왔던 것 같다.
- 많이 배웠지만, 정말 나는 아무것도 모르고 갈 길이 멀다는 것을 다시 한번 느꼈다.