# EncoderModel_Task
## Masked Language Model 학습하기
`AutoModelForMaskedLM`를 사용하여 모델 학습  
main.py 실행

### 폴더 경로

```
├── config
│   └── train_config.yaml
├── data # 추가 필요
│   └── train.txt # 추가 필요
│   └── val.txt # 추가 필요
├── dataset.py
├── infer.py
├── main.py
├── model.py
├── train.py
```

### 실험 결과

![실험 결과](https://user-images.githubusercontent.com/77109972/203215222-d89c162b-f799-4114-805e-f9919b05d9ed.png)

```
원래 문장:다음은 상품에 대한 리뷰입니다. "이거 완전 좋은데?" 해당 리뷰의 감정은 [MASK]입니다.
'>>> 다음은 상품에 대한 리뷰입니다. "이거 완전 좋은데?" 해당 리뷰의 감정은 감동입니다.'
'>>> 다음은 상품에 대한 리뷰입니다. "이거 완전 좋은데?" 해당 리뷰의 감정은 공감입니다.'
'>>> 다음은 상품에 대한 리뷰입니다. "이거 완전 좋은데?" 해당 리뷰의 감정은 동감입니다.'
'>>> 다음은 상품에 대한 리뷰입니다. "이거 완전 좋은데?" 해당 리뷰의 감정은 순수입니다.'
'>>> 다음은 상품에 대한 리뷰입니다. "이거 완전 좋은데?" 해당 리뷰의 감정은 그대로입니다.'
```
