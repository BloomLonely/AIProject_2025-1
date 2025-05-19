# AIProject_2025-1

## Matryoshka Representation Learning
MRL에 대한 것은 Google Docs를 참고해주세요!

다음 이미지는 mrl.ipynb의 결과로 생성된 fine-tuned model에 대한 정보입니다.
용량이 너무 커서 remote unconnected가 뜨네요.. mrl.ipynb 다 실행하면 생성되긴 합니다.

![image](https://github.com/user-attachments/assets/1de7607d-99d5-4ca5-8c3e-3a665f010fb0)

requirements.txt 만으로는 라이브러리 해결이 다 안됩니다.
1. jupyter 설치해야 합니다.
2. sentence-transformer가 3.x 이상이어야 SentenceTransformerTrainingArguments, SentenceTransformerTraininer를 사용할 수 있습니다. 최신 버전으로 설치해주세요.
3. datasets 설치해야 합니다.

detection.py는 클래스로 정의된 모델에 대한 inference고, detection2.py는 지정된 경로에서 모델을 불러옵니다.
