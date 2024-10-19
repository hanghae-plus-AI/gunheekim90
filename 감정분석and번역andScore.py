# HuggingFace 라이브러리 설치
!pip install transformers datasets

# 필요한 라이브러리 임포트
from datasets import load_dataset    # IMDb 데이터셋 로드를 위한 함수 임포트
from transformers import AutoTokenizer, AutoModelForSequenceClassification, MarianMTModel, MarianTokenizer
import torch

# IMDb 데이터셋 로드
# IMDb 데이터셋을 'imdb'라는 이름으로 불러옵니다.
imdb = load_dataset("imdb")  # IMDb 데이터셋을 로드하는 함수와 'imdb' 키워드 입력

# 감정 분석을 위한 모델과 토크나이저 로드
# HuggingFace에서 제공하는 사전 학습된 모델을 로드하여 감정 분석을 수행하세요.
sentiment_tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")  # 감정 분석을 위한 사전 학습된 모델명 입력
sentiment_model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")  # 위와 동일한 모델명 입력

# 번역 모델과 토크나이저 로드 (영어 -> 프랑스어)
# HuggingFace에서 제공하는 모델을 사용하여 번역을 수행하세요.
translation_model_name = "Helsinki-NLP/opus-mt-en-fr"
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)  # 번역 모델명 입력
translation_model = MarianMTModel.from_pretrained(translation_model_name)  # 번역 모델명 입력

# 텍스트 분류 함수 (감정 분석)
def classify_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = sentiment_model(**inputs)
    
    # 예측된 값 (1에서 5까지)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    
    # 별 개수 출력 (prediction 값에 1을 더한 값으로 출력)
    stars = "★" * (prediction + 1)
    
    print(f"Sentiment Score: {prediction + 1}")
    print(f"Rating: {stars}")
    
    return stars


# 번역 함수 (영어 -> 프랑스어)
def translate_to_french(text):
    inputs = translation_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = translation_model.generate(**inputs)
    translated_text = translation_tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# 파이프라인 함수: 감정 분석 -> 번역
def sentiment_and_translation_pipeline(text):
    # 감정 분석 수행
    stars = classify_sentiment(text)

    # 번역 수행
    translated_text = translate_to_french(text)
    print(f"Translated text: {translated_text}")
    
    return stars, translated_text

# 테스트 문장
test_text = "_________"

# 파이프라인 실행
stars, translated_text = sentiment_and_translation_pipeline(test_text)