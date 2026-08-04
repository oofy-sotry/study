# coding: utf-8
"""학습된 NSMC 감성분석 모델을 서빙하는 간단한 Flask 웹 데모"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from flask import Flask, render_template, request

from data_utils import load_vocab, encode
from model import LSTMSentimentClassifier
from train import get_device, MAX_LEN

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')

app = Flask(__name__)

device = get_device()
vocab = load_vocab(os.path.join(CKPT_DIR, 'vocab.json'))
model = LSTMSentimentClassifier(vocab_size=len(vocab)).to(device)
model.load_state_dict(torch.load(os.path.join(CKPT_DIR, 'best_model.pt'), map_location=device))
model.eval()  # 추론 모드 - Dropout을 끄고 항상 같은 결과가 나오게 함 (6.4절에서 배운 학습/추론 모드 차이)
print(f'모델 로드 완료 (device={device}, vocab_size={len(vocab)})')


def predict(text):
    ids, length = encode(text, vocab, MAX_LEN)
    input_ids = torch.tensor([ids], dtype=torch.long).to(device)
    lengths = torch.tensor([length], dtype=torch.long).to(device)
    with torch.no_grad():  # 추론만 할 거라 기울기 계산 불필요 -> 메모리/속도 절약
        logit = model(input_ids, lengths)
        prob = torch.sigmoid(logit).item()
    label = '긍정' if prob >= 0.5 else '부정'
    confidence = prob if prob >= 0.5 else 1 - prob
    return label, confidence


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    review_text = ''
    if request.method == 'POST':
        review_text = request.form.get('review', '').strip()
        if review_text:
            label, confidence = predict(review_text)
            result = {'label': label, 'confidence': confidence}
    return render_template('index.html', result=result, review_text=review_text)


if __name__ == '__main__':
    app.run(debug=False, port=5001)
