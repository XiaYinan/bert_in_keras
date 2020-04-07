import json
import numpy as np
import pandas as pd
from random import choice
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import re, os
import codecs

TF_KERAS=1

dict_path = './chinese_L-12_H-768_A-12/vocab.txt'
token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)

text = "酒店地图不准，通过携程打开地图后，会被带去心斋桥西边的美国街附近的一个酒吧，距离酒店5百多米，寒冷的夜里找不到酒店，给酒店打电话求助，前台态度相当冷漠：找不到酒店和我们没关系，自己想办法。wtf？"
x1, x2 = tokenizer.encode(first=text)

from keras.models import load_model
from keras_bert import get_custom_objects
model = load_model('best_model.h5',custom_objects=get_custom_objects())

y = model.predict([[x1],[x2]])
print(y)