import pandas as pd
import numpy as np
import re
import requests
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Prosept_func:

    def __init__(self):
        pass

    def preprocess_dealerprice(self, url_dealerprice):
        
        df = requests.get(url_dealerprice)
        df = pd.json_normalize(df.json())
        df = df[['product_key', 'product_name']]
        df.drop_duplicates(inplace=True)
        
        return df

    def preprocess_product(self, url_product):
        
        df = requests.get(url_product)
        df = pd.json_normalize(df.json())
        df = df[['id', 'name']]
        df.dropna(inplace=True)
        df.drop_duplicates(subset='name', inplace=True)
        df['id_unique'] = pd.factorize(df['name'])[0]
        df.set_index('id_unique', inplace=True)

        return df

    def preprocess_text(self, text):
        
        text = text.lower()
        text = re.compile(r'(?<=[а-яА-Я])(?=[A-Za-z])|(?<=[A-Za-z])(?=[а-яА-Я])').sub(" ", str(text))
        text = re.sub(r'(\d+)\s*мл', lambda x: str(int(x.group(1)) / 1000) + ' ' + 'л', text)
        text = re.sub(r'[!#$%&\'()*+,./:;<=>?@[\]^_`{|}~—\"\\-]+', ' ', text)
        text = re.sub(r'(средство|мытья|для|чистящее|удаления|очистки|плесени|добавка|prosept)', r'\1 ', str(text))
        text = re.sub(r'\b(?:и|для|д|с|ф|п|ая|007|i)\b', '', text)

        return text

    def vectorize(self, feat, targ):

        corpus = pd.concat([feat, targ]).drop_duplicates().values
        vectorizer = TfidfVectorizer()
        vectorizer_fited = vectorizer.fit(corpus)

        vectors_feat = vectorizer_fited.transform(feat).tocsr()
        vectors_targ = vectorizer_fited.transform(targ).tocsc()

        return vectors_feat, vectors_targ

    def prediction(self, feat, vectors_feat, vectors_targ, n):

        pred_sim = cosine_similarity(vectors_feat, vectors_targ)
        top_n_indices = np.argpartition(pred_sim, -n)[:, -n:]
        top_n_values = pred_sim[np.arange(len(feat))[:, None], top_n_indices]
        sorted_indices = np.argsort(top_n_values, axis=1)[:, ::-1]

        pred = top_n_indices[np.arange(len(feat))[:, None], sorted_indices]
        pred_sim = np.around(top_n_values[np.arange(len(feat))[:, None], sorted_indices], 6)

        return pred, pred_sim

    def get_id_key(self, pred, df_product):
        
        result = []
        product_id = dict(df_product['id'])

        for row in pred:
            new_row = [product_id.get(item, item) for item in row]
            result.append(new_row)

        return result

    def result_to_json(self, pred, pred_sim, df_dealerprice):

        df_result = df_dealerprice['product_key']
        df_result = df_result.to_frame()
        df_result['product_id'] = pred
        df_result['pred_sim'] = pred_sim.tolist()
        json_data = df_result.to_json(orient='records')
        json_result = json.loads(json_data)

        return json_result

    def save_json(self, url, data):
        
        if url:
            requests.post(url, json=data)
        else:
            with open('result.json', 'w') as file:
                json.dump(data, file)
