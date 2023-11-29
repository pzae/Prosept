import pandas as pd
import numpy as np

import sklearn
import scipy
import regex
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

import re

def preprocess(df_1, df_2):

    def split_compound_text(text):
        pattern = re.compile(r'(?<=[а-яА-Я])(?=[A-Za-z])|(?<=[A-Za-z])(?=[а-яА-Я])')
        splitted_text = pattern.sub(" ", str(text))
        return splitted_text

    df_1.product_name = df_1.product_name.apply(split_compound_text)
    df_2.name = df_2.name.apply(split_compound_text)

    def convert_milliliters(text):
        return re.sub(r'(\d+)\s*мл', lambda x: str(int(x.group(1)) / 1000) + ' ' + 'л', text)

    df_1.product_name = df_1.product_name.apply(lambda x: convert_milliliters(x))
    df_2.name = df_2.name.apply(lambda x: convert_milliliters(x))

    def clear_text(text):
        cleaned_text = re.sub(r'[!#$%&\'()*+,./:;<=>?@[\]^_`{|}~—\"\\-]+', ' ', text)
        return cleaned_text

    df_1.product_name = df_1.product_name.apply(clear_text)
    df_2.name = df_2.name.apply(clear_text)

    df_1.product_name = df_1.product_name.apply(lambda x: x.lower())
    df_2.name = df_2.name.apply(lambda x: x.lower())

    df_1.product_name = df_1.product_name.apply(
        lambda x: re.sub(r'(средство|мытья|для|чистящее|удаления|очистки)', r'\1 ', str(x)))
    df_2.name = df_2.name.apply(
        lambda x: re.sub(r'(средство|мытья|для|чистящее|удаления|очистки)', r'\1 ', str(x)))

    df_1.product_name = df_1.product_name.str.replace(r'\b(?:и|для|д|с|ф|п|ая|007|i)\b', '',
                                                                          regex=True)
    df_2.name = df_2.name.str.replace(r'\b(?:и|для|д|с|ф|п|ая|007|i)\b', '', regex=True)

    return df_1, df_2

def prediction(feat, targ, tf_idf, n):
    vectors_targ = tf_idf.transform(targ).tocsc()
    vectors_feat = tf_idf.transform(feat).tocsr()

    pred = np.zeros((len(feat), n), dtype=int)
    pred_sim = np.zeros((len(feat), n), dtype=float)

    for i in range(len(feat)):
        cos_sim = cosine_similarity(vectors_feat[i], vectors_targ)
        top_n_indexes = np.argsort(cos_sim)[0, -n:][::-1]
        top_n_values = cos_sim[0, top_n_indexes]

        pred[i] = top_n_indexes
        pred_sim[i] = list(np.around(np.array(top_n_values),6))

    return pred, pred_sim

def metric(actual, pred):
    count = 0
    for i in range(len(actual)):
        if actual[i] in pred[i]:
            count += 1
    return round(count / len(actual), 4)

def main():
    pd.options.mode.chained_assignment = None

    path_to_dir = '../data/'

    df_dealerprice = pd.read_csv(path_to_dir + 'marketing_dealerprice.csv', sep=';')
    df_dealerprice = df_dealerprice[['product_key', 'product_name']]
    df_dealerprice.drop_duplicates(inplace=True)

    df_product = pd.read_csv(path_to_dir + 'marketing_product.csv', sep=';')
    df_product = df_product[['id', 'name']]
    df_product.dropna(inplace=True)
    df_product.drop_duplicates(subset='name', inplace=True)
    df_product['id_unique'] = pd.factorize(df_product['name'])[0]
    df_product.set_index('id_unique', inplace=True)

    df_productdealerkey = pd.read_csv(path_to_dir + 'marketing_productdealerkey.csv', sep=';')

    df_dealerprice, df_product = preprocess(df_dealerprice, df_product)

    corpus = pd.concat([df_product.name, df_dealerprice.product_name]).drop_duplicates().values

    vectorizer = TfidfVectorizer()
    vectorizer_fited = vectorizer.fit(corpus)

    pred, pred_sim = prediction(df_dealerprice.product_name, df_product.name, vectorizer_fited, 5)

    result = []

    for row in pred:
        new_row = []
        for item in row:
            try:
                new_row.append(df_product.loc[item, 'id'].values[0])
            except:
                new_row.append(df_product.loc[item, 'id'])
        result.append(new_row)

    df_result = df_dealerprice['product_key']
    df_result = df_result.to_frame()
    df_result['product_id'] = result
    df_result['pred_sim'] = pred_sim.tolist()

    df_result.to_csv(r'my_data.csv', index=False)
    df_result.to_json(r'my_data', orient='index')

    df = df_result.merge(df_productdealerkey[['key', 'product_id']], left_on='product_key', right_on='key')

    print(metric(df.product_id_y, df.product_id_x))

if __name__ == "__main__":
    main()