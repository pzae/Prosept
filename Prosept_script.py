import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

df_dealer = pd.read_csv('data/marketing_dealer.csv', sep=';')
df_dealerprice = pd.read_csv('data/marketing_dealerprice.csv', sep=';')
df_product = pd.read_csv('data/marketing_product.csv', sep=';')
df_productdealerkey = pd.read_csv('data/marketing_productdealerkey.csv', sep=';')

df = df_productdealerkey.merge(df_product, left_on='product_id', right_on='id')
df = df.merge(df_dealerprice, left_on='key', right_on='product_key')
df = df[['product_id', 'name', 'product_name', 'product_key']].copy()
df.drop_duplicates(inplace=True)

df['id'] = pd.factorize(df['name'])[0]
df.set_index('id', inplace=True)

df_targ_unique = df.name.drop_duplicates()

corpus = pd.concat([df.name, df.product_name]).drop_duplicates().values

vectorizer = TfidfVectorizer()
vectorizer_fited = vectorizer.fit(corpus)

def prediction(feat, targ, tf_idf, n):
    vectors_targ = tf_idf.transform(targ).tocsc()
    vectors_feat = tf_idf.transform(feat).tocsr()

    pred = np.zeros((len(feat), n), dtype=int)
    pred_proba = np.zeros((len(feat), n), dtype=float)

    for i in range(len(feat)):
        cos_sim = cosine_similarity(vectors_feat[i], vectors_targ)
        top_n_indexes = np.argsort(cos_sim)[0, -n:][::-1]
        top_n_values = cos_sim[0, top_n_indexes]

        pred[i] = top_n_indexes
        pred_proba[i] = top_n_values

    return pred, pred_proba

pred, pred_proba = prediction(df.product_name, df_targ_unique, vectorizer_fited, 5)

actual = list(map(int, df.index))

result = []

for row in pred:
    new_row = []
    for item in row:
        try:
            new_row.append(df.loc[item, 'product_id'].values[0])
        except:
            new_row.append(df.loc[item, 'product_id'])
    result.append(new_row)

df_result = df['product_key']
df_result = df_result.to_frame()
df_result['product_id'] = result

df_result.to_csv(r'my_data.csv', index=False)