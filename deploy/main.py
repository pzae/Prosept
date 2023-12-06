import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Prosept_func import Prosept_func

def main():

    pd.options.mode.chained_assignment = None

    prosept = Prosept_func()

    url_dealerprice = '127.0.0.1:8000/api/dealerprice/'
    df_dealerprice = prosept.preprocess_dealerprice(url_dealerprice)

    url_product = '127.0.0.1:8000/api/product/'
    df_product = prosept.preprocess_product(url_product)

    path_to_dir = 'data/'
    df_productdealerkey = pd.read_csv(path_to_dir + 'marketing_productdealerkey.csv', sep=';')

    df_dealerprice.product_name, df_product.name = df_dealerprice.product_name.apply(
        prosept.preprocess_text), df_product.name.apply(prosept.preprocess_text)

    corpus = pd.concat([df_product.name, df_dealerprice.product_name]).drop_duplicates().values
    vectorizer = TfidfVectorizer()
    vectorizer_fited = vectorizer.fit(corpus)

    pred, pred_sim = prosept.prediction(df_dealerprice.product_name, df_product.name, vectorizer_fited, 15)

    pred_id_key = prosept.get_id_key(pred, df_product)

    df_result, json_result = prosept.result_to_df_json(pred_id_key, pred_sim, df_dealerprice)

    prosept.save_json('http://prosept.sytes.net/api/recommendation/', json_result)

    df_metric = df_result.merge(df_productdealerkey[['key', 'product_id']], left_on='product_key', right_on='key')
    print(prosept.metric(df_metric.product_id_y, df_metric.product_id_x, 5))

if __name__ == "__main__":
    main()
