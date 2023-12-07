import pandas as pd
from Prosept_func import Prosept_func

def main():

    pd.options.mode.chained_assignment = None

    prosept = Prosept_func()

    url_dealerprice = 'http://prosept.sytes.net/api/dealerprice/'
    df_dealerprice = prosept.preprocess_dealerprice(url_dealerprice)

    url_product = 'http://prosept.sytes.net/api/product/'
    df_product = prosept.preprocess_product(url_product)

    df_dealerprice.product_name, df_product.name = df_dealerprice.product_name.apply(
        prosept.preprocess_text), df_product.name.apply(prosept.preprocess_text)

    vectors_feat, vectors_targ = prosept.vectorize(df_dealerprice.product_name, df_product.name)

    pred, pred_sim = prosept.prediction(df_dealerprice.product_name, vectors_feat, vectors_targ, 15)

    pred_id_key = prosept.get_id_key(pred, df_product)

    df_result, json_result = prosept.result_to_df_json(pred_id_key, pred_sim, df_dealerprice)

    prosept.save_json('http://prosept.sytes.net/api/recommendation/', json_result)

if __name__ == "__main__":
    main()
