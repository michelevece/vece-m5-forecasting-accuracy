import os
import pandas as pd
import zipfile
import m5_fe as fe
import lightgbm as lgbm


STATES = ['CA', 'WI', 'TX']


def extract_data(src='../m5-forecasting-accuracy.zip', dst='../extracted/'):
    """
    Extract data from zip file
    @param src: source to zip file
    @param dst: destination directory
    @return: None
    """
    if not os.path.exists(src):
        raise Exception('Source not found')

    if not os.path.exists(dst):
        os.makedirs(dst)

    with zipfile.ZipFile(src, 'r') as zip_ref:
        zip_ref.extractall(dst)

    print('data extracted')


def read_data_csv(src='../extracted'):
    """
    Read data in csv format
    @param src: source directory
    @return: calendar, prices and sales dataframe
    """
    calendar = pd.read_csv(f'{src}/calendar.csv')
    prices = pd.read_csv(f'{src}/sell_prices.csv')
    sales = pd.read_csv(f'{src}/sales_train_evaluation.csv')

    return calendar, prices, sales


def read_data_pqt(src='../processed'):
    """
    Read data in pqt format
    @param src: source directory
    @return: prices, calendar and sales dataframe
    """
    calendar = pd.read_parquet(f'{src}/m5_calendar.pqt')
    prices = pd.read_parquet(f'{src}/m5_prices.pqt')
    sales = pd.read_parquet(f'{src}/m5_sales.pqt')

    return calendar, prices, sales


def load_df(store, src='../processed'):
    """
    Load dataframe of a given store
    @param store: store
    @param src: source directory
    @return: sales dataframe of a given store
    """
    state = store[:2]
    df_calendar = pd.read_parquet(f'{src}/m5_calendar.pqt')
    df_products = pd.read_parquet(f'{src}/m5_{state}_products.pqt')
    df_prices = pd.read_parquet(f'{src}/m5_{state}_prices.pqt')
    df_sales = pd.read_parquet(f'{src}/m5_{store}_sales.pqt')

    # remove state since redundant (already contained in df_products)
    df_sales.drop(columns='store_id', inplace=True)

    # remove snap of other states
    cols = [f'snap_{s}' for s in STATES if s != state]
    df_calendar = df_calendar.drop(columns=cols)

    # merge sales, products, calendar and prices
    df = pd.merge(df_sales, df_products, on='id', how='left')
    df = pd.merge(df, df_calendar, left_on='d', right_index=True, how='left')
    df = pd.merge(df, df_prices, on=['id', 'wm_yr_wk'], how='left')
    df.reset_index(inplace=True, drop=True)

    # cast float32 to float16
    cols = df.select_dtypes('float32').columns.to_list()
    df[cols] = df[cols].astype('float16')

    return df


def predict(df, start, model, x_cols, target='sales', n_weeks=4):
    """
    Make a prediction for the next weeks
    @param df: sales dataframe
    @param start: first day of prediction
    @param model: model to use
    @param x_cols: input columns
    @param target: label column
    @param n_weeks: number of weeks to predict
    @return:
    """

    # forecast horizon
    horizon = n_weeks * 7

    # take a slice of the original dataframe
    df2 = df[(df['d'] >= start - horizon * 2) & (df['d'] < start + horizon)].copy()
    df2['predictions'] = df2[target].values

    # remove ground truth from forecast horizon
    df2.loc[df2['d'] >= start, 'predictions'] = None

    for curr_week in range(n_weeks):
        # if it is not the first week, use predicted values to compute features
        if curr_week:
            fe.compute_recursive(df2, target='predictions', mean_imputation=False)

        # records of the i-th week
        week_start = start + 7 * curr_week
        week_end = start + 7 * (curr_week + 1)
        curr_week_mask = (df2['d'] >= week_start) & (df2['d'] < week_end)

        # compute predictions for the current week
        x = df2.loc[curr_week_mask, x_cols]
        predictions = model.predict(x)
        df2.loc[curr_week_mask, 'predictions'] = predictions

    final_predictions = df2[(df2['d'] >= start) & (df2['d'] < start + horizon)]['predictions']

    return final_predictions


def save_model(model, filename, dst='../models'):
    """
    Save lgbm model
    @param model: lgbm model
    @param filename: filename of the model
    @param dst: destination directory
    @return: None
    """
    if not os.path.exists(dst):
        os.makedirs(dst)

    model.save_model(f'{dst}/{filename}')


def load_model(store, src='../models'):
    """
    Load a model of a given store
    @param store: the store whose model is needed
    @param src: source directory
    @return: the model of a given store
    """
    try:
        return lgbm.Booster(model_file=f'{src}/m5_{store}.txt')
    except:
        raise 'File not found'


def partial_submission(x, y, eval=False):
    """
    Create partial submission, namely related to a single store
    @param x: input dataframe
    @param y: predictions c
    @param eval: True if evaluation, False if validation
    @return: submission
    """
    submission = pd.concat([x[['id', 'd']], y], axis=1)
    submission = submission.pivot(index='id', columns='d', values='predictions')
    submission.reset_index(inplace=True, drop=False)

    # set columns names
    col_names = ['id']
    col_names.extend([f'F{i + 1}' for i in range(28)])
    submission.columns = col_names

    submission['id'] = submission['id'].apply(lambda id: id + ('_evaluation' if eval else '_validation'))

    return submission


def final_submission(src='../partial_submissions/', dst='../final_submission'):
    """
    Merge partial submissions in a single final submission
    @param src: source directory of partial submissions
    @param dst: destination directory of final submission
    @return: final submission
    """
    # read all partial submissions
    dfs = [pd.read_csv(src + f, index_col=False) for f in os.listdir(src) if f.endswith('.csv')]
    # concatenate them
    df = pd.concat(dfs)
    # write on disk
    df.to_csv(f'{dst}/m5_final_submission.csv', index=False)

    return df
