import pandas as pd

STATES = ['CA', 'TX', 'WI']
D_PUBLIC = 1914


def compute_prices(df, prices):
    """
    Compute price-related features
    @param df: sales dataframe
    @param prices: price dataframe
    @return: dataframe with price-related features
    """

    # merge sales with prices
    df = pd.merge(df, prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    df.reset_index(inplace=True, drop=True)

    # price of the same product in the same store during previous and next week
    df['price_last_week'] = df.groupby('id')['sell_price'].shift().astype('float16')
    df['price_next_week'] = df.groupby('id')['sell_price'].shift(-1).astype('float16')

    # price of product belonging to the same department and store in the same week
    df['price_same_dept'] = df.groupby(['dept_id', 'store_id', 'wm_yr_wk'])['sell_price'].transform('mean')\
                              .astype('float16')
    df['price_same_dept'] = df['sell_price'] / df['price_same_dept']      

    # mean price and std
    df['price_mean'] = df.groupby('id')['sell_price'].transform('mean').astype('float16')
    df['price_std'] = df.groupby('id')['sell_price'].transform('std').astype('float16')

    # normalization
    df['price_last_week'] = 1 - df['sell_price'] / df['price_last_week']
    df['price_next_week'] = 1 - df['sell_price'] / df['price_next_week']
    df['price_mean'] = 1 - df['sell_price'] / df['price_mean']
    
    # in case of missing data, assume no change in prices
    df['price_last_week'].fillna(0, inplace=True)
    df['price_next_week'].fillna(0, inplace=True)
    
    return df
    
    
def compute_recursive(df, target='sales', mean_imputation=True):
    """
    Compute sales-related features using also sales included in the forecast horizon
    @param df: sales dataframe
    @param target: target column
    @param mean_imputation: True to replace nan values with mean values
    """

    # rolling mean on the same day of the week, over 6 weeks
    df['rolling_mean_dayofweek'] = df.groupby(['id', 'dayofweek'])[target] \
                                     .transform(lambda x: x.shift().rolling(6, min_periods=4).mean()) \
                                     .astype('float16')
    
    # rolling mean over 28 days, shifted by 7 days
    df['rolling_mean_7_28'] = df.groupby('id')[target] \
                                .transform(lambda x: x.shift(7).rolling(28).mean()) \
                                .astype('float16')

    # mean imputation for missing values
    if mean_imputation:
        avg = df.groupby(['id', 'dayofweek'])['rolling_mean_dayofweek'].transform('mean').astype('float16')
        df['rolling_mean_dayofweek'].fillna(avg)

        avg = df.groupby('id')['rolling_mean_7_28'].transform('mean').astype('float16')
        df[f'rolling_mean_7_28'].fillna(avg)

    return df 


def compute_non_recursive(df, target='sales', horizon=28):
    """
    Compute sales-related features using only days not included in the forecast horizon
    @param df: sales dataframe
    @param target: target column
    @param horizon: forecast horizon
    @return: dataframe with sales-related features
    """

    # rolling mean shifted by @horizon days to avoid leakage
    for r in [28, 90, 180, 365]:
        df[f'rolling_mean_{horizon}_{r}'] = df.groupby('id')[target] \
                                              .transform(lambda x: x.shift(horizon).rolling(r, min_periods=28).mean()) \
                                              .astype('float16')
        # mean imputation
        avg = df.groupby('id')[f'rolling_mean_{horizon}_{r}'].transform('mean').astype('float16')
        df[f'rolling_mean_{horizon}_{r}'].fillna(avg)

    df['rolling_mean_dayofmonth'] = df.groupby(['id', 'dayofmonth'])[target] \
                                      .transform(lambda x: x.shift().rolling(4, min_periods=1).mean()) \
                                      .astype('float16')

    return df
    
    
def compute_average(df, target='sales', d_max=D_PUBLIC):
    """
    Compute average sales.
    Data after day @d_max is excluded
    @param df: sales dataframe
    @param target: target column
    @param d_max: first day of validation data
    @return: dataframe with sales-related features
    """

    # mean on whole training data (except validation)
    # use a mask to exclude validation data
    mask = df['d'] >= d_max
    # replace validation data with nan
    tmp = df.loc[mask][target]  # copy validation data
    df.loc[mask, target] = None 

    # average sales per id, store, department and state
    df['avg_item_store'] = df.groupby('id')[target].transform('mean').astype('float16')
    df['std_item_store'] = df.groupby('id')[target].transform('std').astype('float16')

    df['avg_item_state'] = df.groupby(['item_id'])[target].transform('mean').astype('float16')
    df['avg_dept_store'] = df.groupby(['dept_id', 'store_id'])[target].transform('mean').astype('float16')
    df['avg_dept_state'] = df.groupby(['dept_id'])[target].transform('mean').astype('float16')    
    
    # average sales on snap, event days 
    df['avg_snap'] = df.groupby(['id', 'snap'])[target].transform('mean').astype('float16')
    df['avg_event'] = df.groupby(['id', 'event_name_1'])[target].transform('mean').astype('float16')
    
    # average sales in the same week of month
    #df['avg_dayofmonth'] = df.groupby(['id', 'dayofmonth'])[target].transform('mean').astype('float16')
    df['avg_weekofmonth'] = df.groupby(['id', 'weekofmonth'])[target].transform('mean').astype('float16')
    #df['avg_year'] = df.groupby(['id', 'year'])[target].transform('mean').astype('float16')
    
    # reinsert removed data                                   
    df.loc[mask, target] = tmp                                                                           

    # number of items of the same department in the same store on a given day 
    df['n_items'] = df.groupby(['d', 'store_id', 'dept_id'])[target].transform('size').astype('int16') 
        
    return df
    

def compute_and_save_products(df, state, dst='../processed'):
    """
    Save products info for a given state
    @param df: sales dataframe
    @param state: state
    @param dst: destination directory
    @return: None
    """

    print('Saving products...')

    df = df[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'release']].drop_duplicates()
    df.reset_index(inplace=True, drop=True)
    df.to_parquet(f'{dst}/m5_{state}_products.pqt')
    

def compute_and_save_prices(df, prices, state, dst='../processed'):
    """
    Compute and save prices-related features for a given state
    @param df: sales dataframe
    @param prices: prices dataframe
    @param state: state
    @param dst: destination directory
    @return: None
    """

    print('Computing prices...')
    
    # compute prices related features
    df = df[['id', 'store_id', 'dept_id', 'item_id', 'wm_yr_wk']].drop_duplicates()
    df.reset_index(inplace=True, drop=True)
    df = compute_prices(df, prices)

    df.drop(columns=['store_id', 'dept_id', 'item_id'], inplace=True)
    # cast 
    cols = df.select_dtypes('float16').columns.to_list()
    df[cols] = df[cols].astype('float32')
    # save
    df.to_parquet(f'{dst}/m5_{state}_prices.pqt')
    

def compute_and_save_sales(df, target='sales', dst='../processed', d_max=D_PUBLIC, min_year=2012):
    """
    Compute and save sales-related features for each store
    @param df: input sales dataframe
    @param target: target column
    @param dst: destination directory
    @param d_max: first day of validation data
    @param min_year: minimum year to consider
    @return: None
    """

    print('Computing sales...')
    
    df.drop(columns=['cat_id', 'state_id', 'cat_id', 'state_id', 'dayofyear',
                     'weekofyear', 'wm_yr_wk', 'event_type_1', 'release'],
            inplace=True)
  
    df[target] = df[target].astype('float32')
    
    df = compute_recursive(df, target)
    df = compute_non_recursive(df, target)
    
    # remove records before @min_year
    df = df.loc[df['year'] >= min_year]
    df = df.reset_index(drop=True)
    
    df = compute_average(df, target, d_max)

    df.drop(columns=['dayofweek', 'dayofmonth', 'weekofmonth', 'month', 'year',
                     'event_name_1', 'snap', 'item_id', 'dept_id'],
            inplace=True)
    
    # cast
    cols = df.select_dtypes('float16').columns.to_list()
    df[cols] = df[cols].astype('float32')

    stores = df['store_id'].unique()
    for store in stores:
        df[df['store_id'] == store].to_parquet(f'{dst}/m5_{store}_sales.pqt')

