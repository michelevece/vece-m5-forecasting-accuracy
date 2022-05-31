import pandas as pd
import numpy as np

TARGET = 'sales'
STATES = ['CA', 'WI', 'TX']


def remove_leading_zeros(sales, calendar, target=TARGET):
    """
    Melt sales and remove sales before the release of each product since they are not real
    @param sales: sales dataframe
    @param calendar: calendar dataframe
    @param target: target column
    @return: dataframe with containing only true sales
    """

    # categorical columns and numeric columns
    cat_cols = sales.columns.tolist()[:6]
    num_cols = sales.columns.tolist()[6:]
    
    # melt
    df = sales.melt(id_vars=cat_cols, var_name='d', value_name=target)
    df['d'] = df['d'].astype('int16')

    # merge with calendar
    df = pd.merge(df, calendar, left_on='d', right_index=True, how='left')

    # get index of the release day (= first day in which a product is sold)
    release = (sales[num_cols] > 0).idxmax(axis='columns').astype('int16').rename('release')
    release.index = sales['id']

    # add release as last column 
    df = df.merge(release, left_on='id', how='left', right_index=True)
    
    # delete leading zeros
    # keep only sales after the release
    df = df[df['d'] >= df['release']]
    df.drop(columns='release')
    df.reset_index(inplace=True, drop=True)

    # keep only the snap column relative to the current state
    df['snap'] = np.int8(0)
    
    for state in df['state_id'].unique():
        idx = df[df['state_id'] == state].index
        df.loc[idx, 'snap'] = df.loc[idx, f'snap_{state}']
    
    cols = [f'snap_{state}' for state in STATES]
    df.drop(columns=cols, inplace=True)    
    
    return df


def add_days(sales, d_start=1942, d_end=1969):
    """
    add days of the private forecast horizon
    @param sales: sales dataframe
    @param d_start: first day of private data
    @param d_end: last day of private data
    @return: sales dataframe with added days
    """

    # insert day columns
    cols = sales.columns.tolist()
    cols.extend(np.arange(d_start, d_end + 1).astype('str'))
    sales = sales.reindex(columns=cols)

    # cast nan columns
    cols = sales.select_dtypes('float').columns.tolist()
    sales[cols] = sales[cols].astype('float32')

    return sales

