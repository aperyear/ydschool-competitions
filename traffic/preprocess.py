import os
import numpy as np
import pandas as pd
from datetime import datetime



def create_date_table(start, end, freq="60min", holiday=True):
    df = pd.DataFrame({"Date": pd.date_range(start, end, freq=freq)})
    df["Day"] = df.Date.dt.day
    df['Hour'] = df.Date.dt.hour
    df["DayNum"] = df.Date.dt.weekday
    h_list = ['2020-01-01', '2020-01-24', '2020-01-25', '2020-01-26', 
              '2020-01-27', '2020-04-15', '2020-04-30', '2020-05-01', '2020-05-05']
    if holiday:
        for h in h_list:
            df.loc[df.Date.apply(lambda x: str(x)[:10]) == h, ['Holiday']] = 1

    df = df.fillna(0)
    return df


def col_to_date(csv_path):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={'날짜': 'Date', '시간': 'Hour'})
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    return df


def merge_info(df, table):
    for i, col in table.iterrows():
        df.loc[(df.Date == str(col.Date.date())) & (df.Hour == col.Hour), ['Holiday', 'DayNum']] = col['Holiday'], col['DayNum']
    return df


def preprocess_dfs():
    if not os.path.exists(f"./data/final_train.csv"):
        y_cols = ['10','100','1000','101','1020','1040','1100','120','1200','121','140','150',
                '1510','160','200','201','251','2510','270','300','3000','301','351','352',
                '370','400','450','4510','500','550','5510','600','6000','650','652']

        train_df = col_to_date('./data/train.csv')
        valid_df = col_to_date('./data/validate.csv')

        total_df = pd.concat([train_df, valid_df], axis=0)
        total_df = total_df.drop_duplicates(subset=['Date', 'Hour']).reset_index().drop(['index'], axis=1)

        # 이상치 조정
        total_df.loc[total_df['10'] < 100, y_cols] = np.nan
        total_df = total_df.fillna(method='ffill')
        total_df["DayNum"] = total_df.Date.dt.weekday

        # 결측치 추가
        _feb = {'date': '2020-02-29', 'day_num': 5, 'hours': range(13, 24, 1)}
        _mar = {'date': '2020-03-30', 'day_num': 0, 'hours': range(2, 24, 1)}

        group_data = total_df.groupby(['DayNum', 'Hour']).apply(lambda x: np.mean(x))

        feb, mar = [], []
        for hour in _feb['hours']:
            temp_array = np.array([datetime.strptime(_feb['date'], '%Y-%m-%d')])
            temp_array = np.append(temp_array, group_data.loc[_feb['day_num']].loc[hour].values)
            feb.append(temp_array)

        for hour in _mar['hours']:
            temp_array = np.array([datetime.strptime(_mar['date'], '%Y-%m-%d')])
            temp_array = np.append(temp_array, group_data.loc[_mar['day_num']].loc[hour].values)
            mar.append(temp_array)

        feb_df = pd.DataFrame(feb, columns=total_df.columns)
        mar_df = pd.DataFrame(mar, columns=total_df.columns)

        total_df = pd.concat([total_df, feb_df, mar_df], axis=0)
        total_df['Date'] = pd.to_datetime(total_df['Date'])
        total_df = total_df.sort_values(['Date', 'Hour'], ascending=[True, True])
        total_df = total_df.reset_index().drop(['index'], axis=1)

        test_df = col_to_date('./data/test.csv')
        test_df = test_df[test_df['10'] == -999].reset_index().drop(['index'], axis=1)

        table = create_date_table(start='2020-01-01', end='2020-06-01', freq='60min', holiday=True, corona=False)

        final_train_df = merge_info(total_df, table)
        final_test_df = merge_info(test_df, table)

        final_train_df.to_csv('./data/final_train.csv')
        final_test_df.to_csv('./data/final_test.csv')
        print('------- data is ready ------- ')
    return


if __name__ == '__main__':
    preprocess_dfs()