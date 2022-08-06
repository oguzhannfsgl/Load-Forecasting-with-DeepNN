import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


ramadan_n_sacrifice = ["2016-07-05", "2016-07-06", "2016-07-07", "2016-07-08",
                       "2016-07-08",  "2016-09-12", "2016-09-13", "2016-09-14", "2016-09-15",
                       "2017-06-25", "2017-06-26", "2017-06-27",
                       "2017-09-01", "2017-09-02", "2017-09-03", "2017-09-04",
                       "2018-06-15", "2018-06-16", "2018-06-17",
                       "2018-08-21", "2018-08-22", "2018-08-23", "2018-08-24",
                       "2019-06-03", "2019-06-04", "2019-06-05", "2019-06-06", "2019-06-07",
                       "2019-08-11", "2019-08-12", "2019-08-13", "2019-08-14",
                       "2020-05-24", "2020-05-25", "2020-05-26",
                       "2020-07-31", "2020-08-01", "2020-08-02", "2020-08-03",
                       "2021-05-13", "2021-05-14", "2021-05-15",
                       "2021-07-19", "2021-07-20", "2021-07-21", "2021-07-22", "2021-07-23"]


def read_dataset(df_path, data_from):
    energy_df = pd.read_csv(df_path)
    if data_from=="SPAIN":
        energy_df.rename(columns={"total load actual":"total_load_actual"}, inplace=True)
    return energy_df


def handle_missing_vals(energy_df):
    energy_df["total_load_actual"].fillna(method="backfill", inplace=True)
    energy_df[energy_df["total_load_actual"]==0] = np.nan
    energy_df["total_load_actual"] = energy_df["total_load_actual"].backfill()
    energy_df.loc[energy_df["time"].isnull()==True,"time"] = energy_df.iloc[2089]["time"]
    return energy_df

def extract_features(df, data_from, holidays):
    train_df = pd.DataFrame()
    train_df["total_load_actual"] = df["total_load_actual"]
    train_df["time"] = df["time"]
    train_df["day_of_week"] = [pd.Timestamp(t).day_of_week for t in train_df["time"].to_list()]
    train_df["day_of_year"] = [pd.Timestamp(t).day_of_year-1 for t in train_df["time"].to_list()]
    train_df["hour"] = [pd.Timestamp(t).hour for t in train_df["time"].to_list()]
    train_df["month_of_year"] = [int(t.split(" ")[0].split("-")[1])-1 for t in train_df["time"].to_list()]
    ###
    train_df["quarter"] = [pd.Timestamp(t).quarter for t in train_df["time"].to_list()]
    train_df["day_of_month"] = [pd.Timestamp(t).day for t in train_df["time"].to_list()]
    if data_from=="TURKEY":
        train_df["is_holiday"] = [int(t in holidays or t[:10] in ramadan_n_sacrifice) for t in train_df["time"].to_list()]
    elif data_from=="SPAIN":
        train_df["is_holiday"] = [int(t in holidays) for t in train_df["time"].to_list()]
    train_df["is_holiday"] = [int(train_df["is_holiday"].iloc[i] or ((train_df["day_of_week"].iloc[i]%5==0 or train_df["day_of_week"].iloc[i]%6==0) and train_df["day_of_week"].iloc[i]!=0)) for i in range(len(train_df))]
    train_df["mean_last_3"] = [(train_df["total_load_actual"].iloc[i-24]+train_df["total_load_actual"].iloc[i-48]+train_df["total_load_actual"].iloc[i-72])/3 if i>72 else 0 for i in range(len(train_df["total_load_actual"]))]
    train_df["holiday_to_work"] = [int((train_df["is_holiday"].iloc[i-1]-train_df["is_holiday"].iloc[i])==1) if i>0 else 0 for i in range(len(train_df))]
    return train_df


def normalize_df(train_df, is_custom):
    load_min = train_df["total_load_actual"].min()
    load_max = train_df["total_load_actual"].max()
    if is_custom:
        train_df["total_load_actual"] = (train_df["total_load_actual"]-load_min)/(load_max-load_min)
    else:
        base_features = ["total_load_actual", "month_of_year", "day_of_week", "hour",
                         "is_holiday", "mean_last_3", "holiday_to_work"]
        scaler = MinMaxScaler()
        train_df[base_features] = scaler.fit_transform(train_df[base_features])
    return train_df, load_min, load_max


def split_df(train_df, data_from="SPAIN", input_n=24, output_n=24):
    input_n, output_n = 24, 24
    non_seq_input_n = 14

    # Shifting load values by output_n times forward to use past time load values as features.
    train_df["target"] = train_df["total_load_actual"]
    train_df["total_load_previous"] = train_df["target"].shift(output_n)
    train_df["total_load_1_week"] = train_df["target"].shift(output_n*7)


    if data_from=="TURKEY":
        # Using the data until 2018 for training. Rest will be held for validation
        test_df = train_df[train_df["time"]>"2019"].copy()
        test_df = test_df[test_df["time"]<"2022"].copy()
        train_df = train_df[train_df["time"]<"2019"].iloc[output_n*non_seq_input_n:]
    elif data_from=="SPAIN":
        test_df = train_df[train_df["time"]>"2018"].copy()
        train_df = train_df[train_df["time"]<"2018"].iloc[output_n*non_seq_input_n:]
    return train_df, test_df


def get_np_dataset(train_df, test_df, is_custom=True):
    if not is_custom:
        base_features = ["total_load_previous", "month_of_year", "day_of_week", "hour",
                         "total_load_1_week", "is_holiday", "mean_last_3", "holiday_to_work"]
    else:
        base_features = ["total_load_previous", "day_of_week", "day_of_year", "hour",
                 "day_of_month", "is_holiday", "total_load_1_week"]

    train_x = train_df[base_features].to_numpy()
    train_y = train_df[["target"]].to_numpy()

    test_x = test_df[base_features].to_numpy()
    test_y = test_df[["target"]].to_numpy()
    return train_x, train_y, test_x, test_y


def get_tf_dataset(x, y, input_n=24, output_n=24, batch_size=16):
    def fix_output_length(features, targets):
        targets = targets[:output_n]
        return features, targets
    
    
    features = tf.data.Dataset.from_tensor_slices(x)
    features = features.window(input_n, shift=1, stride=1, drop_remainder=True).flat_map(lambda x: x.batch(input_n))
    targets = tf.data.Dataset.from_tensor_slices(y)
    targets = targets.window(input_n, shift=1, stride=1, drop_remainder=True).flat_map(lambda x: x.batch(input_n))
    dataset = tf.data.Dataset.zip((features, targets)).shuffle(1024*30)
    dataset = dataset.map(fix_output_length, tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def get_pd_dataset(df, data_from, holidays, is_custom):
    energy_df = read_dataset(df, data_from)
    energy_df = handle_missing_vals(energy_df)
    train_df = extract_features(energy_df, data_from, holidays)
    train_df, load_min, load_max = normalize_df(train_df, is_custom)
    train_df, test_df = split_df(train_df, data_from)
    return train_df, test_df, [load_min, load_max]




