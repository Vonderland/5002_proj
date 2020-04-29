from tqdm import tqdm
import pandas as pd
from dateutil.parser import parse
import warnings

warnings.filterwarnings("ignore")
from datetime import datetime, timedelta


def get_interval_data(df, hours, interval, line):
    for hour in hours:
        for minute in [i * interval for i in range(60 // interval)]:
            hour_data = df[df['hour'] == hour]
            if len(hour_data) == 0:
                avg_travel_time = 0
            else:
                avg_travel_time = hour_data[hour_data['minute'] == minute]['avg_travel_time'].values
                if len(avg_travel_time) == 0:
                    avg_travel_time = 0
                else:
                    avg_travel_time = avg_travel_time[0]
            line += "," + str(avg_travel_time)
    return line


def preprocessing_train(interval=20):
    travel_time = pd.read_csv('data/20min_avg_travel_time_training_phase1.csv')
    travel_time['avg_travel_time'] = travel_time['avg_travel_time'].astype('float')
    travel_time.index = travel_time['time_window'].map(lambda x: parse(x.split(',')[0][1:]))
    travel_time['week'] = travel_time.index.map(lambda x: x.weekday)
    travel_time['month'] = travel_time.index.map(lambda x: x.month)
    travel_time['day'] = travel_time.index.map(lambda x: x.day)
    travel_time['hour'] = travel_time.index.map(lambda x: x.hour)
    travel_time['minute'] = travel_time.index.map(lambda x: x.minute)

    # 特征工程再统一去掉
    # travel_time.drop(
    #     pd.date_range(start=datetime(2016, 10, 1), end=datetime(2016, 10, 8), freq='10min', closed='left'),
    #     inplace=True, errors='ignore')

    selected = travel_time[['intersection_id', 'tollgate_id', 'month', 'week', 'day', 'hour', 'minute', 'avg_travel_time']]

    # 分别按照两小时来计算存到对应文件中
    path = 'data/train_6_8_' + str(interval) + '.csv'
    file_6_8 = open(path, "w")
    path = 'data/train_8_10_' + str(interval) + '.csv'
    file_8_10 = open(path, "w")
    path = 'data/train_15_17_' + str(interval) + '.csv'
    file_15_17 = open(path, "w")
    path = 'data/train_17_19_' + str(interval) + '.csv'
    file_17_19 = open(path, "w")

    intersection_ids = list(selected['intersection_id'].unique())
    tollgate_ids = list(selected['tollgate_id'].unique())

    start = datetime(2016, 7, 19)
    # 对时间、十字路口和收费站遍历，计算出每个[十字路口-收费站-月-日-周几-依次6个20分钟窗口内的平均通过时间]数据行
    # 91天到10.17
    for i in tqdm(range(91)):
        now = start + timedelta(days=i)
        month = now.month
        day = now.day
        week = now.weekday()
        day_df = selected[(selected['month'] == month) & (selected['day'] == day)]
        for intersection_id in intersection_ids:
            for tollgate_id in tollgate_ids:
                cur_df = day_df[(day_df['intersection_id'] == intersection_id) & (day_df['tollgate_id'] == tollgate_id)]
                if len(cur_df) == 0:
                    continue
                tmp = intersection_id + "," + str(tollgate_id) + "," + str(month) + "," + str(day) + "," + str(week)
                interval_data = get_interval_data(cur_df, [6, 7], interval, tmp)
                file_6_8.write(interval_data + "\n")

                interval_data = get_interval_data(cur_df, [8, 9], interval, tmp)
                file_8_10.write(interval_data + "\n")

                interval_data = get_interval_data(cur_df, [15, 16], interval, tmp)
                file_15_17.write(interval_data + "\n")

                interval_data = get_interval_data(cur_df, [17, 18], interval, tmp)
                file_17_19.write(interval_data + "\n")

    file_6_8.close()
    file_8_10.close()
    file_15_17.close()
    file_17_19.close()


def preprocessing_test(interval=20):
    travel_time = pd.read_csv('data/test_20min_avg_travel_time.csv')
    travel_time['avg_travel_time'] = travel_time['avg_travel_time'].astype('float')
    travel_time.index = travel_time['time_window'].map(lambda x: parse(x.split(',')[0][1:]))
    travel_time['week'] = travel_time.index.map(lambda x: x.weekday)
    travel_time['month'] = travel_time.index.map(lambda x: x.month)
    travel_time['day'] = travel_time.index.map(lambda x: x.day)
    travel_time['hour'] = travel_time.index.map(lambda x: x.hour)
    travel_time['minute'] = travel_time.index.map(lambda x: x.minute)

    selected = travel_time[
        ['intersection_id', 'tollgate_id', 'month', 'week', 'day', 'hour', 'minute', 'avg_travel_time']]

    path = 'data/test_6_8_' + str(interval) + '.csv'
    file_6_8 = open(path, "w")
    path = 'data/test_15_17_' + str(interval) + '.csv'
    file_15_17 = open(path, "w")

    intersection_ids = list(selected['intersection_id'].unique())
    tollgate_ids = list(selected['tollgate_id'].unique())

    start = datetime(2016, 10, 18)
    # 10.18到24号，7天
    for i in tqdm(range(7)):
        now = start + timedelta(days=i)
        month = now.month
        day = now.day
        week = now.weekday()
        day_df = selected[(selected['month'] == month) & (selected['day'] == day)]
        for intersection_id in intersection_ids:
            for tollgate_id in tollgate_ids:
                cur_df = day_df[(day_df['intersection_id'] == intersection_id) & (day_df['tollgate_id'] == tollgate_id)]
                if len(cur_df) == 0:
                    continue
                tmp = intersection_id + "," + str(tollgate_id) + "," + str(month) + "," + str(day) + "," + str(week)
                interval_data = get_interval_data(cur_df, [6, 7], interval, tmp)
                file_6_8.write(interval_data + "\n")

                interval_data = get_interval_data(cur_df, [15, 16], interval, tmp)
                file_15_17.write(interval_data + "\n")

    file_6_8.close()
    file_15_17.close()

def main():
    preprocessing_train(20)
    preprocessing_test(20)

'''
生成训练集和测试集
1.10-1到10-7号的数据还是先保留着，在特征工程里面统一处理
2.将数据分为上午和下午，并按照6-8、8-10等这样拆开
3.每一天都是一行，每一行包含了收费站、方向、日期等的基本信息，以及该时间段内的两个小时每20分钟的6个值
'''

if __name__ == '__main__':
    main()
