import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.preprocessing import OneHotEncoder, StandardScaler
OHE = OneHotEncoder(sparse=False)
pd.set_option('display.max_rows', 30)

# 月日如果用起来要one-hot，之后产生的feature就太多了而且感觉没必要，可能考虑is_special更合适
# 但是前后依赖这里怎么办呢？
# train[['month','day']] = train.date.str.split('-', expand=True)
def get_x_y(df):
    data = df.drop(columns=['Unnamed: 0', 'route', 'calculated_day', 'date'])
    ohe = OHE.fit_transform(data[['hr', 'weekday']])
    ohe_df = pd.DataFrame(ohe, columns= OHE.get_feature_names(input_features=['hr', 'weekday']))
    data = pd.concat([data, ohe_df],axis = 1)
    data = data.drop(columns=['hr', 'weekday'])
    x = data.drop(columns=['label'])
    y = data.label
    return x, y

def mape_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def scoring(reg, x, y):
    pred = reg.predict(x)
    return -mape_error(pred, y)


#Read the Real Value at Each Time Window
import math
from datetime import datetime,timedelta
import numpy as np
import pandas as pd
names=locals()

link_time={}
time_train={}
time_predict={}
time_check={}
test_time_train={}
test_time_check={}
weathers={}
rainingTotalTime={}

file_path='/Users/vayne/Desktop/DM_Project_0522due/dataSet_phase2/table5.csv'
# Step 1: Load trajectories
fr = open(file_path, 'r')
fr.readline()  # skip the header
traj_data = fr.readlines()
fr.close()

# 建立各Link時間的dictionary
for i in range(24):
    link_time[str(i+100)]={}
    # Step 2: Create a dictionary to store travel time for each route per time window
    travel_times = {}  # key: route_id. Value is also a dictionary of which key is the start time for the time window and value is a list of travel times
    for i in range(len(traj_data)):
        each_traj = traj_data[i].replace('"', '').split(',')
        intersection_id = each_traj[0]
        tollgate_id = each_traj[1]

        route_id = intersection_id + '_' + tollgate_id
        if route_id not in travel_times.keys():
            travel_times[route_id] = {}

        trace_start_time = each_traj[3]
        travel_seq = each_traj[4]
        trace_start_time = datetime.strptime(trace_start_time, "%Y-%m-%d %H:%M:%S")
        time_window_minute = math.floor(trace_start_time.minute / 20) * 20
        start_time_window = datetime(trace_start_time.year, trace_start_time.month, trace_start_time.day,
                                     trace_start_time.hour, time_window_minute, 0)
        time=start_time_window
        tt = float(each_traj[-1])
        # 國慶不管，因道路狀態不一樣
        if(time.month==10 and time.day in [1,2,3,4,5,6,7]):
            continue
        # 中秋節不管，因道路狀態不一樣
        if(time.month==9 and time.day in [15,16,17]):
            continue
        if start_time_window not in travel_times[route_id].keys():
            travel_times[route_id][start_time_window] = [tt]
        else:
            travel_times[route_id][start_time_window].append(tt)

real_value=[]
real_value_dict={}
#fw = open('/Users/vayne/Desktop/dm_pro_engin/real_value.csv', 'w')
#fw.writelines(','.join(['"intersection_id"', '"tollgate_id"', '"time_window"', '"avg_travel_time"']) + '\n')
for route in travel_times.keys():
    if route not in real_value_dict.keys():
        real_value_dict[route] = {}
    route_time_windows = list(travel_times[route].keys())
    route_time_windows.sort()
    for time_window_start in route_time_windows:
        time_window_end = time_window_start + timedelta(minutes=20)
        tt_set = travel_times[route][time_window_start]
        avg_tt = round(sum(tt_set) / float(len(tt_set)), 2)
        out_line =[str(route),time_window_start, str(avg_tt)]
        real_value.append(out_line)    
        
        
real_value_needed=[]
for i in real_value:
    if i[1].hour in [8,9,17,18]:
        real_value_needed.append(i)
        
#convert to dataframe
feature=['route','time','label']
real_value_df=pd.DataFrame(real_value_needed,columns=feature)


#C_3
#find the travel time at each time window
file_path='/Users/vayne/Desktop/dm_pro_engin/phase1_training/table5.csv'

# Step 1: Load trajectories
fr = open(file_path, 'r')
fr.readline()  # skip the header
traj_data = fr.readlines()
fr.close()

# 建立各Link時間的dictionary
for i in range(24):
    link_time[str(i+100)]={}
# Step 2: Create a dictionary to store travel time for each route per time window
travel_times = {}  # key: route_id. Value is also a dictionary of which key is the start time for the time window and value is a list of travel times
for i in range(len(traj_data)):
    each_traj = traj_data[i].replace('"', '').split(',')
    intersection_id = each_traj[0]
    tollgate_id = each_traj[1]

    route_id = intersection_id + '_' + tollgate_id
    if route_id not in travel_times.keys():
        travel_times[route_id] = {}

    trace_start_time = each_traj[3]
    travel_seq = each_traj[4]
    trace_start_time = datetime.strptime(trace_start_time, "%Y-%m-%d %H:%M:%S")
    time_window_minute = math.floor(trace_start_time.minute / 20) * 20
    start_time_window = datetime(trace_start_time.year, trace_start_time.month, trace_start_time.day,
                                     trace_start_time.hour, time_window_minute, 0)
    time=start_time_window
    tt = float(each_traj[-1])
    # 國慶不管，因道路狀態不一樣
    if(time.month==10 and time.day in [1,2,3,4,5,6,7]):
        continue
    # 中秋節不管，因道路狀態不一樣
    if(time.month==9 and time.day in [15,16,17]):
        continue
    if start_time_window not in travel_times[route_id].keys():
        travel_times[route_id][start_time_window] = [tt]
    else:
        travel_times[route_id][start_time_window].append(tt)
    
C_3_travel_times=travel_times['C_3']
C_3_travel_times_need={}
for i in C_3_travel_times:
    if i.hour in [7,8,9,16,17,18]:
        if i not in C_3_travel_times_need:
            C_3_travel_times_need[i]=[]
        if i in C_3_travel_times_need:
            C_3_travel_times_need[i].extend(C_3_travel_times[i])
        
C_3_ratio_8_9={}
C_3_ratio_9_10={}
C_3_ratio_17_18={}
C_3_ratio_18_19={}

date_key=[236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327]

for i in date_key:
    C_3_ratio_8_9[i]={}
    C_3_ratio_9_10[i]={}
    C_3_ratio_17_18[i]={}
    C_3_ratio_18_19[i]={}
    
    for j in [7*60+40,8*60+0,8*60+20,8*60+40,9*60]:
        C_3_ratio_8_9[i][j]=[]
    
    for j in [8*60+40,9*60,9*60+20,9*60+40,10*60]:
        C_3_ratio_9_10[i][j]=[]
        
    for j in [16*60+40,17*60+0,17*60+20,17*60+40,18*60+0]:
        C_3_ratio_17_18[i][j]=[]
        
    for j in [17*60+40,18*60+0,18*60+20,18*60+40,19*60+0]:
        C_3_ratio_18_19[i][j]=[]

for i in C_3_travel_times_need:
    temp_date=i.month*31+i.day
    temp_hour=i.hour
    temp_min=i.minute
    
    
    if temp_hour*60+temp_min>=460 and temp_hour*60+temp_min<=540:
        C_3_ratio_8_9[temp_date][int(temp_hour*60+temp_min)].append(C_3_travel_times_need[i])
        
    if temp_hour*60+temp_min>=520 and temp_hour*60+temp_min<=600:
        C_3_ratio_9_10[temp_date][int(temp_hour*60+temp_min)].append(C_3_travel_times_need[i])
    
    if temp_hour*60+temp_min>=16*60+40 and temp_hour*60+temp_min<=18*60:
        C_3_ratio_17_18[temp_date][int(temp_hour*60+temp_min)].append(C_3_travel_times_need[i])
        
    if temp_hour*60+temp_min>=17*60+40 and temp_hour*60+temp_min<=19*60:
        C_3_ratio_18_19[temp_date][int(temp_hour*60+temp_min)].append(C_3_travel_times_need[i])
        
for i in date_key:
    k=0
    if C_3_ratio_8_9[i][480]==[]:k+=1
    if C_3_ratio_8_9[i][500]==[]:k+=1
    if C_3_ratio_8_9[i][520]==[]:k+=1 
    if k>1:del C_3_ratio_8_9[i]
        
for i in date_key:
    k=0
    if C_3_ratio_9_10[i][540]==[]:k+=1
    if C_3_ratio_9_10[i][560]==[]:k+=1
    if C_3_ratio_9_10[i][580]==[]:k+=1 
    if k>1:del C_3_ratio_9_10[i]
        
for i in date_key:
    k=0
    if C_3_ratio_17_18[i][17*60]==[]:k+=1 
    if C_3_ratio_17_18[i][17*60+20]==[]:k+=1 
    if C_3_ratio_17_18[i][17*60+40]==[]:k+=1
    if k>1:del C_3_ratio_17_18[i]
        
for i in date_key:
    k=0
    if C_3_ratio_18_19[i][18*60]==[]:k+=1 
    if C_3_ratio_18_19[i][18*60+20]==[]:k+=1 
    if C_3_ratio_18_19[i][18*60+40]==[]:k+=1
    if k>1:del C_3_ratio_18_19[i]
        
for i in C_3_ratio_8_9:
    for k in [480,500,520]:
        if C_3_ratio_8_9[i][k]==[] and C_3_ratio_8_9[i][k-20]!=[] and C_3_ratio_8_9[i][k+20]!=[]:
            C_3_ratio_8_9[i][k]=[np.mean([np.mean(C_3_ratio_8_9[i][k-20]),np.mean(C_3_ratio_8_9[i][k+20])])]
            
for i in C_3_ratio_9_10:
    for k in [9*60,9*60+20,9*60+40]:
        if C_3_ratio_9_10[i][k]==[] and C_3_ratio_9_10[i][k-20]!=[] and C_3_ratio_9_10[i][k+20]!=[]:
            C_3_ratio_9_10[i][k]=[np.mean([np.mean(C_3_ratio_9_10[i][k-20]),np.mean(C_3_ratio_9_10[i][k+20])])]
            
for i in C_3_ratio_17_18:
    for k in [17*60,17*60+20,17*60+40]:
        if C_3_ratio_17_18[i][k]==[] and C_3_ratio_17_18[i][k-20]!=[] and C_3_ratio_17_18[i][k+20]!=[]:
            C_3_ratio_17_18[i][k]=[np.mean([np.mean(C_3_ratio_17_18[i][k-20]),np.mean(C_3_ratio_17_18[i][k+20])])]
            
for i in C_3_ratio_18_19:
    for k in [18*60,18*60+20,18*60+40]:
        if C_3_ratio_18_19[i][k]==[] and C_3_ratio_18_19[i][k-20]!=[] and C_3_ratio_18_19[i][k+20]!=[]:
            C_3_ratio_18_19[i][k]=[np.mean([np.mean(C_3_ratio_18_19[i][k-20]),np.mean(C_3_ratio_18_19[i][k+20])])]
            
C_3_ratio_8_9_key=list(C_3_ratio_8_9.keys())
for i in C_3_ratio_8_9_key:
    k=0
    if C_3_ratio_8_9[i][480]==[]:k+=1
    if C_3_ratio_8_9[i][500]==[]:k+=1
    if C_3_ratio_8_9[i][520]==[]:k+=1 
    if k>0:del C_3_ratio_8_9[i]
        
C_3_ratio_9_10_key=list(C_3_ratio_9_10.keys())
for i in C_3_ratio_9_10_key:
    k=0
    if C_3_ratio_9_10[i][540]==[]:k+=1
    if C_3_ratio_9_10[i][560]==[]:k+=1
    if C_3_ratio_9_10[i][580]==[]:k+=1 
    if k>0:del C_3_ratio_9_10[i]

C_3_ratio_17_18_key=list(C_3_ratio_17_18.keys())
for i in C_3_ratio_17_18_key:
    k=0
    if C_3_ratio_17_18[i][17*60]==[]:k+=1 
    if C_3_ratio_17_18[i][17*60+20]==[]:k+=1 
    if C_3_ratio_17_18[i][17*60+40]==[]:k+=1
    if k>0:del C_3_ratio_17_18[i]

C_3_ratio_18_19_key=list(C_3_ratio_18_19.keys())
for i in  C_3_ratio_18_19_key:
    k=0
    if C_3_ratio_18_19[i][18*60]==[]:k+=1 
    if C_3_ratio_18_19[i][18*60+20]==[]:k+=1 
    if C_3_ratio_18_19[i][18*60+40]==[]:k+=1
    if k>0:del C_3_ratio_18_19[i]
        
for i in C_3_ratio_8_9:
    del C_3_ratio_8_9[i][8*60-20]
    del C_3_ratio_8_9[i][9*60]
    
for i in C_3_ratio_9_10:
    del C_3_ratio_9_10[i][9*60-20]
    del C_3_ratio_9_10[i][10*60]
    
for i in C_3_ratio_17_18:
    del C_3_ratio_17_18[i][17*60-20]
    del C_3_ratio_17_18[i][18*60]
    
for i in C_3_ratio_18_19:
    del C_3_ratio_18_19[i][18*60-20]
    del C_3_ratio_18_19[i][19*60]
    
for i in C_3_ratio_8_9:
    temp={480:[],500:[],520:[]}
    temp2=[]
    for j in C_3_ratio_8_9[i]: 
        if C_3_ratio_8_9[i][j]!=[]:
            temp[j].append(np.mean(C_3_ratio_8_9[i][j]))
            temp2.append(np.mean(C_3_ratio_8_9[i][j]))
        if C_3_ratio_8_9[i][j]==[]:
            temp[j].append(0)
            temp2.append(0)
    for k in temp:
        C_3_ratio_8_9[i][k]=temp[k]/np.mean(temp2)
        
for i in C_3_ratio_9_10:
    temp={540:[],560:[],580:[]}
    temp2=[]
    for j in C_3_ratio_9_10[i]: 
        if C_3_ratio_9_10[i][j]!=[]:
            temp[j].append(np.mean(C_3_ratio_9_10[i][j]))
            temp2.append(np.mean(C_3_ratio_9_10[i][j]))
        if C_3_ratio_9_10[i][j]==[]:
            temp[j].append(0)
            temp2.append(0)
    for k in temp:
        C_3_ratio_9_10[i][k]=temp[k]/np.mean(temp2)
        
for i in C_3_ratio_17_18:
    temp={17*60:[],17*60+20:[],17*60+40:[]}
    temp2=[]
    for j in C_3_ratio_17_18[i]: 
        if C_3_ratio_17_18[i][j]!=[]:
            temp[j].append(np.mean(C_3_ratio_17_18[i][j]))
            temp2.append(np.mean(C_3_ratio_17_18[i][j]))
        if C_3_ratio_17_18[i][j]==[]:
            temp[j].append(0)
            temp2.append(0)
    for k in temp:
        C_3_ratio_17_18[i][k]=temp[k]/np.mean(temp2)
        
for i in C_3_ratio_18_19:
    temp={18*60:[],18*60+20:[],18*60+40:[]}
    temp2=[]
    for j in C_3_ratio_18_19[i]: 
        if C_3_ratio_18_19[i][j]!=[]:
            temp[j].append(np.mean(C_3_ratio_18_19[i][j]))
            temp2.append(np.mean(C_3_ratio_18_19[i][j]))
        if C_3_ratio_18_19[i][j]==[]:
            temp[j].append(0)
            temp2.append(0)
    for k in temp:
        C_3_ratio_18_19[i][k]=temp[k]/np.mean(temp2)
        
C_3_ratio_8_9_480=[]
C_3_ratio_8_9_500=[]
C_3_ratio_8_9_520=[]

for i in C_3_ratio_8_9:
    for j in C_3_ratio_8_9[i]:
        if j==480: C_3_ratio_8_9_480.extend(C_3_ratio_8_9[i][j])
        if j==500: C_3_ratio_8_9_500.extend(C_3_ratio_8_9[i][j])
        if j==520: C_3_ratio_8_9_520.extend(C_3_ratio_8_9[i][j])
    
    
C_3_ratio_9_10_540=[]
C_3_ratio_9_10_560=[]
C_3_ratio_9_10_580=[]

for i in C_3_ratio_9_10:
    for j in C_3_ratio_9_10[i]:
        if j==540: C_3_ratio_9_10_540.extend(C_3_ratio_9_10[i][j])
        if j==560: C_3_ratio_9_10_560.extend(C_3_ratio_9_10[i][j])
        if j==580: C_3_ratio_9_10_580.extend(C_3_ratio_9_10[i][j])
            
C_3_ratio_17_18_1020=[]
C_3_ratio_17_18_1040=[]
C_3_ratio_17_18_1060=[]

for i in C_3_ratio_17_18:
    for j in C_3_ratio_17_18[i]:
        if j==1020: C_3_ratio_17_18_1020.extend(C_3_ratio_17_18[i][j])
        if j==1040: C_3_ratio_17_18_1040.extend(C_3_ratio_17_18[i][j])
        if j==1060: C_3_ratio_17_18_1060.extend(C_3_ratio_17_18[i][j])
    
C_3_ratio_18_19_1080=[]
C_3_ratio_18_19_1100=[]
C_3_ratio_18_19_1120=[]

for i in C_3_ratio_18_19:
    for j in C_3_ratio_18_19[i]:
        if j==1080: C_3_ratio_18_19_1080.extend(C_3_ratio_18_19[i][j])
        if j==1100: C_3_ratio_18_19_1100.extend(C_3_ratio_18_19[i][j])
        if j==1120: C_3_ratio_18_19_1120.extend(C_3_ratio_18_19[i][j])
            
C_3_ratio_8_9_480.sort()
C_3_ratio_8_9_480_used=C_3_ratio_8_9_480[5:len(C_3_ratio_8_9_480)-5]

C_3_ratio_8_9_500.sort()
C_3_ratio_8_9_500_used=C_3_ratio_8_9_500[5:len(C_3_ratio_8_9_500)-5]

C_3_ratio_8_9_520.sort()
C_3_ratio_8_9_520_used=C_3_ratio_8_9_520[5:len(C_3_ratio_8_9_520)-5]

C_3_ratio_9_10_540.sort()
C_3_ratio_9_10_540_used=C_3_ratio_9_10_540[5:len(C_3_ratio_9_10_540)-5]

C_3_ratio_9_10_560.sort()
C_3_ratio_9_10_560_used=C_3_ratio_9_10_560[5:len(C_3_ratio_9_10_560)-5]

C_3_ratio_9_10_580.sort()
C_3_ratio_9_10_580_used=C_3_ratio_9_10_580[5:len(C_3_ratio_9_10_580)-5]

C_3_ratio_17_18_1020.sort()
C_3_ratio_17_18_1020_used=C_3_ratio_17_18_1020[5:len(C_3_ratio_17_18_1020)-5]

C_3_ratio_17_18_1040.sort()
C_3_ratio_17_18_1040_used=C_3_ratio_17_18_1040[5:len(C_3_ratio_17_18_1040)-5]

C_3_ratio_17_18_1060.sort()
C_3_ratio_17_18_1060_used=C_3_ratio_17_18_1060[5:len(C_3_ratio_17_18_1060)-5]

C_3_ratio_18_19_1080.sort()
C_3_ratio_18_19_1080_used=C_3_ratio_18_19_1080[5:len(C_3_ratio_18_19_1080)-5]

C_3_ratio_18_19_1100.sort()
C_3_ratio_18_19_1100_used=C_3_ratio_18_19_1100[5:len(C_3_ratio_18_19_1100)-5]

C_3_ratio_18_19_1120.sort()
C_3_ratio_18_19_1120_used=C_3_ratio_18_19_1120[5:len(C_3_ratio_18_19_1120)-5]

names=locals()
C_3_ratio=[]
key1=['8_9_480','8_9_500','8_9_520','9_10_540','9_10_560','9_10_580','17_18_1020','17_18_1040','17_18_1060','18_19_1080','18_19_1100','18_19_1120']
for i in key1:
    C_3_ratio.append(np.mean(names['C_3_ratio_'+str(i)+'_used']))
    
C_3_ratio_8=C_3_ratio[0:3]
C_3_ratio_9=C_3_ratio[3:6]
C_3_ratio_17=C_3_ratio[6:9]
C_3_ratio_18=C_3_ratio[9:12]


print(C_3_ratio_8)
print(C_3_ratio_9)
print(C_3_ratio_17)
print(C_3_ratio_18)

train_path = '/Users/vayne/Desktop/dm_pro_engin/data_feature/C_3.csv'
test_path = '/Users/vayne/Desktop/dm_pro_engin/data_testing/C_3.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

train_x, train_y = get_x_y(train)
test_x, test_y = get_x_y(test)

gdbt = GradientBoostingRegressor(n_estimators=30, learning_rate=0.1, 
                                 max_depth=5, random_state=0, loss='ls').fit(train_x, train_y)

#print(mape_error(train_y, gdbt.predict(train_x)))
print(mape_error(test_y, gdbt.predict(test_x)))
result = gdbt.predict(train_x)
train_compare = pd.concat([train_y, pd.Series(result)],axis = 1)
#print(train_compare)

result = gdbt.predict(test_x)
test_compare = pd.concat([test_y, pd.Series(result)],axis = 1)
#print(test_compare)

train_path = '/Users/vayne/Desktop/dm_pro_engin/data_feature/C_3.csv'
test_path = '/Users/vayne/Desktop/dm_pro_engin/data_testing/C_3.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
test_hr=list(test['hr'])
test_date=list(test['date'])

result_with_hrdate=[]
for i in range(len(result)):
    result_with_hrdate.append([test_date[i],test_hr[i],result[i]])
    
predict_result=[]
for i in result_with_hrdate:
    if i[1]==8:
        temp=[]
        for k in C_3_ratio_8:
            temp.append([i[0],i[1],float(i[2]*k)])
        predict_result.extend(temp)
    if i[1]==9:
        temp=[]
        for k in C_3_ratio_9:
            temp.append([i[0],i[1],float(i[2]*k)])
        predict_result.extend(temp)
    if i[1]==17:
        temp=[]
        for k in C_3_ratio_17:
            temp.append([i[0],i[1],float(i[2]*k)])
        predict_result.extend(temp)
    if i[1]==18:
        temp=[]
        for k in C_3_ratio_18:
            temp.append([i[0],i[1],float(i[2]*k)])
        predict_result.extend(temp)
        
final_result=[]
for i in predict_result:
    final_result.append([i[0],i[1],i[2]])
    

final_result.sort(key = lambda x: x[0]) 

#有的时间窗口真实数据里是没有值的，所以要筛一下，去掉predicted value里的一些值
real_value_df_C_3.loc[real_value_df_C_3['route'] == 'C_3']

result_index=[]
for i in range(0,84,3):
    result_index.append([i,i+3])

final_result2=[]
for i in result_index:
    temp_result=final_result[i[0]:i[1]]
    temp_date=datetime.strptime(temp_result[0][0], "%m-%d")
    temp_month=temp_date.month
    temp_day=temp_date.day
    temp_hr=temp_result[0][1]
    
    for j in range(len(temp_result)):
        final_result2.append([temp_result[j],datetime.strptime('2016-'+str(temp_month)+'-'+str(temp_day)+' '+str(temp_hr)+':'+str(20*j)+':00', "%Y-%m-%d %H:%M:%S")])
        
real_value_df_C_3_time=list(real_value_df_C_3['time'])

final_result3=[]
for i in final_result2:
    if i[1] in real_value_df_C_3_time:final_result3.append(i[0])

pred_value=[]
for i in final_result3:
    pred_value.append(i[2])
    

#load the real value 
real_value_df_C_3=real_value_df.loc[real_value_df['route'] =='C_3']

real_value=list(real_value_df_C_3['label'])

real_value1=[]
for i in real_value:
    real_value1.append(float(i))


mape_list=[]
for i in range(len(pred_value)):
    mape_list.append(np.abs((real_value1[i] - pred_value[i]) / real_value1[i]))
    
np.mean(mape_list)


test_compare=[]
for i in range(len(pred_value)):
    test_compare.append([real_value1[i],pred_value[i]])
test_compare



