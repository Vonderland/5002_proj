1. 用aggregate_travel_time.py生成test的20分钟窗口平均数据
2. 用data_preprocess.py将训练集和测试集都处理成上下午中6个20分钟时间窗口平均通过时间的格式                                        


Validation 2:
用2016/9/19-2016/9/25 and 2016/10/10-2016/10/17 的数据作为validation set，对gbdt进行调参，并乘以ratio后的结果

model_with_new_data1：
根据validation的调出来的参数，运用到testing set上的结果
