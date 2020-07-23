import pandas as pd

def pd_(df:pd.DataFrame):
    '''将一行的pd根据某一列的数据变成多行'''
    # method 1
    df.drop(['hobby'], axis=1).join(df['hobby'].str.split('、', expand=True).stack().reset_index(level=1, drop=True).rename('df_name1'))
    
    # method 2
    #一、先将‘爱好’字段拆分
    df['hobby']=df['hobby'].map(lambda x:x.split(','))
    #二、然后直接调用explode()方法
    df_new=df.explode('hobby')

    #统计并画出柱状图
    df_new['df_name1'].value_counts().plot.bar()

  
 def gen_feature(data): 
    '''将组中得到的结果反映在组中每一条数据中。'''
    df = data.copy()
    df = df[ ['user','item','sim_weight','loc_weight','time_weight','rank_weight','index'] ]
    feat = df[ ['index','user','item'] ]
    df = df.groupby( ['user','item'] )[ ['sim_weight','loc_weight','time_weight','rank_weight'] ].agg( ['sum','mean'] ).reset_index()
    # build column names
    cols = [ f'item_{j}_{i}' for i in ['sim_weight','loc_weight','time_weight','rank_weight'] for j in ['sum','mean'] ]
    # rename df
    df.columns = [ 'user','item' ]+ cols
    feat = pd.merge( feat, df, on=['user','item'], how='left')
    feat = feat[ cols ] 
    return feat

def gen_group_dict(data):
    '''为组内聚合的结果产生对应的dict'''
    df =data.copy()
    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    
    '''
    为dict生成默认值
    i2i_sim_seq.setdefault((item,relate_item), [])
    i2i_sim_seq[ (item,relate_item) ].append( (loc1, loc2, t1, t2, len(items) ) )
    我常用：
    i2i_sim_seq[(item,relate_item)] = [(loc1, loc2, t1, t2, len(items)]
    '''

