# 将一行的变成多行
df.drop(['爱好'], axis=1).join(df['爱好'].str.split('、', expand=True).stack().reset_index(level=1, drop=True).rename('df_name1'))

#一、先将‘爱好’字段拆分
df['爱好']=df['爱好'].map(lambda x:x.split(','))
#二、然后直接调用explode()方法
df_new=df.explode('爱好')

#统计并画出柱状图
df_new['df_name1'].value_counts().plot.bar()
