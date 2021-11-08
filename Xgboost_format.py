#!/usr/bin/env python
# coding: utf-8

# In[71]:


# import文一覧
#データ解析用ライブラリ
import pandas as pd
import numpy as np

#データ可視化ライブラリ
import matplotlib.pyplot as plt
import seaborn as sns

#Xgboostライブラリ
import xgboost as xgb

#訓練データとモデル評価用データに分けるライブラリ
from sklearn.model_selection import train_test_split

#LavelEncoderのライブラリをインポート
from sklearn.preprocessing import LabelEncoder


# In[72]:


#一通り評価指標をインポート
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae


# In[84]:


#csvデータの読み込み
sol_df = pd.read_csv('2015_winter_FP.csv')
sol_df = sol_df.drop('Cell_number', axis=1)
sol_df_copy = sol_df


# In[85]:


#時系列データに変換するためにデータ型を文字列型に変換
sol_df_copy = sol_df_copy.astype({'year': 'str', 'month': 'str', 'day':'str', 'time':'str'})


# In[86]:


#時系列データの作成
sol_df_copy['datetime'] = '200' + sol_df_copy['year'] + '-' + sol_df_copy['month'] + '-' + sol_df_copy['day'] + ' ' + sol_df_copy['time'] + ':00'


# In[87]:


sol_df_copy['datetime'] =  pd.to_datetime(sol_df_copy['datetime']) 


# In[88]:


sol_df = sol_df.dropna()


# In[89]:


sol_df_copy = sol_df_copy.dropna()


# In[90]:


# 終値を24時間分移動させる
sol_df_shift = sol_df_copy
sol_df_shift.EGC = sol_df_shift.EGC.shift(-24)


# In[91]:


# 最後の行を除外
sol_df_shift = sol_df_shift[:-24]
 
# 念のためデータをdf_2として新しいデータフレームへコピ−
df_2 = sol_df_shift.copy()


# In[92]:


xgb_params = {
    "learning_rate":0.05,
    "seed":4
}


# In[93]:


#object型の変数を取得
categories = sol_df_shift.columns[sol_df_shift.dtypes == "object"]


# In[97]:


#データ型を元に戻す
for cat in categories:
    sol_df_shift[cat] = sol_df_shift[cat].astype('int')


# In[115]:


#データの区分け
train_set, test_set = train_test_split(sol_df_shift, test_size=0.2, random_state=4, shuffle=False, stratify=None)


# In[116]:


#データの確認
test_set


# In[117]:


#訓練データを説明変数データ(X_train)と目的変数データ(y_train)に分割
X_train = train_set.drop(['EGC','datetime'], axis=1)
y_train = train_set['EGC']


# In[118]:


#モデル評価用データを説明変数データ(X_train)と目的変数データ(y_train)に分割
X_test = test_set.drop(['EGC', 'datetime'], axis=1)
y_test = test_set['EGC']


# In[119]:


#xgboost用データ
xgb_train = xgb.DMatrix(X_train,label=y_train)
xgb_eval = xgb.DMatrix(X_test, label=y_test)
evals = [(xgb_train, "train"), (xgb_eval, "eval")]


# In[120]:


#モデル作成
model_xgb = xgb.train(xgb_params,
                      xgb_train,
                      evals=evals,
                      num_boost_round=1000,
                      early_stopping_rounds=20,
                      verbose_eval=10,)


# In[121]:


y_pred = model_xgb.predict(xgb_eval)


# In[122]:


#現状の予測値と実際の値の違いを可視化
actual_pred_df = pd.DataFrame({
    "actual":y_test,
    "pred":y_pred,
    "datetime":test_set['datetime']
})


# In[123]:


pred_df = actual_pred_df.drop('datetime', axis=1)


# In[158]:


#あらゆる指標の保存
RMSE = mse(pred_df['actual'], pred_df['pred'])
R2 = r2_score(pred_df['actual'], pred_df['pred'])
MAE = mae(pred_df['actual'], pred_df['pred'])
MAEP = MAE/pred_df['actual'].mean()
RMSEP = RMSE/pred_df['actual'].mean()


# In[159]:


#for文を使って他の指標も作成
actual_1 = pred_df['actual'].values.tolist()
pred_1 = pred_df['pred'].values.tolist()


# In[160]:


pred_df_copy = pred_df
pred_df_copy = pred_df_copy[pred_df_copy['actual'] != 0]


# In[161]:


# 保留の指標
#MAPE = np.mean(np.abs((pred_df_copy['actual'] - pred_df_copy['pred'] / pred_df_copy['actual'])) * 100)
#RMSEP = np.sqrt(np.mean(((pred_df_copy['pred'] - pred_df_copy['actual']) / pred_df_copy['actual'])**2))*100


# In[162]:


pred_df


# In[163]:


fig = plt.figure(figsize=(15,7))
ax = plt.subplot(111)
x = actual_pred_df['datetime']
y1 = pred_df['actual']
y2 = pred_df['pred']
ax.set_xlabel('Seasons', fontsize=15)
ax.set_ylabel('Insolation[kw]', fontsize=15)
ax.plot(x,y1,'b-')
ax.plot(x,y2,'r-')
plt.text(0.1, 0.9, 'RMSE = {}'.format(str(round(RMSE, 5))), transform=ax.transAxes, fontsize=15)
plt.text(0.1, 0.8, 'R^2 = {}'.format(str(round(R2, 5))), transform=ax.transAxes, fontsize=15)
plt.show()


# In[166]:


#R^2,RMSE
plt.figure(figsize=(8,8))
ax = plt.subplot(111)
ax.scatter('actual', 'pred', data=pred_df)
ax.set_xlabel('True Value[kw]', fontsize=15)
ax.set_ylabel('Pred Value[kw]', fontsize=15)
ax.set_xlim(pred_df.min().min()-0.1 , pred_df.max().max()+0.1)
ax.set_ylim(pred_df.min().min()-0.1 , pred_df.max().max()+0.1)
x = np.linspace(pred_df.min().min()-0.1, pred_df.max().max()+0.1, 2)
y = x
ax.plot(x,y,'r-')
plt.text(0.1, 0.9, 'RMSE = {}'.format(str(round(RMSE, 3))), transform=ax.transAxes, fontsize=15)
plt.text(0.4, 0.9, 'RMSEP = {}'.format(str(round(RMSEP, 3))), transform=ax.transAxes, fontsize=15)
plt.text(0.4, 0.8, 'MAE = {}'.format(str(round(MAE, 3))), transform=ax.transAxes, fontsize=15)
plt.text(0.7, 0.9, 'MAEP = {}'.format(str(round(MAEP, 3))), transform=ax.transAxes, fontsize=15)
plt.text(0.1, 0.8, 'R^2 = {}'.format(str(round(R2, 3))), transform=ax.transAxes, fontsize=15)

plt.show()


# In[ ]:




