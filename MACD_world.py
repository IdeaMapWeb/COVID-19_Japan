import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from pandas_datareader import data
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL

def get_stock(stock,start,end):
    df = data.DataReader(stock, 'stooq',start)["Close"]
    df = df.iloc[::-1]
    return df[start:end]

from numba import jit
@jit(nopython=True)
def EMA3(x, n):
    alpha = 2/(n+1)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1,len(x)):
        y[i] = alpha*x[i] + (1-alpha)*y[i-1]
    return y

def EMA1(x, n):
    a= 2/(n+1)
    return pd.Series(x).ewm(alpha=a).mean()

def calc_dff(dK,dD):
    dff= dK.copy()
    for j in range(len(dK)):
        dff[j]=100*(dK[j]-dD[j])/(abs(dK[j])+abs(dD[j])+1)
    return dff
"""
stock0 = 'ZM' #'9437'
stock = stock0 #+ '.JP'
start = dt.date(2020,1,1)
end = dt.date(2020,7,12)
df = pd.DataFrame(get_stock(stock, start, end))
date_df=df['Close'].index.tolist() #ここがポイント
print(date_df[0:30])
series = df['Close'].values.tolist()
"""
#pandasでCSVデータ読む。C:\Users\user\simulation\COVID-19\csse_covid_19_data\japan\test
data = pd.read_csv('data/covid19/time_series_covid19_confirmed_global.csv')
data_r = pd.read_csv('data/covid19/time_series_covid19_recovered_global.csv')
data_d = pd.read_csv('data/covid19/time_series_covid19_deaths_global.csv')

confirmed = [0] * (len(data.columns) - 4)
day_confirmed = [0] * (len(data.columns) - 4)
confirmed_r = [0] * (len(data_r.columns) - 4)
day_confirmed_r = [0] * (len(data.columns) - 4)
confirmed_d = [0] * (len(data_d.columns) - 4)
diff_confirmed = [0] * (len(data.columns) - 4)
days_from_1_Jun_20 = np.arange(0, len(data.columns) - 4, 1)
beta_ = [0] * (len(data_r.columns) - 4)
gamma_ = [0] * (len(data_d.columns) - 4)
daystamp = "131"

#city,city0 = "Japan","Japan"
#city,city0 = "Hubei","Hubei"
#city,city0  = "Korea, South","Korea, South"
#city,city0 = "Iran","Iran"
#city,city0 = "Italy","Italy"
#city,city0 = "Spain","Spain"
#city,city0 = "Iraq","Iraq"
#city,city0 = "Singapore","Singapore"
#city,city0 = "Germany","Germany"
#city,city0 = "China","China"
#city,city0 = "US","US"
#city,city0 = "France","France"
#city,city0 = "United Kingdom","United Kingdom"
#city,city0 = "Switzerland","Switzerland"
#city,city0 = "Indonesia","Indonesia"
#city,city0 = "Guangdong","Guangdong"
#city,city0 = "Zhejiang","Zhejiang"
#city,city0 = "New York","New York"
#city,city0 = "total","total"
#city,city0 ="Thailand","Thailand"
#city,city0="Brazil","Brazil"
#city,city0="India","India"
#city,city0="US","US"
#city,city0="Sweden","Sweden"
#city,city0="Turkey","Turkey"
#city,city0="South Africa","South Africa"
#city,city0="Russia","Russia"
#city,city0="Mexico","Mexico"
#city,city0="Australia","Australia"
city,city0="Israel","Israel"


skd=1 #2 #1 #4 #3 #2 #slopes average factor
#データを加工する
t_cases = 0
t_recover = 0
t_deaths = 0
for i in range(0, len(data_r), 1):
    if (data_r.iloc[i][1] == city): #for country/region
        print(str(data_r.iloc[i][0]))
        for day in range(4, len(data.columns), 1):            
            confirmed_r[day - 4] += float(data_r.iloc[i][day])
            if day < 1+skd:
                day_confirmed_r[day-4] += float(data_r.iloc[i][day])
            else:
                day_confirmed_r[day-4] += (float(data_r.iloc[i][day]) - float(data_r.iloc[i][day-skd]))/(skd)
        t_recover += float(data_r.iloc[i][day])        
for i in range(0, len(data_d), 1):
    if (data_d.iloc[i][1] == city): #for country/region
        print(str(data_d.iloc[i][0]) )
        for day in range(4, len(data.columns), 1):
            confirmed_d[day - 4] += float(data_d.iloc[i][day]) #fro drawings
        t_deaths += float(data_d.iloc[i][day])        
for i in range(0, len(data), 1):
    if (data.iloc[i][1] == city): #for country/region
        print(str(data.iloc[i][0]))
        for day in range(4, len(data.columns), 1):
            confirmed[day - 4] += data.iloc[i][day] -  confirmed_r[day - 4] -confirmed_d[day-4]
            if day == 1:
                day_confirmed[day-4] += data.iloc[i][day]
            else:
                day_confirmed[day-4] += data.iloc[i][day] - data.iloc[i][day-1]

def EMA(x, n):
    a= 2/(n+1)
    return pd.Series(x).ewm(alpha=a).mean()                
                
day_confirmed[0]=0                
df = pd.DataFrame()                
date = pd.date_range("20200122", periods=len(day_confirmed))
df = pd.DataFrame(df,index = date)
df['Close'] = day_confirmed
df.to_csv('data/day_comfirmed_new_{}.csv'.format(city0))
date_df=df['Close'].index.tolist() #ここがポイント
print(date_df[len(day_confirmed)-20:len(day_confirmed)])
series = df['Close'].values.tolist()
stock0 = city0
stock = stock0
start = dt.date(2020,2,1)
end = dt.date(2020,7,24)

bunseki = "trend" #series" #cycle" #trend
cycle, trend = sm.tsa.filters.hpfilter(series, 144)
series2 = trend

y12 = EMA3(series2, 12)
y26 = EMA3(series2, 26)
MACD = y12 -y26
signal = EMA3(MACD, 9)
hist_=MACD-signal

ind3=date_df[:]
print(len(series),len(ind3))

fig, (ax1,ax2) = plt.subplots(2,1,figsize=(1.6180 * 8, 4*2),dpi=200)
ax1.bar(ind3,series,label="series")
ax1.plot(ind3,series2, "o-", color="blue",label="series2")
ax1.plot(ind3,y12, ".-", color="red",label="y12")
ax1.plot(ind3,y26, ".-", color="green",label="y26")
ax2.plot(ind3,MACD,label="MACD")
ax2.plot(ind3,signal,label="signal")
ax2.bar(ind3,hist_)
ax1.legend()
ax2.legend()
ax1.set_ylim(1,)
#ax2.set_ylim(0,)
ax1.set_yscale('log')
#ax2.set_yscale('log')
ax1.grid()
ax2.grid()
plt.savefig("./fig/world/{}/ema_decompose_%5K%25D_{}_{}new{}.png".format(stock0,stock,bunseki,start))
plt.pause(1)
plt.close()

fig, (ax1,ax2) = plt.subplots(2,1,figsize=(1.6180 * 8, 4*2),dpi=200)
ax1.bar(ind3,series,label="series")
ax1.plot(ind3,series2, "o-", color="blue",label="series2")
ax1.plot(ind3,y12, ".-", color="red",label="y12")
ax1.plot(ind3,y26, ".-", color="green",label="y26")
ax2.plot(ind3,MACD,label="MACD")
ax2.plot(ind3,signal,label="signal")
ax2.bar(ind3,hist_)
ax1.legend()
ax2.legend()
ax1.set_ylim(0,)
#ax2.set_ylim(0,)
#ax1.set_yscale('log')
#ax2.set_yscale('log')
ax1.grid()
ax2.grid()
plt.savefig("./fig/world/{}/ema_decompose_%5K%25D_{}_{}linear{}.png".format(stock0,stock,bunseki,start))
plt.pause(1)
plt.close()

df['Close']=series  #series" #cycle" #trend
df['series2']=series2
df['y12'] = EMA1(df['Close'], 12)
df['y26'] =  EMA1(df['Close'], 26)
df['MACD'] = df['y12'] -df['y26']
df['signal'] = EMA1(df['MACD'], 9)
df['hist_']=df['MACD']-df['signal']
date_df=df['Close'].index.tolist()
print(df[len(series2)-20:len(series2)])

fig, (ax1,ax2) = plt.subplots(2,1,figsize=(1.6180 * 8, 4*2),dpi=200)
ax1.bar(ind3,series, label="series")
ax1.plot(df['series2'],"o-", color="blue",label="series2")
ax1.plot(df['y12'],".-", color="red",label="y12")
ax1.plot(df['y26'],".-", color="green",label="y26")
ax2.plot(df['MACD'],label="MACD")
ax2.plot(df['signal'],label="signal")
ax2.bar(date_df,df['hist_'])
ax1.legend()
ax2.legend()
ax1.set_ylim(1,)
#ax2.set_ylim(0,)
ax1.set_yscale('log')
#ax2.set_yscale('log')
ax1.grid()
ax2.grid()
plt.savefig("./fig/world/{}/ema_df_decompose_%5K%25D_{}_{}new{}.png".format(stock0,stock,bunseki,start))
plt.pause(1)
plt.close()