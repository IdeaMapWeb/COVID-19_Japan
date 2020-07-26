#include package
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import pandas as pd

#pandasでCSVデータ読む。C:\Users\user\simulation\COVID-19\csse_covid_19_data\japan\test
data = pd.read_csv('data/covid19/test_confirmed.csv',encoding="cp932")
data_r = pd.read_csv('data/covid19/test_recovered.csv',encoding="cp932")
data_d = pd.read_csv('data/covid19/test_deaths.csv',encoding="cp932")
data_s = pd.read_csv('data/covid19/test_serious.csv',encoding="cp932")
#data = pd.read_csv('data/test_confirmed_.csv',encoding="cp932")
#data_r = pd.read_csv('data/test_recovered_.csv',encoding="cp932")
#data_d = pd.read_csv('data/test_deaths_.csv',encoding="cp932")

confirmed = [0] * (len(data.columns) - 1)
day_confirmed = [0] * (len(data.columns) - 1)
confirmed_r = [0] * (len(data_r.columns) - 1)
day_confirmed_r = [0] * (len(data.columns) - 1)
confirmed_d = [0] * (len(data_d.columns) - 1)
diff_confirmed = [0] * (len(data.columns) - 1)
confirmed_s = [0] * (len(data.columns) - 1)
days_from_1_Jun_20 = np.arange(0, len(data.columns) - 1, 1)
beta_ = [0] * (len(data_r.columns) - 1)
gamma_ = [0] * (len(data_d.columns) - 1)
daystamp = "531"

city,city0 = "東京","tokyo"
#city,city0 = "大阪","oosaka" 
#city,city0 = "北海道","hokkaido"
#city,city0 = "愛知","aichi"
#city,city0 = "石川","ishikawa"
#city,city0 = "富山","toyama"
#city,city0 = "千葉","chiba" 
#city,city0 = "埼玉","saitama"
#city,city0 = "福岡","fukuoka"
#city,city0 = "兵庫","hyougo"
#city,city0 = "京都","kyoto"
#city,city0 = "神奈川","kanagawa"
#city,city0 = "沖縄","okinawa"
#city,city0 = "合計","total_japan"
#city,city0 = "総計","total_japan"
#city,city0 = "東京以外","extokyo"

skd=1 #2 #1 #4 #3 #2 #slopes average factor
#データを加工する
t_cases = 0
t_recover = 0
t_deaths = 0
for i in range(0, len(data_r), 1):
    if (data_r.iloc[i][0] == city): #for country/region
        print(str(data_r.iloc[i][0]))
        for day in range(1, len(data.columns), 1):            
            confirmed_r[day - 1] += data_r.iloc[i][day]
            if day < 1+skd:
                day_confirmed_r[day-1] += data_r.iloc[i][day]
            else:
                day_confirmed_r[day-1] += (data_r.iloc[i][day] - data_r.iloc[i][day-skd])/(skd)
        t_recover += data_r.iloc[i][day]        
for i in range(0, len(data_d), 1):
    if (data_d.iloc[i][0] == city): #for country/region
        print(str(data_d.iloc[i][0]) )
        for day in range(1, len(data.columns), 1):
            confirmed_d[day - 1] += float(data_d.iloc[i][day]) #fro drawings
        t_deaths += float(data_d.iloc[i][day])        
for i in range(0, len(data), 1):
    if (data.iloc[i][0] == city): #for country/region
        print(str(data.iloc[i][0]))
        for day in range(1, len(data.columns), 1):
            confirmed[day - 1] += data.iloc[i][day] -  confirmed_r[day - 1] -confirmed_d[day-1]
            if day == 1:
                day_confirmed[day-1] += data.iloc[i][day]
            else:
                day_confirmed[day-1] += data.iloc[i][day] - data.iloc[i][day-1]
for i in range(0, len(data_d), 1):
    if (data_d.iloc[i][0] == city): #for country/region
        print(str(data_d.iloc[i][0]) )
        for day in range(1, len(data.columns), 1):
            confirmed_s[day - 1] += float(data_s.iloc[i][day])
            
def EMA(x, n):
    a= 2/(n+1)
    return pd.Series(x).ewm(alpha=a).mean()                
                
day_confirmed[0]=0                
df_c = pd.DataFrame(day_confirmed)                
df_c.to_csv('data/day_comfirmed_{}.csv'.format(city0))
m=15
dc_m = EMA(day_confirmed,m) #EMA(day_confirmed,m) #df_c.rolling(m).mean()
dc_m.to_csv('data/day_comfirmed_EMAmean{}_{}.csv'.format(m,city0))
dc_m_l=dc_m.values.tolist()

tl_confirmed = 0

for i in range(skd, len(confirmed), 1):        
    if confirmed[i] > 0:    
        gamma_[i]=float(day_confirmed_r[i])/float(confirmed[i])
    else:
        continue
tl_confirmed = confirmed[len(confirmed)-1] + confirmed_r[len(confirmed)-1] + confirmed_d[len(confirmed)-1]
t_cases = tl_confirmed


#matplotlib描画
fig, ax1 = plt.subplots(1,1,figsize=(1.6180 * 4, 4*1))
ax3 = ax1.twinx()

lns1=ax3.plot(days_from_1_Jun_20[1:], confirmed[1:], "o-", color="red",label = "cases")
lns2=ax3.plot(days_from_1_Jun_20[1:], confirmed_r[1:], "o-", color="green",label = "recovered")
lns3=ax3.plot(days_from_1_Jun_20[1:], confirmed_d[1:], "o-", color="blue",label = "deaths")
lns4=ax3.plot(days_from_1_Jun_20[1:], confirmed_s[1:], "o-", color="black",label = "serious")
lns10=ax1.bar(days_from_1_Jun_20[1:], day_confirmed[1:], label = "day_cases")

lns_ax3 = lns1  +lns2+lns3 +lns4
labs_ax3 = [l.get_label() for l in lns_ax3]
ax3.legend(lns_ax3, labs_ax3, loc=2)

ax1.legend(loc=1)

ax3.set_title(city0 +" ; {} cases, {} recovered, {} deaths".format(t_cases,t_recover,t_deaths))
ax3.set_xlabel("days from 1, Jun, 2020")
ax3.set_ylabel("cases, recovered,deaths,serious ")
ax1.set_ylabel("day_cases")
ax1.set_xlim(0,60)
ax3.set_ylim(10,20000)
ax1.set_ylim(10,2000) #1500
ax1.set_yscale('log')
ax3.set_yscale('log')

ax1.grid()

plt.pause(1)
plt.savefig('./fig/original_data_{}_{}_old_.png'.format(city,daystamp)) 
plt.close() 

fig, ax1 = plt.subplots(1,1,figsize=(1.6180 * 4, 4*1))
ax3 = ax1.twinx()

lns1=ax3.plot(days_from_1_Jun_20[1:], confirmed[1:], "o-", color="red",label = "cases")
lns2=ax3.plot(days_from_1_Jun_20[1:], confirmed_r[1:], "o-", color="green",label = "recovered")
lns3=ax3.plot(days_from_1_Jun_20[1:], confirmed_d[1:], "o-", color="blue",label = "deaths")
lns10=ax1.bar(days_from_1_Jun_20[1:], day_confirmed[1:], label = "day_cases")

lns_ax3 = lns1 +lns2+lns3
labs_ax3 = [l.get_label() for l in lns_ax3]
ax3.legend(lns_ax3, labs_ax3, loc=2)

ax1.legend(loc=1)

ax3.set_title(city0 +" ; {} cases, {} recovered, {} deaths".format(t_cases,t_recover,t_deaths))
ax3.set_xlabel("days from 1, Jun, 2020")
ax3.set_ylabel("cases, recovered,deaths ")
ax1.set_ylabel("day_cases")
ax1.set_xlim(0,60)
ax3.set_ylim(0,10000)
ax1.set_ylim(0,1000) #1500
#ax3.set_yscale('log')

ax1.grid()

plt.pause(1)
plt.savefig('./fig/original_data_{}_{}linear_old.png'.format(city,daystamp)) 
plt.close() 

fig, ax1 = plt.subplots(1,1,figsize=(1.6180 * 4, 4*1))
ax3 = ax1.twinx()

lns1=ax3.plot(days_from_1_Jun_20[1:], confirmed[1:], "o-", color="red",label = "cases")
lns10=ax1.bar(days_from_1_Jun_20[1:], day_confirmed[1:], label = "day_cases")
#lns11=ax1.plot(days_from_1_Jun_20[m:], df_c[m:],"o-", color="blue", label = "day_cases_")
lns12=ax1.plot(days_from_1_Jun_20[m:], dc_m[m:],"o-", color="blue", label = "day_cases_mean{}".format(m))

lns_ax3 = lns1  #+lns2+lns3
labs_ax3 = [l.get_label() for l in lns_ax3]
ax3.legend(lns_ax3, labs_ax3, loc=0)

ax1.legend(loc=2)

ax3.set_title(city0 +" ; {} cases, {} recovered, {} deaths".format(t_cases,t_recover,t_deaths))
ax3.set_xlabel("days from 1, Jun, 2020")
ax3.set_ylabel("cases ")
ax1.set_ylabel("day_cases")
ax1.set_xlim(0,60)
ax3.set_ylim(100,20000)
ax1.set_ylim(10,2000) #1500
ax1.set_yscale('log')
ax3.set_yscale('log')

ax1.grid()

plt.pause(1)
plt.savefig('./fig/EMAmean{}_data_{}_{}_.png'.format(m,city,daystamp)) 
plt.close() 

fig, ax1 = plt.subplots(1,1,figsize=(1.6180 * 4, 4*1))
ax3 = ax1.twinx()

lns1=ax3.plot(days_from_1_Jun_20[1:], confirmed[1:], "o-", color="red",label = "cases")
lns4=ax1.plot(days_from_1_Jun_20[1:36], confirmed_s[20:55], "o-", color="black",label = "serious")
#lns10=ax1.bar(days_from_1_Jun_20[1:], day_confirmed[1:], label = "day_cases")

lns_ax1 = lns4
labs_ax1 = [l.get_label() for l in lns_ax1]
ax1.legend(lns_ax1, labs_ax1, loc=2)

ax3.legend(loc=1)

ax1.set_title(city0 +" ; {} cases, {} recovered, {} deaths".format(t_cases,t_recover,t_deaths))
ax1.set_xlabel("days from 1, Jun, 2020")
ax1.set_ylabel("serious, day_cases ")
ax3.set_ylabel("cases")
ax3.set_xlim(0,60)
ax1.set_xlim(0,60)
ax3.set_ylim(100,10000)
ax1.set_ylim(5,500) #1500
ax3.set_yscale('log')
ax1.set_yscale('log')

ax1.grid()

plt.pause(1)
plt.savefig('./fig/original_data_{}_{}_new.png'.format(city,daystamp)) 
plt.close() 
