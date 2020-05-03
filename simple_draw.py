#include package
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import pandas as pd

daystamp= '430'
#pandasでCSVデータ読む。
data = pd.read_csv('COVID-19/csse_covid_19_data/csse_covid_19_time_series'+daystamp+'/time_series_covid19_confirmed_global.csv')
data_r = pd.read_csv('COVID-19/csse_covid_19_data/csse_covid_19_time_series'+daystamp+'/time_series_covid19_recovered_global.csv')
data_d = pd.read_csv('COVID-19/csse_covid_19_data/csse_covid_19_time_series'+daystamp+'/time_series_covid19_deaths_global.csv')

    
confirmed = [0] * (len(data.columns) - 4)
day_confirmed = [0] * (len(data.columns) - 4)
confirmed_r = [0] * (len(data_r.columns) - 4)
day_confirmed_r = [0] * (len(data.columns) - 4)
confirmed_d = [0] * (len(data_d.columns) - 4)
diff_confirmed = [0] * (len(data.columns) - 4)
days_from_22_Jan_20 = np.arange(0, len(data.columns) - 4, 1)
days_from_22_Jan_20_ = np.arange(0, len(data.columns) - 4, 1)
beta_ = [0] * (len(data_r.columns) - 4)
gamma_ = [0] * (len(data_d.columns) - 4)


city = "Japan"
#city ="world"
#city = "Germany"
#city = "Italy"
#city = "Spain"
#city = "United Kingdom"
#city ="US"
#city = "Iran"
#city = "Switzerland"
#city ="Sweden"
#city = "France"
#city = "Hong Kong"
#city = "Beijing"
#city = "Taiwan"
#city = "Hubei"
#city = "Korea, South"
#city = "India"
#city = "Thailand"
#city ="Russia"
#city = "Singapore"
#city = "South Africa"
#city = "Netherlands"
#city = "New Zealand"

skd=5 #2 #1 #4 #3 #2 #slopes average factor
#データを加工する
t_cases = 0
t_recover = 0
t_deaths = 0
for i in range(0, len(data_r), 1):
    if (data_r.iloc[i][1] == city): #for country/region
    #if (data_r.iloc[i][0] == city):  #for province:/state  
        print(str(data_r.iloc[i][0]) + " of " + data_r.iloc[i][1])
        for day in range(4, len(data.columns), 1):            
            confirmed_r[day - 4] += data_r.iloc[i][day]
            if day < 4+skd:
                day_confirmed_r[day-4] += data_r.iloc[i][day]
            else:
                day_confirmed_r[day-4] += (data_r.iloc[i][day] - data_r.iloc[i][day-skd])/(skd)
        t_recover += data_r.iloc[i][day]        
for i in range(0, len(data_d), 1):
    if (data_d.iloc[i][1] == city): #for country/region
    #if (data_d.iloc[i][0] == city):  #for province:/state  
        print(str(data_d.iloc[i][0]) + " of " + data_d.iloc[i][1])
        for day in range(4, len(data.columns), 1):
            confirmed_d[day - 4] += data_d.iloc[i][day] #fro drawings
        t_deaths += data_d.iloc[i][day]        
for i in range(0, len(data), 1):
    if (data.iloc[i][1] == city): #for country/region
    #if (data.iloc[i][0] == city):  #for province:/state  
        print(str(data.iloc[i][0]) + " of " + data.iloc[i][1])
        for day in range(4, len(data.columns), 1):
            confirmed[day - 4] += data.iloc[i][day] -  confirmed_r[day - 4] -confirmed_d[day-4]
            if day == 4:
                day_confirmed[day-4] += data.iloc[i][day]
            else:
                day_confirmed[day-4] += data.iloc[i][day] - data.iloc[i][day-1]

tl_confirmed = 0

for i in range(skd, len(confirmed), 1):        
    if confirmed[i] > 0:    
        gamma_[i]=day_confirmed_r[i]/confirmed[i]
    else:
        continue
tl_confirmed = confirmed[len(confirmed)-1] + confirmed_r[len(confirmed)-1] + confirmed_d[len(confirmed)-1]
t_cases = tl_confirmed


#matplotlib描画
fig, ax1 = plt.subplots(1,1,figsize=(1.6180 * 4, 4*1))
ax3 = ax1.twinx()

lns1=ax3.plot(days_from_22_Jan_20, confirmed, "o-", color="red",label = "cases")
lns2=ax3.plot(days_from_22_Jan_20, confirmed_r, "o-", color="green",label = "recovered")
lns3=ax3.plot(days_from_22_Jan_20, confirmed_d, "o-", color="blue",label = "deaths")
lns10=ax1.bar(days_from_22_Jan_20, day_confirmed, label = "day_cases")

lns_ax3 = lns1 +lns2+lns3
labs_ax3 = [l.get_label() for l in lns_ax3]
ax3.legend(lns_ax3, labs_ax3, loc=0)

ax1.legend(loc=3)

ax3.set_title(city +" ; {} cases, {} recovered, {} deaths".format(t_cases,t_recover,t_deaths))
ax3.set_xlabel("days from 22, Jan, 2020")
ax3.set_ylabel("cases, recovered,deaths ")
ax1.set_ylabel("day_cases")
ax1.set_xlim(0,100)
ax3.set_ylim(1,)
ax1.set_ylim(0,)
ax3.set_yscale('log')

ax1.grid()

plt.pause(1)
plt.savefig('./fig/original_data_{}_{}.png'.format(city,daystamp)) 
plt.close() 

fig, ax1 = plt.subplots(1,1,figsize=(1.6180 * 4, 4*1))
ax3 = ax1.twinx()

lns1=ax3.plot(days_from_22_Jan_20, confirmed, "o-", color="red",label = "cases")
lns2=ax3.plot(days_from_22_Jan_20, confirmed_r, "o-", color="green",label = "recovered")
lns3=ax3.plot(days_from_22_Jan_20, confirmed_d, "o-", color="blue",label = "deaths")
lns10=ax1.bar(days_from_22_Jan_20, day_confirmed, label = "day_cases")

lns_ax3 = lns1 +lns2+lns3
labs_ax3 = [l.get_label() for l in lns_ax3]
ax3.legend(lns_ax3, labs_ax3, loc=0)

ax1.legend(loc=3)

ax3.set_title(city +" ; {} cases, {} recovered, {} deaths".format(t_cases,t_recover,t_deaths))
ax3.set_xlabel("days from 22, Jan, 2020")
ax3.set_ylabel("cases, recovered,deaths ")
ax1.set_ylabel("day_cases")
ax1.set_xlim(0,100)
ax3.set_ylim(0,)
ax1.set_ylim(0,)
#ax3.set_yscale('log')

ax1.grid()

plt.pause(1)
plt.savefig('./fig/original_data_{}_{}linear.png'.format(city,daystamp)) 
plt.close() 

