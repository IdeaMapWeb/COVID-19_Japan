#https://qiita.com/Student-M/items/4e3e286bf08b7320b665

#include package
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import pandas as pd

daystamp='430'
#pandasでCSVデータ読む。
data = pd.read_csv('COVID-19/csse_covid_19_data/csse_covid_19_time_series'+daystamp+'/time_series_covid19_confirmed_global.csv')
data_r = pd.read_csv('COVID-19/csse_covid_19_data/csse_covid_19_time_series'+daystamp+'/time_series_covid19_recovered_global.csv')
data_d = pd.read_csv('COVID-19/csse_covid_19_data/csse_covid_19_time_series'+daystamp+'/time_series_covid19_deaths_global.csv')

    
confirmed = [0] * (len(data.columns) - 4)
confirmed_r = [0] * (len(data_r.columns) - 4)
confirmed_d = [0] * (len(data_d.columns) - 4)
day_confirmed = [0] * (len(data_d.columns) - 4)
days_from_22_Jan_20 = np.arange(0, len(data.columns) - 4, 1)
day_est_i_1=[0] * (len(data.columns) - 4)

#City
#city = "Hubei"
city,beta,gamma,delta,N0 ="Japan",5.66349161e-06, 4.30639119e-02, 3.81651749e-02, 13929 #2.64008081e+04
#city,beta,gamma,delta,N0 ="Korea, South",3.02108000e-05,3.02108000e-02, 3.10338816e-02, 10780 #1.49018020e+04 #10738 #1.49018020e+04
#city,beta,gamma,delta,N0 ="Italy",2.28556230e-06, 5.29960566e-02, 5.29960566e-02, 207428
#xcity,beta,delta,gamma,N0 ="Spain",2.28556230e-01,0.001 ,5.29960566e-02, 1.74008081e+05 
city,beta,gamma,delta,N0 ="Thailand",3.67004495e-05, 8.47181334e-02, 6.17333779e-02, 5.75491592e+03
#city,beta,gamma,delta,N0 ="Germany",1.71086885e-06, 5.73654635e-02, 2.72319310e-01, 1.61477390e+05 
#city = "Bahrain"
#city,beta,gamma,N0 ="Switzerland",2.28556230e-05, 5.29960566e-02, 29705
#city = "United Kingdom"
#city = "Diamond Princess"
#city ="Sweden"
city,beta,delta,gamma,N0 ="US",2.24107762e-07, 2.15474878e-02,5.15474878e-02, 1.26445545e+06
#city,beta,delta,gamma,N0 ="world",2.24107762e-07, 2.15474878e-02,5.15474878e-02, 3340820
#city,beta,gamma,delta,N0 ="Russia",2.28556230e-06, 5.29960566e-02, 5.29960566e-02, 207428*1.5
#city,beta,gamma,delta,N0 ="India",2.28556230e-06, 5.29960566e-02, 5.29960566e-02, 37257*1.6

f1,f2 = 1,1 #data & data_r fitting factor

#データを加工する
t_cases = 0

for i in range(0, len(data_r), 1):
    if (data_r.iloc[i][1] == city): #for country/region
    #if (data_r.iloc[i][0] == city):  #for province:/state  
        print(str(data_r.iloc[i][0]) + " of " + data_r.iloc[i][1])
        for day in range(4, len(data.columns), 1):            
            confirmed_r[day - 4] += data_r.iloc[i][day]
            #t_recover += data_r.iloc[i][day]
for i in range(0, len(data_d), 1):
    if (data_d.iloc[i][1] == city): #for country/region
    #if (data_d.iloc[i][0] == city):  #for province:/state  
        print(str(data_d.iloc[i][0]) + " of " + data_d.iloc[i][1])
        for day in range(4, len(data.columns), 1):
            confirmed_d[day - 4] += data_d.iloc[i][day] #fro drawings
            #t_deaths += data_d.iloc[i][day]            
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
            
#define differencial equation of sir model
def sir_eq(v,t,beta,gamma,delta):
    a = -beta*v[0]*v[1]
    b = beta*v[0]*v[1] - gamma*v[1]
    c = gamma*v[1]-delta*v[2]
    d = delta*v[2]
    return [a,b,c,d]

def estimate_i(ini_state,beta,gamma,delta):
    v=odeint(sir_eq,ini_state,t,args=(beta,gamma,delta))
    est=v[0:int(t_max/dt):int(1/dt)] 
    return est[:,0],est[:,1],est[:,2],est[:,3] #v0-S,v1-I,v2-H v3-R

#define logscale likelihood function
def y(params):
    est_i_0,est_i_1,est_i_2,est_i_3=estimate_i(ini_state,params[0],params[1],params[2])
    return np.sum(f1*(est_i_1-obs_i_1)*(est_i_1-obs_i_1)+f2*(est_i_3-obs_i_2)*(est_i_3-obs_i_2))

#solve seir model
fc=1.2
N,S0,I0,H0,R0=int(N0*fc),int(N0*fc),int(1),int(0),int(0)
ini_state=[S0,I0,H0,R0]

t_max=len(days_from_22_Jan_20)
dt=0.01
t=np.arange(0,t_max,dt)

fig, ax1 = plt.subplots(1,1,figsize=(1.6180 * 4, 4*1))
ax2 = ax1.twinx()
lns1=ax1.plot(t,odeint(sir_eq,ini_state,t,args=(beta,gamma,delta))) #0.0001,1,3
ax1.legend(['Susceptible','Infected','Hospital','Recovered'])

obs_i_1 = confirmed
obs_i_2 = confirmed_r

lsn2=ax1.plot(obs_i_1,"o", color="red",label = "data_1")
lns3=ax1.plot(obs_i_2,"o", color="green",label = "data_2")
ax2.legend()
plt.pause(2)
plt.savefig('./fig/SIHR_{}0.png'.format(city)) 
plt.close()

#optimize logscale likelihood function
mnmz=minimize(y,[beta,gamma,delta],method="nelder-mead")
print(mnmz)
beta,gamma,delta = mnmz.x[0],mnmz.x[1],mnmz.x[2] #感染率、入院率、除去率（回復率）

"""
while np.abs(S1-S0)*100/S0>1:
    ini_state=[S0,I0,H0,R0]
    mnmz=minimize(y,[beta,gamma,delta,N],method="nelder-mead")
    S1=S0
    beta,gamma,delta,N = mnmz.x[0],mnmz.x[1],mnmz.x[2],mnmz.x[3] #感染率、除去率（回復率）
    S0=N
    print(mnmz)
"""
#N_total = S_0+I_0+R_0
#R0 = N_total*beta_const *(1/gamma_const)
beta_const,gamma,delta = mnmz.x[0],mnmz.x[1],mnmz.x[2] #感染率、除去率（回復率）

print(beta_const,gamma,delta)
r0 = S0*beta_const*(1/gamma)
print(r0)

#plot reult with observed data
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(1.6180 * 4, 4*2))
ax3 = ax1.twinx()
ax4 = ax2.twinx()

lns1=ax1.semilogy(confirmed,"o", color="red",label = "data_I")
lns2=ax1.semilogy(confirmed_r,"o", color="green",label = "data_R")
lns3=ax1.semilogy(confirmed_d,"o", color="blue",label = "data_D")
est_i_0,est_i_1,est_i_2,est_i_3=estimate_i(ini_state,mnmz.x[0],mnmz.x[1],mnmz.x[2])
print("S0,est_i_1[-1],est_i_2[-1],est_i_3[-1]=",S0,est_i_1[-1],est_i_2[-1],est_i_3[-1])
print("S0-est_i_1[-1]-est_i_2[-1]-est_i_3[-1]=",S0-est_i_1[-1]-est_i_2[-1]-est_i_3[-1])
lns4=ax1.semilogy(est_i_0, label = "estimation_S")
lns5=ax1.semilogy(est_i_1, label = "estimation_I")
lns6=ax1.semilogy(est_i_2, label = "estimation_H")
lns7=ax1.semilogy(est_i_3, label = "estimation_R")
lns8=ax1.bar(days_from_22_Jan_20,day_confirmed, label = "day_confirmed")
lns_1=ax3.plot((S0-est_i_1-est_i_2-est_i_3)*r0/S0,".", color="black", label = "R")
ax3.set_ylim(0,)
ax1.set_yscale('log')

lns_ax1 = lns1 + lns2 + lns3 + lns4 +lns5 + lns6 +lns7
labs_ax1 = [l.get_label() for l in lns_ax1]
ax1.legend(lns_ax1, labs_ax1, loc=0)
ax1.set_ylim(1,)
ax1.set_title('SIHR_{} f1_{:,d} f2_{:,d}; S0={:.0f} g*(R-1)={:.2e} r0={:.2f} R={:.2f}'.format(city,f1,f2,S0,gamma*((S0-est_i_1[-1]-est_i_2[-1]-est_i_3[-1])*r0/S0-1),r0,(S0-est_i_1[-1]-est_i_2[-1]-est_i_3[-1])*r0/S0))
ax1.set_ylabel("Susceptible, Infected, recovered ")
ax3.set_ylabel("R")
ax3.legend()

t_max=150 #len(days_from_22_Jan_20)
t=np.arange(0,t_max,dt)
day_est_i_1=[0] * (t_max)

lns4=ax2.plot(confirmed,"o", color="red",label = "data")
lns5=ax2.plot(confirmed_r,"o", color="green",label = "data_r")
est_i_0,est_i_1,est_i_2,est_i_3=estimate_i(ini_state,mnmz.x[0],mnmz.x[1],mnmz.x[2])
lns6=ax2.plot(est_i_0, label = "estimation_S")
lns7=ax2.plot(est_i_1, label = "estimation_I")
lns8=ax2.plot(est_i_2, label = "estimation_H")
lns9=ax2.plot(est_i_3, label = "estimation_R")
#lns10=ax4.plot((S0-est_i_1-est_i_2-est_i_3)*r0/S0,".", color="black", label = "effective_R0")
lns_10=ax4.plot(gamma*((S0-est_i_1-est_i_2-est_i_3)*r0/S0-1),".", color="blue", label = "g*(R-1)")

lns_ax2 = lns4+lns5 + lns6 + lns7+ lns8+ lns9 #+lns10
labs_ax2 = [l.get_label() for l in lns_ax2]
ax2.legend(lns_ax2, labs_ax2, loc=0)
ax4.legend()
ax4.set_ylim()
ax2.set_ylim(1,)
ax2.set_title('SIHR_{};b={:.2e} g={:.2e} d={:.2e} r0={:.2f}'.format(city,beta_const,gamma,delta,r0))
ax2.set_ylabel("Susceptible, Infected, recovered ")
ax4.set_ylabel("g*(R-1)")
#ax2.set_yscale('log')

ax1.grid()
ax2.grid()
plt.savefig('./fig/SIHR_{}f1_{:,d}f2_{:,d};b_{:.2e}d_{:.2e}g_{:.2e}r0_{:.2f}S0_{:.0f}I0_{:.0f}R0_{:.0f}_.png'.format(city,f1,f2,beta_const,gamma,delta,r0,S0,I0,R0)) 
plt.pause(1)
plt.close()

fig, ax1 = plt.subplots(1,1,figsize=(1.6180 * 4, 4*1))
ax2 = ax1.twinx()
#lns1=ax2.plot(t,odeint(sir_eq,ini_state,t,args=(mnmz.x[0],mnmz.x[1],mnmz.x[2])))
est_i_0,est_i_1,est_i_2,est_i_3=estimate_i(ini_state,mnmz.x[0],mnmz.x[1],mnmz.x[2])
for day in range(0, len(est_i_1)-1, 1):
    if day == 0:
         day_est_i_1[day] += est_i_1[day] + est_i_2[day]+ est_i_3[day]
    else:
         day_est_i_1[day] += est_i_1[day] +  est_i_2[day]+  est_i_3[day] - est_i_1[day-1] -  est_i_2[day-1]- est_i_3[day-1]
lns5=ax1.plot(day_est_i_1,".",color="red", label = "day_est")
lns6=ax2.plot(est_i_0, label = "estimation_S")
lns7=ax2.plot(est_i_1, label = "estimation_I")
lns8=ax2.plot(est_i_2, label = "estimation_H")
lns9=ax2.plot(est_i_3, label = "estimation_R")

obs_i_1 = confirmed
obs_i_2 = confirmed_r
lsn2=ax2.plot(obs_i_1,"o", color="red",label = "cases")
lns3=ax2.plot(obs_i_2,"o", color="green",label = "recovered")
lns4=ax2.plot(confirmed_d,"o", color="blue",label = "deaths")
lns10=ax1.bar(days_from_22_Jan_20,day_confirmed, label = "day_cases")
ax1.set_title('SIHR_{}N={}_b={:.2e}_g={:.2e}_d={:.2e}_r0={:.2f}'.format(city,S0,beta_const,gamma,delta,r0))
lns_ax2= lns2 +lns3 +lns4 +lns6 +lns7 +lns8 +lns9
labs_ax2 = [l.get_label() for l in lns_ax2]
ax2.legend(lns_ax2, labs_ax2, loc=2)
ax1.legend(loc=1)
ax2.set_ylabel("Susceptible, Infected, recovered ")
ax1.set_ylabel("day_comfirmed")
ax2.set_yscale('log')
ax2.set_ylim(1,)
ax2.grid()
plt.pause(1)
plt.savefig('./fig/SIHR_{}_f1={}f2={}s0={}.png'.format(city,f1,f2,S0)) 
plt.close()