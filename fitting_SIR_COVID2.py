#https://qiita.com/Student-M/items/4e3e286bf08b7320b665

#include package
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import pandas as pd

#pandasでCSVデータ読む。
data = pd.read_csv('COVID-19/csse_covid_19_data/csse_covid_19_time_series426/time_series_covid19_confirmed_global.csv')
data_r = pd.read_csv('COVID-19/csse_covid_19_data/csse_covid_19_time_series426/time_series_covid19_recovered_global.csv')
data_d = pd.read_csv('COVID-19/csse_covid_19_data/csse_covid_19_time_series426/time_series_covid19_deaths_global.csv')

    
confirmed = [0] * (len(data.columns) - 4)
confirmed_r = [0] * (len(data_r.columns) - 4)
confirmed_d = [0] * (len(data_d.columns) - 4)
days_from_22_Jan_20 = np.arange(0, len(data.columns) - 4, 1)

#City
#city = "Hubei"
city = "Korea, South"
city,beta,gamma,N0 ="Korea, South",3.02108000e-01, 3.10338816e-02, 1.49018020e+04
city,beta,gamma,N0 ="Italy",2.28556230e-01, 5.29960566e-02, 1.64008081e+05
city,beta,gamma,N0 ="Spain",2.28556230e-01, 5.29960566e-02, 1.64008081e+05
#city = "Iran"
#city = "Japan" #beta,gamma = 5.09688288e-06, 4.73538801e-02
city,beta,gamma,N0 ="Japan",2.28556230e-01, 5.29960566e-02, 2.64008081e+04
city,beta,gamma,N0 ="Thailand",1.73744816e-01, 5.46705107e-02, 5.50388554e+03
city,beta,gamma,N0 ="Germany",2.28556230e-01, 5.29960566e-02, 1.64008081e+05 #"Germany"
#city = "Bahrain"
#city,beta,gamma,N0 ="Switzerland",2.28556230e-01, 5.29960566e-02, 3.64008081e+04
#city = "United Kingdom"
#city = "Diamond Princess"
#city ="Sweden"
#city,beta,gamma,N0 ="US",2.24107762e-01, 5.15474878e-02, 1.66445545e+06

f1,f2 = 1,0 #data & data_r fitting factor

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
            
#define differencial equation of sir model
def sir_eq(v,t,beta,gamma,N):
    a = -beta*v[0]*v[1]/N
    b = beta*v[0]*v[1]/N - gamma*v[1]
    c = gamma*v[1]
    return [a,b,c]

def estimate_i(ini_state,beta,gamma,N):
    v=odeint(sir_eq,ini_state,t,args=(beta,gamma,N))
    est=v[0:int(t_max/dt):int(1/dt)] 
    return est[:,0],est[:,1],est[:,2] #v0-S,v1-I,v2-R

#define logscale likelihood function
def y(params):
    est_i_0,est_i_1,est_i_2=estimate_i(ini_state,params[0],params[1],params[2])
    return np.sum(f1*(est_i_1-obs_i_1)*(est_i_1-obs_i_1)+f2*(est_i_2-obs_i_2)*(est_i_2-obs_i_2))

#solve seir model

N,S0,I0,R0=int(N0*1),int(N0*1),int(1),int(0)
ini_state=[S0,I0,R0]

t_max=len(days_from_22_Jan_20)
dt=0.01
t=np.arange(0,t_max,dt)
plt.plot(t,odeint(sir_eq,ini_state,t,args=(beta,gamma,N))) #0.0001,1,3
plt.legend(['Susceptible','Infected','Recovered'])

obs_i_1 = confirmed
obs_i_2 = confirmed_r

plt.plot(obs_i_1,"o", color="red",label = "data_1")
plt.plot(obs_i_2,"o", color="green",label = "data_2")
plt.legend()
plt.pause(1)
plt.close()

#optimize logscale likelihood function
mnmz=minimize(y,[beta,gamma,N],method="nelder-mead")
print(mnmz)
N=mnmz.x[2]
S1=N
beta,gamma,N = mnmz.x[0],mnmz.x[1],mnmz.x[2] #感染率、除去率（回復率）
print("S1,S0=",S1,S0)

while np.abs(S1-S0)*100/S0>0.1:
    ini_state=[S0,I0,R0]
    mnmz=minimize(y,[beta,gamma,N],method="nelder-mead")
    S1=S0
    beta,gamma,N = mnmz.x[0],mnmz.x[1],mnmz.x[2] #感染率、除去率（回復率）
    S0=N
    print(mnmz)

#N_total = S_0+I_0+R_0
#R0 = N_total*beta_const *(1/gamma_const)
beta_const,gamma,N = mnmz.x[0],mnmz.x[1],mnmz.x[2] #感染率、除去率（回復率）

print(beta_const,gamma,N)
r0 = beta_const*(1/gamma)
print(r0)

#plot reult with observed data
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(1.6180 * 4, 4*2))
ax3 = ax1.twinx()
ax4 = ax2.twinx()

lns1=ax1.semilogy(confirmed,"o", color="red",label = "data_I")
lns2=ax1.semilogy(confirmed_r,"o", color="green",label = "data_R")
est_i_0,est_i_1,est_i_2=estimate_i(ini_state,mnmz.x[0],mnmz.x[1],mnmz.x[2])
print("N,est_i_1[-1],est_i_2[-1]=",N,est_i_1[-1],est_i_2[-1])
print("N-est_i_1[-1]-est_i_2[-1]=",N-est_i_1[-1]-est_i_2[-1])
lns3=ax1.semilogy(est_i_1, label = "estimation_I")
lns0=ax1.semilogy(est_i_2, label = "estimation_R")
lns_1=ax3.plot((N-est_i_1-est_i_2)*r0/N,".", color="black", label = "effective_R0")
#S-I-R
lns_4=ax3.plot(gamma*((N-est_i_1-est_i_2)*r0/N-1),".", color="blue", label = "g*(R-1)")
ax3.set_ylim(-1,)
ax1.set_yscale('log')

lns_ax1 = lns1+lns2 + lns3 + lns0
labs_ax1 = [l.get_label() for l in lns_ax1]
ax1.legend(lns_ax1, labs_ax1, loc=0)
ax1.set_ylim(1,)
ax1.set_title('SIR_{} f1_{:,d} f2_{:,d};N={:.0f} S0={:.0f} I0={:.0f} R0={:.0f} R={:.2f}'.format(city,f1,f2,N,S0,I0,R0,(N-est_i_1[-1]-est_i_2[-1])*r0/N))
#ax1.set_title('SIR_{} f1_{:,d} f2_{:,d};N={:.0f} S0={:.0f} I0={:.0f} R0={:.0f} g*(R-1)={:.2f}'.format(city,f1,f2,N,S0,I0,R0,gamma*((N-est_i_1[-1]-est_i_2[-1])*r0/N-1)))
ax1.set_ylabel("Susceptible, Infected, recovered ")
ax3.set_ylabel("effective_R0")
ax3.legend()

t_max=200 #len(days_from_22_Jan_20)
t=np.arange(0,t_max,dt)

lns4=ax2.plot(confirmed,"o", color="red",label = "data")
lns5=ax2.plot(confirmed_r,"o", color="green",label = "data_r")
est_i_0,est_i_1,est_i_2=estimate_i(ini_state,mnmz.x[0],mnmz.x[1],mnmz.x[2])
lns6=ax2.plot(est_i_0, label = "estimation_S")
lns7=ax2.plot(est_i_1, label = "estimation_I")
lns8=ax2.plot(est_i_2, label = "estimation_R")
lns_6=ax4.plot((S0-est_i_1-est_i_2)*r0/N,".", color="black", label = "effective_R0")

lns_ax2 = lns4+lns5 + lns6 + lns7+ lns8 +lns_6
labs_ax2 = [l.get_label() for l in lns_ax2]
ax2.legend(lns_ax2, labs_ax2, loc=0)
ax4.set_ylim(0,)
ax2.set_ylim(1,)
ax2.set_title('SIR_{};b={:.2e} g={:.2e} r0={:.2f}'.format(city,beta_const,gamma,r0))
ax2.set_ylabel("Susceptible, Infected, recovered ")
ax4.set_ylabel("effective_R0")
#ax2.set_yscale('log')


ax1.grid()
ax2.grid()
plt.savefig('./fig/SIR_N0{}_{}f1_{:,d}f2_{:,d};b_{:.2e}g_{:.2e}r0_{:.2f}N_{:.0f}S0_{:.0f}I0_{:.0f}R0_{:.0f}.png'.format(city,N0,f1,f2,beta_const,gamma,r0,N,S0,I0,R0)) 
plt.show()
plt.close()
