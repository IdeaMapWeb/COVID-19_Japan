import pandas as pd

test0 = pd.read_csv('COVID-19/csse_covid_19_data/japan/test_confirmed.csv') #,encoding="cp932")
test1 = pd.read_csv('COVID-19/csse_covid_19_data/japan/test_recovered.csv') #,encoding="cp932")
test2 = pd.read_csv('COVID-19/csse_covid_19_data/japan/test_deaths.csv') #,encoding="cp932")
day_list={326,327,328,329,331,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418}
#day=[]
for day in day_list:
    #original data input
    data = pd.read_csv('COVID-19/csse_covid_19_data/japan/{}.csv'.format(day),encoding="cp932")
    #test0 = pd.read_csv('COVID-19/csse_covid_19_data/japan/test_confirmed.csv',encoding="cp932")
    data.to_csv('COVID-19/csse_covid_19_data/japan/tokyo_confirmed.csv',  columns=['Region','cases'], index=False)
    data.to_csv('COVID-19/csse_covid_19_data/japan/tokyo_recovered.csv',  columns=['Region','recovered'], index=False)
    data.to_csv('COVID-19/csse_covid_19_data/japan/tokyo_deaths.csv',  columns=['Region','deaths'], index=False)

    print(data)

    #
    test0_ = pd.read_csv('COVID-19/csse_covid_19_data/japan/tokyo_confirmed.csv')
    test1_ = pd.read_csv('COVID-19/csse_covid_19_data/japan/tokyo_recovered.csv')
    test2_ = pd.read_csv('COVID-19/csse_covid_19_data/japan/tokyo_deaths.csv')

    print(test0)
    #列の追加
    s=str(day)

    test0['{}'.format(s)] = test0_['cases']
    test1['{}'.format(s)] = test1_['recovered']
    test2['{}'.format(s)] = test2_['deaths']

    print(test0)

    test0.to_csv('COVID-19/csse_covid_19_data/japan/test_confirmed.csv',index=False)
    test1.to_csv('COVID-19/csse_covid_19_data/japan/test_recovered.csv',index=False)
    test2.to_csv('COVID-19/csse_covid_19_data/japan/test_deaths.csv',index=False)
