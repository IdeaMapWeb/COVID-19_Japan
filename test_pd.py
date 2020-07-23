import pandas as pd

test0 = pd.read_csv('data/covid19/test_confirmed.csv',encoding="cp932")
test1 = pd.read_csv('data/covid19/test_recovered.csv',encoding="cp932")
test2 = pd.read_csv('data/covid19/test_deaths.csv',encoding="cp932")
day_list={531,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,630,701,702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720}
print(day_list)
#day=[]
for day in day_list:
    #original data input
    print(day)
    data = pd.read_csv('data/covid19/{}.csv'.format(day),encoding="cp932")
    #test0 = pd.read_csv('COVID-19/csse_covid_19_data/japan/test_confirmed.csv',encoding="cp932")
    data.to_csv('data/covid19/tokyo_confirmed.csv',  columns=['Prefecture','cases'], index=False)
    data.to_csv('data/covid19/tokyo_recovered.csv',  columns=['Prefecture','recovered'], index=False)
    data.to_csv('data/covid19/tokyo_deaths.csv',  columns=['Prefecture','deaths'], index=False)

    print(data)

    #
    test0_ = pd.read_csv('data/covid19/tokyo_confirmed.csv')
    test1_ = pd.read_csv('data/covid19/tokyo_recovered.csv')
    test2_ = pd.read_csv('data/covid19/tokyo_deaths.csv')

    print(test0)
    #列の追加
    s=str(day)

    test0['{}'.format(s)] = test0_['cases']
    test1['{}'.format(s)] = test1_['recovered']
    test2['{}'.format(s)] = test2_['deaths']

    print(test0)

    test0.to_csv('data/covid19/test_confirmed.csv',index=False)
    test1.to_csv('data/covid19/test_recovered.csv',index=False)
    test2.to_csv('data/covid19/test_deaths.csv',index=False)
