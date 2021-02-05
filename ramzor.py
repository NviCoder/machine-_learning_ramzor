import pandas as pd
import numpy as np
import math


def get_rank(N, P, G):
    k, m = 2.0, 8.0
    if (pd.isna(N) or pd.isna(G)):
        NGG=1
    elif (N==0 or G==0):
        NGG=1
    elif (math.isinf(N) or math.isinf(G)):
        NGG = math.exp(1)
    else:
        NGG = N*G*G
    P = 0 if pd.isna(P) else min(P,1.0)
    score = k + math.log(NGG) + P/m
    return min(max(score,2.0),10.0)


csv_ver30122020 = False
if csv_ver30122020:
    data_covid = pd.read_csv("corona_city_table_ver_0028.csv", sep=",", header=0)
    data_covid.drop(["City_Name", "Cumulated_recovered", "Cumulated_deaths", "Cumulated_number_of_diagnostic_tests"],
                    axis='columns', inplace=True)
else:
    data_covid = pd.read_csv("corona_city_table_ver_03022021.csv", sep=",", header=0)
    data_covid.drop(["City_Name", "Cumulated_recovered", "Cumulated_deaths", "Cumulated_number_of_diagnostic_tests", "colour","final_score"],
                    axis='columns', inplace=True)

data_covid["Date"] = pd.to_datetime(data_covid["Date"], dayfirst=True)
data_covid["Cumulative_verified_cases"] = pd.to_numeric(data_covid["Cumulative_verified_cases"])
data_covid[["Ni","Pi","Gi","Rank"]] = -1.0

data_pop = pd.read_csv("population.csv", sep=",", header=0)

#drop cities with no data about population
data_pop = data_pop.dropna(subset=['popTot'])
data_covid = data_covid.drop(data_covid[~data_covid.City_Code.isin(data_pop.city_code)].index)
data_covid.reset_index(drop=True, inplace=True)


for index, row in data_covid.iterrows():
    current_city_code = row['City_Code']
    current_city_pop = data_pop.query('city_code == @current_city_code')['popTot'].values[0]
    date_week_ago = row['Date'] - pd.DateOffset(days=7)
    row_week_ago = data_covid.query('Date == @date_week_ago & City_Code == @current_city_code')
    if row_week_ago.empty:
        data_covid.at[index, 'Ni'] = pd.NA
        data_covid.at[index, 'Pi'] = pd.NA
        data_covid.at[index, 'Gi'] = pd.NA
    else:
        week_new_verified = data_covid.at[index, 'Cumulative_verified_cases'] - row_week_ago.iloc[0]['Cumulative_verified_cases']
        data_covid.at[index, 'Ni'] = week_new_verified/(current_city_pop / 10000.0)
        data_covid.at[index, 'Pi'] = week_new_verified / (data_covid.at[index, 'Cumulated_number_of_tests'] - row_week_ago.iloc[0]['Cumulated_number_of_tests'])
        #set Gi
        Ni_week_ago = row_week_ago.iloc[0]['Ni']
        if pd.isna(Ni_week_ago):
            data_covid.at[index, 'Gi'] = pd.NA
        else:
            # this '2' is a guess for suitable value
            data_covid.at[index, 'Gi'] = 2 if Ni_week_ago==0 else (week_new_verified/(current_city_pop / 10000.0) ) /  Ni_week_ago

    #set ramzor rank
    data_covid.at[index, 'Rank'] = get_rank(N=data_covid.at[index, 'Ni'], P=data_covid.at[index, 'Pi'], G=data_covid.at[index, 'Gi'])

if csv_ver30122020:
    data_covid.to_csv('ramzor.csv', sep=',')
else:
    data_covid.to_csv('ramzor2.csv', sep=',')