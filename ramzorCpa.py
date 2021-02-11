#Ramzor 
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler


#Clean the population.csv file
data_population = pd.read_csv("population.csv", sep=",", header=0)
data_population.drop(["nameH","nameE1", "mahoz", "napa" ,"ezor tivee","maamad","shiyuch"
                     ,"Yisood", "tzoora", "irgun", "vaada", "police", "year", "nameE2", "eshcol","height","popJewish"],axis='columns', inplace=True)

#drop enpty rows
#rows that we have at the csv file                                   0123456789
#*city_code*	*dat*	*popTot* *popNotArab* *popArab*	*coordinates*1234512345

data_population = data_population.dropna(subset=['city_code', 'popTot', 'dat', 'coordinates'])

#Add new row for distance
data_population['coordinates1']=""
data_population['coordinates2']=""
#data_population['distance']=0.0

#Split coordinates for each city
for index, row in data_population.iterrows():
    current_city_popNotArab = row['popNotArab']
    current_city_popArab= row['popArab']
    print(current_city_popArab)
    if math.isnan(current_city_popNotArab) or current_city_popNotArab == "":
        if math.isnan(current_city_popArab) or current_city_popArab == "": #current_city_popNotArab =0 current_city_popArab=0
            data_population.drop([index,index])
            continue
        else: #current_city_popNotArab =0 current_city_popArab=1
            data_population.at[index, 'popNotArab'] = 0.0
    else:
         if math.isnan(current_city_popArab) or current_city_popArab == "":#current_city_popNotArab =1 current_city_popArab=0
             data_population.at[index, 'popArab'] = 0.0
        #else - current_city_popNotArab =1 current_city_popArab=1
            
    current_city_coordinates = row['coordinates']                                           
    data_population.at[index, 'coordinates1'] = str(current_city_coordinates)[0:5]
    data_population.at[index, 'coordinates2'] = str(current_city_coordinates)[5:10]      

#Drop coordinates colomn
data_population.drop(["coordinates", "popTot"],axis='columns', inplace=True)
data_population.set_index('city_code', inplace=True)


#normalize data
sclr = MinMaxScaler()
for col in data_population.head():
    data_population[col] = sclr.fit_transform(data_population[[col]])

#Generate the csv file
data_population.to_csv('population_for_cpa.csv', sep=',')
print("success!")

