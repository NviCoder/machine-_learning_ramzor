#Ramzor 
import pandas as pd
import numpy as np
import math
#from sklearn.decomposition import PCA
#from sklearn.pipeline import make_pipeline


#distances for city X
CITY_CODE = "5000"
DAT = 4
COORDINATES1 = 18027
COORDINATES2 = 66486 

#Distance function
def l_2(x, y):
        return math.sqrt(sum(map(lambda a, b: (a - b) ** 2, x, y)))


#Clean the population.csv file
data_population = pd.read_csv("population.csv", sep=",", header=0)
data_population.drop(["nameH","nameE1", "mahoz", "napa" ,"ezor tivee","maamad","shiyuch"
                     ,"Yisood", "tzoora", "irgun", "vaada", "police", "year", "nameE2", "eshcol","height","popJewish"],axis='columns', inplace=True)

#drop enpty rows
#rows that we have at the csv file                                   0123456789
#*city_code*	*dat*	*popTot* *popNotArab* *popArab*	*coordinates*1234512345

data_population = data_population.dropna(subset=['city_code'])
data_population = data_population.dropna(subset=['popTot'])
data_population = data_population.dropna(subset=['dat'])
data_population = data_population.dropna(subset=['coordinates'])


#Add new row for distance
data_population['coordinates1']=""
data_population['coordinates2']=""
data_population['distance']=0.0


#Split coordinates for each city
for index, row in data_population.iterrows():
    
    current_city_popNotArab = row['popNotArab']
    current_city_popArab= row['popArab']
    print(current_city_popArab)
    if  math.isnan(current_city_popNotArab) or current_city_popNotArab == "":
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
data_population.drop(["coordinates"],axis='columns', inplace=True)

'''
#Exem the function
for index, row in data_population.iterrows():
    data_population.at[index, 'distance'] = l_2(COORDINATES1,row['coordinates1']) + l_2(COORDINATES2,row['coordinates2'])
'''

#Generate the csv file
data_population.to_csv('population_for_cpa.csv', sep=',')
print("sucsses!")

