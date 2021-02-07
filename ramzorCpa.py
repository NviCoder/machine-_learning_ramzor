#Ramzor 
import pandas as pd
import numpy as np
import math
#from sklearn.decomposition import PCA
#from sklearn.pipeline import make_pipeline

#distances for city X
CITY_NAME = "TEL AVIV - YAFO"
DAT = 4
COORDINATES = 1802766486 
HEIGHT = 17


#disnatce function
def get_distance(dat, popTot, coordinates, height):
    return HEIGHT - height


#Clean the population.csv file
data_population = pd.read_csv("population.csv", sep=",", header=0)
data_population.drop(["nameH", "city_code", "mahoz", "napa" ,"ezor tivee","maamad","shiyuch"
                     ,"Yisood", "tzoora", "irgun", "vaada", "police", "year", "nameE2", "eshcol"],axis='columns', inplace=True)

#drop enpty rows
#rows that we have at the csv file
#*nameE1*	*dat*	*popTot*	*popNotArab*	*popJewish*	*popArab*	*coordinates*	*height*
data_population = data_population.dropna(subset=['dat'])

#Add new row for distance
data_population['distance']=0.0

#calculate the dis
for index, row in data_population.iterrows():
    
    current_city_height = row['height']
    if current_city_height == " ":
        data_population.at[index, 'distance'] = pd.NA
    else:                                                   
        data_population.at[index, 'distance'] = get_distance(dat=data_population.at[index, 'dat'], popTot=data_population.at[index, 'popTot'],
          coordinates=data_population.at[index, 'coordinates'], height=data_population.at[index, 'height'])





data_population.to_csv('population_for_cpa.csv', sep=',')
print("sucsses!")

