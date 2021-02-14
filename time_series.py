import pandas as pd
import numpy as np
import math
import random
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

# select params
#city_code = 3000
window_size = 3
look_forward = 1
train_ratio = 0.75
knn = 1  # set -1 for all the cities
split_date = None
adaptive = False




models =["random forest", "linear regression", "svm"]

#model_type = "random forest"
#model_type = "linear regression"
#model_type = "svm"
#model_type = "lstm"


def select_model(model_type):
    if model_type == "random forest":
        return RandomForestRegressor(n_estimators=100, criterion='mse',
                                     min_samples_split=math.ceil(math.log2(trainX.size)))
    elif model_type == "linear regression":
        return LinearRegression()
    elif model_type == "svm":
        return SVR(kernel='rbf', C=100, gamma='scale', epsilon=.1)
    elif model_type == "lstm":
        # create LSTM network
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, window_size)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    print("model_type error")
    return None


# read data set
ds = pd.read_csv("ramzor2.csv", sep=",", header=0)
proc_pop = pd.read_csv("population_for_cpa.csv", sep=",", header=0)
proc_pop = proc_pop.drop(
    proc_pop[~proc_pop.city_code.isin(ds.City_Code)].index)
proc_pop.set_index('city_code', inplace=True)

#for random cities
def get_all_cities_list():
    return proc_pop.index.values.tolist()

#Random cities
all_cities = get_all_cities_list()
random_cities_list = random.sample(all_cities,20)
print(random_cities_list)



def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def split_train_test(data, train_ratio, look_forward):
    """
    Split to train and test by ratio.
    Arguments:
        data: Sequence of observations.
        train_ratio: Ratio of observations as train.
        look_forward: Amount of steps in Y
    Returns:
        Pandas DataFrames of trainX, trainy, testX, testy.
    """
    X = data.values
    train_size = int(len(X) * train_ratio)
    train, test = X[0:train_size], X[train_size:len(X)]
    # trainX, trainy, testX, testy
    return train[:, :-look_forward], train[:, -1], test[:, :-look_forward], test[:, -1]

    # return train[:, :-1], train[:, -1], test[:, :-1], test[:, -1] #trainX, trainy, testX, testy


def filter_by_city(city_code): return ds[ds['City_Code'] == city_code].copy()


def predict_static(model, testX):
    """
    :param model: Learning model, with predict function
    :param testX: List of data to predict
    :return: List of predictions
    """

    predictions = list()
    if model_type != "lstm":
        for i in range(len(testX)):
            predictions.append(model.predict(testX[i].reshape(1, -1)))
    else:
        predictions = model.predict(testX)
    return predictions


def predict_adaptive(model, trainX, trainy, testX, testy):
    """
    After each prediction the model learn the last data
    :param model: Learning model, with predict and fit function
    :param trainX: train's features
    :param trainy: train's values
    :param testX: test's features
    :param testy: test's values
    :return: List of predictions
    """
    predictions = list()
    for i in range(len(testX)):
        predictions.append(model.predict(testX[i].reshape(1, -1)))
        trainX = np.concatenate((trainX, testX[i].reshape(1, -1)), axis=0)
        trainy = np.append(trainy, testy[i])
        model.fit(trainX, trainy)
    return predictions


class AsLastColumn:
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self, x):
        return x[:, -1]


def one_city_train_test(city_code):
    ds_city = filter_by_city(city_code)
    ds_city.drop(['index', 'City_Code', 'Cumulative_verified_cases',
                  'Cumulated_number_of_tests', 'Ni', 'Pi', 'Gi'], axis='columns', inplace=True)

    # First time: set split date by the ratio. More times: set ratio by split date.
    global split_date
    if (split_date == None):
        split_date = ds_city.iloc[int(len(ds_city) * train_ratio)]['Date']
        current_ratio = train_ratio
    else:
        test_list = ds_city.index[ds_city['Date'] == split_date].to_list()
        if not test_list:
            current_ratio = 0
        else:
            fist_test_row = ds_city.index[ds_city['Date'] == split_date].to_list()[
                0]
            current_ratio = (
                fist_test_row - ds_city.index.to_list()[0]) / float(len(ds_city))

    ds_city.set_index('Date', inplace=True)
    ds_city = series_to_supervised(ds_city, window_size, look_forward)
    return split_train_test(ds_city, current_ratio, look_forward)


def all_cities_train():
    return train_by_cities_list(get_all_cities_list())


def l2_cities(city_code1, city_code2):
    vec1 = proc_pop.loc[city_code1]
    vec2 = proc_pop.loc[city_code2]
    return mean_squared_error(vec1, vec2)


def knn_cities_train(city_code, top_k):
    cities = get_all_cities_list()
    if top_k > len(cities):
        return all_cities_train()
    sort_by_dis = sorted(enumerate(cities), key=(
        lambda y: l2_cities(y[1], city_code)))
    head_of_list = [sort_by_dis[i][1] for i in range(top_k)]
    print(top_k, "nearest neighbor cities:", head_of_list)
    return train_by_cities_list(head_of_list)


def train_by_cities_list(cities):
    trainX, trainy = np.empty(shape=(0, window_size)), np.empty(shape=(1, 0))
    for city in cities:
        trainX_city, trainy_city, _, _ = one_city_train_test(city)
        trainX = np.concatenate((trainX, trainX_city), axis=0)
        trainy = np.append(trainy, trainy_city)
    return trainX, trainy

#preperment for result
    '''

    ''' 
counter = 0
result = pd.DataFrame({
                    'model':[' '],
                    'number of cities': [str(len(random_cities_list))],
                    'window':[str(window_size)],
                    'look_forward':[str(look_forward)],
                    'train_ratio':[str(train_ratio)],
                    'knn':[str(knn)],
                    'adaptive':[str(adaptive)],
                    '   ====>  ':[' '],
                    'avg_error_model': ['0']
                        })

for model_type in models:
    print("model type: ",model_type)
    
    sum_error_model = 0
    sum_error_monkey  = 0

    for city_code in random_cities_list:
        
        # Split train-test
        trainX, trainy, testX, testy = one_city_train_test(city_code)
        if knn == -1:
            trainX, trainy = all_cities_train()
        elif knn > 1:
            trainX, trainy = knn_cities_train(city_code, knn)

        # Select model and learn
        model = select_model(model_type)

        if model_type == "lstm":
            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
            testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

            model.fit(trainX, trainy, epochs=25, batch_size=1, verbose=2)
        else:
            model.fit(trainX, trainy)
        if model_type == "random forest":
            # optional to RandomForestRegressor
            print('importances', model.feature_importances_)

        #Iteration number  + city code

        #print("Iteration: ",counter," For city: ",city_code)
        # prediction
        if adaptive:
            predictions = predict_adaptive(model, trainX, trainy, testX, testy)
        else:
            predictions = predict_static(model, testX)

        error = mean_squared_error(testy, predictions)
        sum_error_model = sum_error_model + error
        #print("model error:", error)
        

        # monkey model learning
        model_monkey = AsLastColumn()
        predictions_monkey = predict_static(model_monkey, testX)
        error_monkey = mean_squared_error(testy, predictions_monkey)
        sum_error_monkey = sum_error_monkey + error_monkey
        #print("monkey error:", error_monkey)
        
        print("sum_error_model: ", sum_error_model)
        # plot expected vs predicted vs monkey
        '''
        pyplot.plot(testy, label='Expected')
        pyplot.plot(predictions, label='Predicted')
        pyplot.plot(predictions_monkey, label='Monkey')
        pyplot.legend()
        pyplot.show()
        '''
    avg_error_model = sum_error_model / len(random_cities_list)
    avg_error_monkey = sum_error_monkey / len(random_cities_list)
    result.at[counter, 'avg_error_model'] = avg_error_model
    #result.at[counter, 'avg_error_monkey'] = avg_error_monkey
    result.at[counter, 'model'] = model_type


    counter = counter + 1
result.at[counter, 'model'] = "monkey"
result.at[counter, 'avg_error_model'] = avg_error_monkey


result.to_csv('result.csv', sep=',')
print("success!")