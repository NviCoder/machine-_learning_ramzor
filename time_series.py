import pandas as pd
import numpy as np
import math
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

#select params
city_code = 5000
window_size = 3
look_forward = 3
train_ratio = 0.75
split_date = None
#model_type = "random forest"
#model_type = "linear regression"
model_type = "svm"

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
    return train[:, :-look_forward], train[:, -1], test[:, :-look_forward], test[:, -1] #trainX, trainy, testX, testy
    #return train[:, :-1], train[:, -1], test[:, :-1], test[:, -1] #trainX, trainy, testX, testy
filter_by_city = lambda city_code: ds[ds['City_Code'] == city_code].copy()

def predict_static(model, testX):
    """
    :param model: Learning model, with predict function
    :param testX: List of data to predict
    :return: List of predictions
    """
    predictions = list()
    for i in range(len(testX)):
        predictions.append(model.predict(testX[i].reshape(1, -1)))
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
        return x[:,-1]

def read_pop():
    data_pop = pd.read_csv("population.csv", sep=",", header=0)
    # TODO drop cities not in ds_city
    return data_pop.dropna(subset=['popTot'])

def get_all_cities_list():
    return read_pop()['city_code']

def one_city_train_test(city_code):
    ds_city = filter_by_city(city_code)
    ds_city.drop(['index','City_Code','Cumulative_verified_cases','Cumulated_number_of_tests','Ni','Pi','Gi'], axis='columns', inplace=True)

    # First time: set split date by the ratio. More times: set ratio by split date.
    global split_date
    if (split_date == None):
        split_date = ds_city.iloc[int(len(ds_city) * train_ratio)]['Date']
        current_ratio = train_ratio
    else:
        fist_test_row = ds_city.index[ds_city['Date'] == split_date].to_list()[0]
        current_ratio = (fist_test_row - ds_city.index.to_list()[0]) / float(len(ds_city))

    ds_city.set_index('Date', inplace=True)
    ds_city = series_to_supervised(ds_city, window_size, look_forward)
    return split_train_test(ds_city, current_ratio, look_forward)

def all_cities_train():
    return train_by_cities_list(get_all_cities_list())

def l2_cities(city_code1, city_code2):
    #read_pop()
    #TODO
    def l_2(x, y):
        return math.sqrt(sum(map(lambda a, b: (a - b) ** 2, x, y)))
    return l_2([city_code1], [city_code2])

def knn_cities_train(city_code, top_k):
    cities = get_all_cities_list()
    if top_k > cities.size:
        return all_cities_train()
    sort_by_dis = sorted(enumerate(cities), key=(lambda y: l2_cities(y[1], city_code)))
    head_of_list = [sort_by_dis[i][1] for i in range(top_k)]
    return train_by_cities_list(head_of_list)

def train_by_cities_list(cities):
    trainX, trainy = np.empty(shape=(0, window_size)), np.empty(shape=(1, 0))
    for city in cities:
        trainX_city, trainy_city, _, _ = one_city_train_test(city)
        trainX = np.concatenate((trainX, trainX_city), axis=0)
        trainy = np.append(trainy, trainy_city)
    return trainX, trainy


# Split train-test
ds = pd.read_csv("ramzor2.csv", sep=",", header=0)
trainX, trainy, testX, testy = one_city_train_test(city_code)
#trainX, trainy = all_cities_train()
#trainX, trainy = knn_cities_train(city_code, 3)

# Select model
if model_type == "random forest":
    model = RandomForestRegressor(n_estimators=100, criterion='mse', min_samples_split = math.ceil(math.log2(trainX.size)))
elif model_type == "linear regression":
    model = LinearRegression()
elif model_type == "svm":
    model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=.1)
else:
    print("model_type error")
    exit(-1)

#learn
model.fit(trainX, trainy)
if model_type == "random forest":
    print('importances', model.feature_importances_) # optional to RandomForestRegressor

#prediction
predictions = predict_static(model, testX)
#predictions = predict_adaptive(model, trainX, trainy, testX, testy)

error = mean_squared_error(testy, predictions)
print("model error:", error)

# monkey model learning
model_monkey = AsLastColumn()
predictions_monkey = predict_static(model_monkey, testX)
error_monkey = mean_squared_error(testy, predictions_monkey)
print("monkey error:", error_monkey)

# plot expected vs predicted vs monkey
pyplot.plot(testy, label='Expected')
pyplot.plot(predictions, label='Predicted')
pyplot.plot(predictions_monkey, label='Monkey')
pyplot.legend()
pyplot.show()