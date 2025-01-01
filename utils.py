from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy
import pandas as pd
import math
import os

# setting a seed for reproducibility
numpy.random.seed(10)
# read all stock files in directory indivisual_stocks_5yr
def read_all_stock_files(folder_path):
    allFiles = []
    for (_, _, files) in os.walk(folder_path):
        allFiles.extend(files)
        break

    dataframe_dict = {}
    for stock_file in allFiles:
        df = pd.read_csv(folder_path + "/" +stock_file)
        dataframe_dict[(stock_file.split('_'))[0]] = df

    return dataframe_dict
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)
# create dataset from the dataframe
def create_preprocessed_Dataset(df):
    df.drop(df.columns.difference(['date', 'close']), axis=1, inplace=True)
    df = df['close']
    dataset = df.values
    dataset = dataset.reshape(-1, 1)
    dataset = dataset.astype('float32')

    # split into train and test sets
    train_size = len(dataset) - 2
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    # trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    # testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    return trainX, trainY, testX, testY
# extract input dates and opening price value of stocks
def getData(df):
    # Create the lists / X and Y data sets
    dates = []
    prices = []

    # Get the last row of data (this will be the data that we test on)
    last_row = df.tail(1)

    # Get all of the data except for the last row
    df = df.head(len(df) - 1)

    # Get all of the rows from the Date Column
    df_dates = df.loc[:, 'date']
    # Get all of the rows from the close Column
    df_close = df.loc[:, 'close']

    # Create the independent data set X
    for date in df_dates:
        # Check if date is a string before processing
        if isinstance(date, str):
            date_parts = date.split('/')
            if len(date_parts) == 3:  # Ensure there are three parts
                try:
                    # Extract month, day, year
                    month = int(date_parts[0])
                    day = int(date_parts[1])
                    year = int(date_parts[2])
                    dates.append([day])  # Store the day
                except ValueError as e:
                    print(f"Error converting date parts to int: {date_parts}. Error: {e}")
            else:
                print(f"Invalid date format: {date}")
        else:
            print(f"Date is not a string: {date}")

    # Create the dependent data set 'y'
    for close_price in df_close:
        if isinstance(close_price, (int, float)):  # Ensure close_price is a number
            prices.append(float(close_price))

    # Safeguard for last_date
    if not last_row.empty and isinstance(last_row['date'].iloc[0], str):
        last_date_parts = last_row['date'].iloc[0].split('/')
        if len(last_date_parts) == 3:
            last_date = int(last_date_parts[1])  # Extract the day
        else:
            print(f"Invalid last date format: {last_row['date'].iloc[0]}")
            last_date = 0  # Default value if format is invalid
    else:
        print("Last row is empty or last date is not a string.")
        last_date = 0  # Default value if last row is empty

    last_price = float((list(last_row['close']))[0]) if not last_row.empty else 0.0
    return dates, prices, last_date, last_price

def KNN(dates, prices, test_date, df):
    knn = KNeighborsRegressor(n_neighbors=2)
    trainX, trainY, testX, testY = create_preprocessed_Dataset(df)
    # trainX = [item for sublist in trainX for item in sublist]
    # testX = [item for sublist in testX for item in sublist]
    X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.33, random_state=42)
    knn.fit(trainX, trainY)
    decision_boundary = knn.predict(trainX)
    y_pred = knn.predict(X_test)
    test_score = mean_squared_error(y_test, y_pred)
    prediction = knn.predict(testX)[0]

    return (decision_boundary, prediction, test_score)

def LSTM_model(dates, prices, test_date, df):
    df.drop(df.columns.difference(['date', 'close']), axis=1, inplace=True)
    df = df['close']
    dataset = df.values
    dataset = dataset.reshape(-1, 1)
    dataset = dataset.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = len(dataset) - 2
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.33, random_state=42)
    # reshape input to be [samples, time steps, features]
    X_train = numpy.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = numpy.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

    # make predictions
    trainPredict = model.predict(X_train)
    mainTestPredict = model.predict(X_test)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    y_train = scaler.inverse_transform([y_train])

    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    mainTestPredict = scaler.inverse_transform(mainTestPredict)
    mainTestPredict = [item for sublist in mainTestPredict for item in sublist]
    y_test = scaler.inverse_transform([y_test])
    test_score = mean_squared_error(y_test[0], mainTestPredict)
    # calculate root mean squared error
    trainPredict = [item for sublist in trainPredict for item in sublist]

    # print(trainPredict, testPredict[0])

    return (trainPredict, (testPredict[0])[0], test_score)