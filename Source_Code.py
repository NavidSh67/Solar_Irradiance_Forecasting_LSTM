import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

class SolarRadiationForecast:
    def __init__(self, data_file):
        self.data_file = data_file
        self.df = None
        self.training_set_scaled = None
        self.test_set_scaled = None
        self.X_train = None
        self.y_train = None
        self.model = None
        self.prediction_test = None
        self.predictions = None
        self.real_values = None

    def load_data(self):
        self.df = pd.read_csv(self.data_file, index_col='Date', parse_dates=True)
        self.df.index.freq = 'H'
        self.df.dropna(inplace=True)

    def prepare_training_test_sets(self):
        training_set = self.df.iloc[:8712, 0:1].values
        test_set = self.df.iloc[8712:, 0:1].values

        sc = MinMaxScaler(feature_range=(0, 1))
        self.training_set_scaled = sc.fit_transform(training_set)
        self.test_set_scaled = sc.fit_transform(test_set)

    def prepare_sequences(self, window_size):
        X_train = []
        y_train = []

        for i in range(window_size, len(self.training_set_scaled)):
            X_train.append(self.training_set_scaled[i - window_size:i, 0:1])
            y_train.append(self.training_set_scaled[i, 0])

        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def build_model(self):
        model = Sequential()
        model.add(LSTM(units=60, return_sequences=True, input_shape=(self.X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=60, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=60, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=60))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        self.model = model

    def train_model(self, epochs, batch_size):
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size)

    def save_model(self, model_file):
        self.model.save(model_file)

    def load_model(self, model_file):
        self.model = load_model(model_file)

    def plot_loss(self):
        plt.plot(range(len(self.model.history.history['loss'])), self.model.history.history['loss'])
        plt.xlabel('Epoch Number')
        plt.ylabel('Loss')
        plt.show()

    def forecast(self, window_size, prediction_steps):
        self.prediction_test = []
        first_batch = self.training_set_scaled[-window_size:]
        current_batch = first_batch.reshape((1, window_size, 1))

        for _ in range(prediction_steps):
            current_pred = self.model.predict(current_batch)[0]
            self.prediction_test.append(current_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

        self.prediction_test = np.array(self.prediction_test)
        self.predictions = sc.inverse_transform(self.prediction_test)
        self.real_values = self.test_set_scaled[:prediction_steps]

    def plot_forecast(self):
        plt.plot(self.real_values, color='red', label='Actual Solar Irradiance')
        plt.plot(self.predictions, color='blue', label='Predicted Solar Irradiance')
        plt.title('LSTM Model - Solar Irradiance Forecasting')
        plt.xlabel('Time (hr)')
        plt.ylabel('Solar Irradiance (kW/m2)')
        plt.legend()
        plt.show()

    def calculate_metrics(self):
        rmse = math.sqrt(mean_squared_error(self.real_values, self.predictions))
        r_square = r2_score(self.real_values, self.predictions)
        return rmse, r_square


# Usage example
forecast = SolarRadiationForecast('Solar Radiation.csv')
forecast.load_data()
forecast.prepare_training_test_sets()
forecast.prepare_sequences(window_size=24)
forecast.build_model()
forecast.train_model(epochs=30, batch_size=32)
forecast.save_model('LSTM-Univariant')
forecast.load_model('LSTM-Univariant')
forecast.plot_loss()
forecast.forecast(window_size=24, prediction_steps=48)
forecast.plot_forecast()
rmse, r_square = forecast.calculate_metrics()
print("RMSE:", rmse)
print("R^2:", r_square)
