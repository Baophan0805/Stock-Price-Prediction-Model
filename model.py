import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import vnquant.data as dt
import tensorflow as tf


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#Load data by vnquant
import vnquant.data as dt

company = 'ASG'
start = '2018-01-01'
end = '2021-12-31' 

data = dt.DataLoader(company,start,end,minimal=True,data_source='vnd')
data = data.download()
# Prepare data
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(data['close'].values.reshape(-1,1))

#Data preprocessing
prediction_days = 60
x_train = []
y_train = []
# 60-fold validation
for x in range(prediction_days, len(scaled_data)):
  x_train.append(scaled_data[x - prediction_days:x,0])
  y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape data
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
y_train = np.reshape(y_train,(y_train.shape[0],1))

# Build the model
model = Sequential()
model.add(LSTM(units=128,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=64))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mean_absolute_error')

# Train
best_model = tf.keras.callbacks.EarlyStopping(monitor = 'loss',
                             patience=3,
                             verbose=2
                             )
model.fit(x_train,y_train,epochs=100, batch_size = 50,verbose=2,callbacks=[best_model])

# Load test data 
test_start = '2022-01-01'
test_end = '2022-03-31'
test_data = dt.DataLoader(company,test_start,test_end,minimal=True,data_source='vnd').download()

actual_prices = test_data['close'].values

total_dataset = pd.concat((data['close'], test_data['close']),axis=0)
model_inputs = total_dataset[
    len(total_dataset)-len(test_data)-prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

y_train = scaler.inverse_transform(y_train) #real data
# final_model = tf.keras.models.load_model('save_model.hdf5')
train_predicted_prices = model.predict(x_train)
train_predicted_prices = scaler.inverse_transform(train_predicted_prices) #train predicted price

# Predict on Test Data
x_test = []
# Sliding window technique, use 60 past days to predict the next day
for x in range(prediction_days, len(model_inputs)):
  x_test.append(model_inputs[x-prediction_days:x,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

test_predicted_prices = model.predict(x_test) #predict 60 days in the test period
test_predicted_prices = scaler.inverse_transform(test_predicted_prices)

# Plot the test predictions
# plt.plot(actual_prices,color="black",label=f"Actual {company} Price")
# plt.plot(predicted_prices,color='green',label=f"Predicted {company} Price")
# plt.title(f"{company} Share Price")
# plt.xlabel("Time")
# plt.ylabel(f"{company} Share price")
# plt.legend()
# plt.show
# Arrange to plot 
train_data1 = data['close'][prediction_days:]
test_data1 = test_data['close']
# data , test_data
plt.figure(figsize=(24,8))
plt.plot(total_dataset,color="red",label=f"Actual {company} Price") # real dta
train_data1['predicted'] = train_predicted_prices
plt.plot(train_data1['predicted'],color='green',label = "train predicted price") # train predicted price
test_data1['predicted'] = test_predicted_prices
plt.plot(test_data1['predicted'], color='blue',label="test predicted price")
plt.title(f"{company} Share Price")
plt.xlabel("Time")
plt.ylabel("Thousand VND")
plt.legend()

# Evaluation metric
def _metric_measure(actual, predicted,tag,company):
  mape = mean_absolute_percentage_error(actual,predicted)
  mae = mean_absolute_error(actual,predicted) * 1000
  print("--Model evaluation on {} data of {}--".format(tag,company))
  print('Mean Absolute Percentage Error: {}'.format(mape))
  print('Mean Absolute Error: {}'.format(mae))

