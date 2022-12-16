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

