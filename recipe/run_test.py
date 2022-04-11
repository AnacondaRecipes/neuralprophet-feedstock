from neuralprophet import NeuralProphet
import pandas as pd

data_location = "https://raw.githubusercontent.com/ourownstory/neuralprophet-data/main/datasets/"

df = pd.read_csv(data_location + 'wp_log_peyton_manning.csv')
df.head(3)


m = NeuralProphet()
metrics = m.fit(df)

future = m.make_future_dataframe(df=df, periods=365)
print(future)
forecast = m.predict(df=future)
print(forecast)