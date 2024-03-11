import streamlit as st        #module for streamlit webapp
from streamlit import subheader    #subheader is used to give subheader for for section and subsections
from datetime import date            #date and time
import pandas as pd

import yfinance as yf               #yahoo finanace changed to yfinanace , here yf is used as yfinanace in entire pg
from yfinance import ticker           #tickers are generally abbrevation used for all listed accomdities
from prophet import Prophet          #fbprophet changed to prophet , used for forcasting future trends
from prophet.plot import plot_plotly
from plotly import graph_objs as go     #generally used for plotting graphs



START = "2010-01-01"
# TODAY= date.today().strftime("%y-%m-%d")
END="2023-04-15"

st.title("RARE EARTH FILTHY LUCRE PREDICTION USING MACHINE LEARNING")

st.text(' ABHAY KUMAR JOSHI [1DB19CS003]')
st.text('  ARYAMAN VIJAY [1DB19CS015] ')
st.text(' DISHA KAUR L [1DB19CS046] ')





stocks = ("GC=F","SI=F","PL=F","PA=F","HINDALCO.NS")
selected_stock = st.selectbox("Select Data set for prediction ", stocks)

n_years = st.slider("years of prediction",1,4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker,START,END)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load Data...")
data=load_data(selected_stock)
data_load_state.text("Loading data ... done!  [(GC=F)->GOLD,(SI=F)->SILVER,(PL=F)->PLATINIUM,(PA=F)->PALLADIUM]")

st:subheader(' This is  Raw data(describe)  ')
st.write(data.describe())


st:subheader('  Record from data.head ')
st.write(data.head())

st:subheader(' Record from data.tail ')
st.write(data.tail())
# ------------------------




# st.subheader('Data from 2010 - 2023')
# st.write(df.describe())


# # VISUALISATION
# st.subheader('closing price vs time chart')
# fig = plt.figure(figsize=(12,6))
# plt.plot(df.Close )
# st.pyplot(fig)

# st.subheader('closing price vs time chart with 100MA ')
# ma100=df.Close.rolling(100).mean()
# fig = plt.figure(figsize=(12,6))
# plt.plot(ma100)
# plt.plot(df.Close)
# st.pyplot(fig)

# st.subheader('closing price vs time chart with 100MA & 200MA')
# ma100=df.Close.rolling(100).mean()
# ma200=df.Close.rolling(200).mean()
# fig = plt.figure(figsize=(12,6))
# plt.plot(ma100)
# plt.plot(ma200)
# plt.plot(df.Close)
# st.pyplot(fig)


# # SPLITTING Data into Training and Testing
# data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
# data_testing = pd.DataFrame(df['Close'][0:int(len(df)*0.70):int(len(df))])

# from sklearn.preprocessing import MinMaxScaler
# Scaler = MinMaxScaler(feature_range=(0.1))

# data_training_array = Scaler.fit_transform(data_training)


# #splitting Data  into x_trains and y_trains

# x_trains =[]
# y_trains = []

# for i in range(100,data_training_array.shape[0]):
#     x_trains.append(data_training_array[i-100:i])
#     y_trains.append(data_training_array[i,0])



# x_trains ,y_trains =np.array(x_trains),np.array(y_trains)




# Model Loading







# -----------------------
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter (x=data['Date'],y=data['Open'],name='stock_open'))
    fig.add_trace(go.Scatter (x=data['Date'],y=data['Close'],name='stock_close'))
    fig.layout.update(title_text="Time Series Data (Time vs price)",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# # SPLITTING Data into Training and Testing
# data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
# data_testing = pd.DataFrame(df['Close'][0:int(len(df)*0.70):int(len(df))])

# from sklearn.preprocessing import MinMaxScaler
# Scaler = MinMaxScaler(feature_range=(0.1))

# data_training_array = Scaler.fit_transform(data_training)


# #splitting Data  into x_trains and y_trains

# x_trains =[]
# y_trains = []

# for i in range(100,data_training_array.shape[0]):
#     x_trains.append(data_training_array[i-100:i])
#     y_trains.append(data_training_array[i,0])


plot_raw_data()    


#Forcasting

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date":"ds","Close":"y"})


m=Prophet()
m.fit(df_train)
future=m.make_future_dataframe(periods=period)
forcast = m.predict(future)


st:subheader(' Forecast Data ')
st.write(forcast.tail())


st.write('Forecaste Data')
fig1 = plot_plotly(m,forcast)
st.plotly_chart(fig1)

st.write('Forecast Components')
fig2 = m.plot_components(forcast)
st.write(fig2)
