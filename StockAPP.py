# IMPORTS

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from yahoo_fin import stock_info
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score

sns.set_style("darkgrid")
scaler = MinMaxScaler(feature_range=(0, 1))
st.title("             STOCK TREND PREDICTION AND COMPARISON")
choice=st.selectbox("What Do you Want to do?",["Stock Comparison","Stock Prediction"])


def Comparison(): #Compares the stocks of various brands
    st.title("Stock Trend comparison")

    options = [
                            {'label': 'Morgan Stanley', 'value': 'MS'},
                            {'label': 'JP Morgan', 'value': 'JPM'},
                            {'label': 'Wells Fargo & co.', 'value': 'WFC'},
                            {'label': 'Goldman Sachs', 'value': 'GS'},
                            {'label': 'Amazon.com, Inc.', 'value': 'AMZN'},
                            {'label': 'Tesla, Inc.', 'value': 'TSLA'},
                            {'label': 'Snap, Inc.', 'value': 'SNAP'},
                            {'label': 'Intel Comporation', 'value': 'INTC'},
                            {'label': 'Pinterest, Inc.', 'value': 'PINS'},
                            {'label': 'SoFi Technologies, Inc.', 'value': 'SOFI'},
                            {'label': 'Apple, Inc.', 'value': 'AAPL'},
                            {'label': 'New York Community Bancorp, Inc.', 'value': 'NYCB'},
                            {'label': 'Ford Motor Company', 'value': 'F'},
                            {'label': 'Advanced Micro Devices, Inc.', 'value': 'AMD'},
                            {'label': 'Bank of America Corporation', 'value': 'BAC'},
                            {'label': 'American Airlines Group Inc.', 'value': 'AAL'},
                            {'label': 'Southwestern Energy Company', 'value': 'SWN'},
                            {'label': 'AT&T Inc.', 'value': 'T'},
                            {'label': 'NIO, Inc.', 'value': 'NIO'},
                            {'label': 'Petróleo Brasileiro S.A. - Petrobras', 'value': 'PBR'},
                            {'label': 'Microsoft Corporation', 'value': 'MSFT'},
                            {'label': 'Meta Platforms, Inc.', 'value': 'META'},
                            {'label': 'Credit Suisse Group AG', 'value': 'CS'},
                            {'label': 'NVIDIA Corporation', 'value': 'NVDA'},
                            {'label': 'Alphabet Inc.', 'value': 'GOOGL'}
    ]
    options_dict = {item['value']: item['label'] for item in options}
    data = [(key, value) for key, value in options_dict.items()]
    cmpnies=pd.DataFrame(data,columns=["Code","Company"]) #converting dictionary to dataframe
    st.write(cmpnies)
    strinpt = str(st.text_input("Enter the codes of companies that you want to compare separated by ','", "AAPL,GOOGL", key="company_codes"))
    companies = strinpt.split(",")

    # Define unique keys for these st.text_input widgets
    start = st.text_input("Enter the start date", "2019-11-14", key="start_date_comparison")
    end = st.text_input("Enter the end date", "2023-11-03", key="end_date_comparison")

    colmn = st.text_input("Enter the attribute you want to compare:\nAvailable attributes are:\nHigh,Low,Close,Adj Close,Volume", "Close", key="attribute")

    fig = plt.figure(figsize=(10, 5))
    axes = fig.add_axes([0, 0, 1, 1])

    for comp in companies:
        data = yf.download(comp, start, end)[colmn]
        axes.plot(data)

    axes.set_ylabel("Price")
    axes.set_xlabel("date")
    axes.legend(companies)
    st.pyplot(fig)
    

    for comp in companies:
        # Create a Ticker object for the stock symbol
        ticker = yf.Ticker(comp)

        news_data = ticker.news

        st.sidebar.title("Finance News for " + options_dict[comp])
        st.sidebar.subheader("------------------------------------------------")
        for idx, news_item in enumerate(news_data):
            st.sidebar.subheader(f"{idx + 1}: {news_item['title']}")
            st.sidebar.markdown("URL: [Read more](" + news_item['link'] + ")")
            st.sidebar.markdown("----")
        st.sidebar.subheader("------------------------------------------------")





def Prediction(): #Predicts the future stock price for a particular brand
    st.title("Stock Trend prediction")

    start=st.text_input("Enter the start date","2019-11-14")
    end=st.text_input("Enter the end date","2023-11-03")
    date_range = pd.date_range(start=start, end=end)

    options = [
                            {'label': 'Morgan Stanley', 'value': 'MS'},
                            {'label': 'JP Morgan', 'value': 'JPM'},
                            {'label': 'Wells Fargo & co.', 'value': 'WFC'},
                            {'label': 'Goldman Sachs', 'value': 'GS'},
                            {'label': 'Amazon.com, Inc.', 'value': 'AMZN'},
                            {'label': 'Tesla, Inc.', 'value': 'TSLA'},
                            {'label': 'Snap, Inc.', 'value': 'SNAP'},
                            {'label': 'Intel Comporation', 'value': 'INTC'},
                            {'label': 'Pinterest, Inc.', 'value': 'PINS'},
                            {'label': 'SoFi Technologies, Inc.', 'value': 'SOFI'},
                            {'label': 'Apple, Inc.', 'value': 'AAPL'},
                            {'label': 'New York Community Bancorp, Inc.', 'value': 'NYCB'},
                            {'label': 'Ford Motor Company', 'value': 'F'},
                            {'label': 'Advanced Micro Devices, Inc.', 'value': 'AMD'},
                            {'label': 'Bank of America Corporation', 'value': 'BAC'},
                            {'label': 'American Airlines Group Inc.', 'value': 'AAL'},
                            {'label': 'Southwestern Energy Company', 'value': 'SWN'},
                            {'label': 'AT&T Inc.', 'value': 'T'},
                            {'label': 'NIO, Inc.', 'value': 'NIO'},
                            {'label': 'Petróleo Brasileiro S.A. - Petrobras', 'value': 'PBR'},
                            {'label': 'Microsoft Corporation', 'value': 'MSFT'},
                            {'label': 'Meta Platforms, Inc.', 'value': 'META'},
                            {'label': 'Credit Suisse Group AG', 'value': 'CS'},
                            {'label': 'NVIDIA Corporation', 'value': 'NVDA'},
                            {'label': 'Alphabet Inc.', 'value': 'GOOGL'}
    ]
    options_dict = {item['value']: item['label'] for item in options}

    listtech = ["GOOGL", "NVDA", "CS", "META", "MSFT", "PBR", "T", "SWN", "AAL", "AMD", "F", "NYCB", "AAPL", "SOFI", "PINS", "INTC", "SNAP", "AMZN"]

    # Create a drop-down list in Streamlit
    selected_option = st.selectbox("Select an option", [item['label'] for item in options])

    # Get the selected value
    Tick = [item['value'] for item in options if item['label'] == selected_option][0]

    # You can use the selected_value in your app as needed
    st.write("You selected:", Tick)

    df = yf.download(Tick, start, end)

    st.subheader(f"Data from {start} to {end}")
    st.write(df.tail(10))

    st.subheader(f"Summary of Data from {start} to {end}")
    st.write(df.describe())
    intrest = st.text_input("Enter column you want to analyze", "Close")

    st.subheader("Moving averages")
    avg100 = df[intrest].rolling(100).mean()
    avg200 = df[intrest].rolling(200).mean()
    closing = df[intrest]
    fig1 = plt.figure(figsize=(10, 5))
    axes1 = fig1.add_axes([0, 0, 1, 1]) 
    axes1.plot(avg100)
    axes1.plot(avg200)
    axes1.plot(closing)
    axes1.set_ylabel('Price')
    axes1.set_xlabel("date")
    axes1.set_title('ROLLING AVERAGES')
    axes1.legend(["100 days avg", "200 days avg", "Closing price"])
    st.pyplot(fig1)

    # Importing the Model
    modeltech = load_model("keras_stockmodel.h5")
    modelfin = load_model("keras_stockmodelfin.h5")

    if Tick in listtech:
        model = modeltech
    else:
        model = modelfin

    def acc_sf(): #Plot depicting the accuracy of Model
        col = np.array(df[intrest])
        col = col.reshape(-1, 1)
        col = scaler.fit_transform(col)
        scale_factor = 1 / scaler.scale_
        # Splitting the data into x_test and Y_test
        inp = []

        for i in range(100, int(len(col))):
            inp.append(col[i-100:i])  # n-100th day to nth day

        inp = np.array(inp)
        st.subheader("ACCURACY OF MODEL")
        op = model.predict(inp)
        fig1 = plt.figure(figsize=(10, 5))
        axes1 = fig1.add_axes([0, 0, 1, 1]) 
        axes1.plot(col[100:] * scale_factor)
        axes1.plot(op * scale_factor)
        axes1.set_ylabel('Price')
        axes1.set_xlabel("days")
        axes1.set_title('Fig depicting Accuracy of Model')
        axes1.legend(["Actual", "Predicted"])
        st.pyplot(fig1)
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        # Reverse the scaling to get the original values
        op = op * scaler.scale_  # Reverse scaling
        op = op + scaler.min_  # Reverse min-max scaling

        # Calculate regression metrics
        mae = mean_absolute_error(col[100:], op)
        mse = mean_squared_error(col[100:], op)

        # Display the regression metrics
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")


    acc_sf()

    def pred(): #Prediction of future stock prices
        st.subheader("Model prediction for Next days")
        col = np.array(df[intrest])[-100:]
        col = col.reshape(-1, 1)
        col = scaler.fit_transform(col)

        inp = col
        inp = inp.reshape(1, 100, 1)

        op = []
        for p in range(30):
            val = model.predict(inp)
            op.append(val)
            inp = np.concatenate((inp, val.reshape(1, 1, 1)), axis=1)
            inp = inp[:, 1:, :]

        op = np.array(op)
        finval = np.concatenate((col.reshape(100, 1, 1), op), axis=0).ravel()
        scale_factor = 1 / scaler.scale_
        fig1 = plt.figure(figsize=(10, 5))
        axes1 = fig1.add_axes([0, 0, 1, 1]) 
        axes1.plot(finval * scale_factor)
        axes1.plot(col * scale_factor)
        axes1.set_ylabel('Price')
        axes1.set_xlabel("days")
        axes1.set_title('FORCAST FOR NEXT 30 days')
        axes1.legend(["Prediction for next 30 days", "Past 100 days"])
        st.pyplot(fig1)

    pred()
    
    # Create a Ticker object for the stock symbol
    ticker = yf.Ticker(Tick)

    news_data = ticker.news

    st.sidebar.title("Finance News for " + options_dict[Tick])
    st.sidebar.subheader("------------------------------------------------")
    for idx, news_item in enumerate(news_data):
        st.sidebar.subheader(f"{idx + 1}: {news_item['title']}")
        st.sidebar.markdown("URL: [Read more](" + news_item['link'] + ")")
        st.sidebar.markdown("----")



if(choice=="Stock Comparison"):
    Comparison()
else:
    Prediction()

