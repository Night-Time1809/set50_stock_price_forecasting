import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
import pickle
import re

def convert_into_uppercase(a):
    return a.group(1) + a.group(2).upper()

@st.cache
def load_data(ticker):
    data = yf.download(ticker)
    data.reset_index(inplace=True)
    return data

stock_dict = pickle.load(open("stock_names_dict.pkl", "rb"))
stock_symbol = list(stock_dict.keys())

st.markdown("# 📈 SET50 Stock Price Forecasting")
selected_stock = st.selectbox("Select stock", stock_symbol)
st.write("")

with st.container():
    company = re.sub("(^|\s)(\S)", convert_into_uppercase, (stock_dict[selected_stock]['Company']).lower())
    st.markdown(f"## {selected_stock}")
    st.markdown(f"### {company}")

    data_load_state = st.text("Load data...")
    data = load_data(ticker=f"{selected_stock}.BK")
    data_load_state.text("Loading data...done!")

    fig = go.Figure(data=go.Ohlc(x=data["Date"],
                    open=data["Open"],
                    high=data["High"],
                    low=data["Low"],
                    close=data["Close"]))
    st.plotly_chart(fig)

    # TODAY = date.today().strftime("%d %B %Y")
    # st.markdown(f"### {TODAY}")
    # # data_TODAY = data[data["Date"] == date.today().strftime("%Y-%m-%d")]
    # data_TODAY = data[data["Date"] == "2023-01-06"]
    # st.write(data)
    date_ = data["Date"].iloc[-1]
    st.write(date_)



# print(soup.prettify())

# START = "2015-01-01"
# TODAY = date.today().strftime("%Y-%m-%d")


# st.markdown("# SET50 Stock Price Forecasting")
# stocks = ["ADVANC", "AOT", "AWC"]
# stocks = {"ADVANC": "ADVANCED INFO SERVICE PUBLIC COMPANY LIMITED", "AOT": }
# # selected_stocks = st.selectbox("Select stock for forecasting", stocks)

# @st.cache
# def load_data(ticker):
#     data = yf.download(ticker, START, TODAY)
#     data.reset_index(inplace=True)
#     return data

# data_load_state = st.text("Load data...")
# stock_symbol = f"{selected_stocks}.BK"
# data = load_data(stock_symbol)
# data_load_state.text("Loading data...done!")

# st.subheader("Raw data")
# st.write(data.tail())

# def plot_raw_data():
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="stock_open"))
#     fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="stock_close"))
#     fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
#     st.plotly_chart(fig)

# plot_raw_data()