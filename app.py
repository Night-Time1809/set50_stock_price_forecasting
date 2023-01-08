import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
import pickle
import re
from dateutil.relativedelta import relativedelta

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
data_load_state = st.text("Load data...")
data = load_data(ticker=f"{selected_stock}.BK")
data_load_state.text("Loading data...done!")

with st.container():
    company = re.sub("(^|\s)(\S)", convert_into_uppercase, (stock_dict[selected_stock]['Company']).lower())
    st.markdown(f"## {selected_stock}")
    st.markdown(f"### {company}")

    show_data = st.radio("", ("Max", "5D", "1W", "1M", "6M", "1Y", "5Y"), horizontal=True, label_visibility="hidden")
    if show_data == "5D":
        START = relativedelta(days=-5)
        data_selected = data[data["Date"] >= (date.today() + START).strftime("%Y-%m-%d")]
    elif show_data == "1W":
        START = relativedelta(weeks=-1)
        data_selected = data[data["Date"] >= (date.today() + START).strftime("%Y-%m-%d")]
    elif show_data == "1M":
        START = relativedelta(months=-1)
        data_selected = data[data["Date"] >= (date.today() + START).strftime("%Y-%m-%d")]
    elif show_data == "6M":
        START = relativedelta(months=-6)
        data_selected = data[data["Date"] >= (date.today() + START).strftime("%Y-%m-%d")]
    elif show_data == "1Y":
        START = relativedelta(years=-1)
        data_selected = data[data["Date"] >= (date.today() + START).strftime("%Y-%m-%d")]
    elif show_data == "5Y":
        START = relativedelta(years=-5)
        data_selected = data[data["Date"] >= (date.today() + START).strftime("%Y-%m-%d")]
    else:
        data_selected = data.copy()

    fig = go.Figure(data=go.Ohlc(x=data_selected["Date"],
                    open=data_selected["Open"],
                    high=data_selected["High"],
                    low=data_selected["Low"],
                    close=data_selected["Close"]))
    st.plotly_chart(fig)

    date_ = (data["Date"].iloc[-1]).strftime("%d %B %Y")
    st.markdown(f"### {date_}")
    data_ = (data.iloc[-2:])

    col_stock = st.columns(3)
    with col_stock[0]:
        st.metric(label="Open", value=f"{round((data['Open'].iloc[-1]), 2)} THB")
        st.metric(label="Close", value=f"{round((data['Close'].iloc[-1]), 2)} THB")
        
    with col_stock [1]:
        st.metric(label="High", value=f"{round((data['High'].iloc[-1]), 2)} THB")
        change = data_["Close"].iloc[-1] - data_["Close"].iloc[-2]
        if change > 0:
            st.metric(label="Change", value=f"↑ {round(change, 2)} THB")
        elif change < 0:
            st.metric(label="Change", value=f"↓ {round(change, 2)} THB")
        else:
            st.metric(label="Change", value=f"{round(change, 2)} THB")

    with col_stock [2]:
        st.metric(label="Low", value=f"{round((data['Low'].iloc[-1]), 2)} THB")
        percent_change = (data_["Close"].iloc[-1] - data_["Close"].iloc[-2])/data_["Close"].iloc[-2] * 100
        if percent_change > 0:
            st.metric(label="% Change", value=f"↑ {round(percent_change, 2)}")
        elif percent_change < 0:
            st.metric(label="% Change", value=f"↓ {round(percent_change, 2)}")
        else:
            st.metric(label="% Change", value=f"{round(percent_change, 2)}")
st.write("")
st.markdown("## Model Training")
st.markdown("##### Data Preparation")
col_pre = st.columns(3)
with col_pre[0]:
    WINDOWS = st.number_input("WINDOWS (day)", min_value=1, max_value=365, value=7)
with col_pre[1]:
    HORIZONS = st.number_input("HORIZONS (day)", min_value=1, max_value=365, value=1)
with col_pre[2]:
    test_split = st.number_input("Test Splitting Ratio", min_value=0.1, max_value=0.5, step=0.1, value=0.2)

st.markdown("##### Training")
selected_model = st.selectbox("Select model", ["Neural Network", "LSTM"])
col_train = st.columns(2)
with col_train[0]:
    layers = st.number_input("LAYERS", min_value=1, max_value=10, value=3)
    structure = []
    for i in range(layers):
       node = st.number_input(f"Nodes of Layer {i+1}", min_value=1, max_value=300, value=32)
       structure.append(node)

with col_train[1]:
    EPOCH = st.number_input("EPOCH", min_value=1, value=100)
    BATCH_SIZE = st.number_input("BATCH SIZE", min_value=32, max_value=300, value=256)

model_name = f"{selected_model}-{structure}-EPOCH{EPOCH}-BATCH_SIZE{BATCH_SIZE}-WINDOW{WINDOWS}-HORIZON{HORIZONS}-test_split{test_split}"
st.write(model_name)

# def dense_model(structure, data, model_name, epoch, batch_size, save_path):
