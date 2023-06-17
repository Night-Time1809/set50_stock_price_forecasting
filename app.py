import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
import pickle
import re
from dateutil.relativedelta import relativedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import shutil
import pandas as pd

def convert_into_uppercase(a):
    return a.group(1) + a.group(2).upper()

@st.cache_data
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
path = "model"
if st.button("Delete all existing model"):
    # for file_name in os.listdir(f"{path}/{selected_stock}"):
    #     # os.remove(f"{path}/{selected_stock}/{file_name}")
    #     st.write(file_name)

    shutil.rmtree(f"{path}/{selected_stock}", ignore_errors=True)

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

structure_string = [str(i) for i in structure]
model_name = f"{selected_stock}-{selected_model}-{'_'.join(structure_string)}-EPOCH{EPOCH}-BATCH_SIZE{BATCH_SIZE}-WINDOW{WINDOWS}-HORIZON{HORIZONS}-test_split{str(test_split).replace('.', '')}"
st.write(model_name)

def get_labelled_windows(x, horizon):
    return x[:, :-horizon], x[:, -horizon:]

def make_windows(x, window_size, horizon):
    window_step = np.expand_dims(np.arange(window_size + horizon), axis=0)
    window_indexes = window_step + np.expand_dims(np.arange(len(x) - (window_size + horizon - 1)), axis=0).T
    windowed_array = x[window_indexes]
    windows, labels = get_labelled_windows(windowed_array, horizon=horizon)

    return windows, labels

def make_train_test_splits(windows, labels, test_split=0.2):
    split_size = int(len(windows) * (1 - test_split))
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]
    return train_windows, train_labels, test_windows, test_labels

def normalization(train, test):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(train)
    
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)

    return train_scaled, test_scaled, scaler

stock_prices = data["Close"].to_numpy()
price_windows, price_labels = make_windows(stock_prices, WINDOWS, HORIZONS)
train_windows, train_labels, test_windows, test_labels = make_train_test_splits(price_windows, price_labels)
train_windows_scaled, test_windows_scaled, scaler_windows = normalization(train_windows, test_windows)
train_labels_scaled, test_labels_scaled, scaler_labels = normalization(train_labels, test_labels)
data_train_test = {"train": {"windows": train_windows_scaled.astype(np.float32),
                             "labels": train_labels_scaled.astype(np.float32)},
                   "test": {"windows": test_windows_scaled.astype(np.float32),
                            "labels": test_labels_scaled.astype(np.float32)}}

def create_model_checkpoint(save_path):
    return tf.keras.callbacks.ModelCheckpoint(filepath=f"{save_path}",
                                              monitor="val_loss",
                                              verbose=1,
                                              save_best_only=True)

@st.cache_resource
def dense_model(structure, data, model_name, epoch, batch_size, save_path):
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    X_train = data["train"]["windows"]
    y_train = data["train"]["labels"]
    X_test = data["test"]["windows"]
    y_test = data["test"]["labels"]

    input = tf.keras.layers.Input(shape=(X_train.shape[1],))

    for i in range(len(structure)):
        if i == 0:
            x = tf.keras.layers.Dense(structure[i], activation="relu")(input)
        else:
            x = tf.keras.layers.Dense(structure[i], activation="relu")(x)

    output = tf.keras.layers.Dense(y_train.shape[1])(x)
    # model = tf.keras.Model(inputs=input,
    #                        outputs=output,
    #                        name=model_name)
    model = tf.keras.Model(inputs=input,
                           outputs=output)

    model.compile(loss="mae",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["mae", "mse"])
    
    history = model.fit(x=X_train,
                        y=y_train,
                        epochs=epoch,
                        batch_size=batch_size,
                        validation_data=(X_test, y_test),
                        callbacks=[create_model_checkpoint(save_path=save_path)])

    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    # plt.figure(figsize=(10,7))
    # epoch_plot = np.arange(1, epoch+1)
    # loss = np.array(history.history["loss"])
    # val_loss = np.array(history.history["val_loss"])
    # plt.plot(epoch_plot, loss, label="Traing")
    # plt.plot(epoch_plot, val_loss, label="Loss")
    # plt.ylabel("MAE", fontsize=14)
    # plt.xlabel("Epoch", fontsize=14)
    # plt.legend(fontsize=14)
    # plt.grid(True);
    
    model = tf.keras.models.load_model(save_path)

    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)

    # mse_train = mse(model, y_train, preds_train)
    # mse_test = mse(model, y_test, preds_test)

    # mae_train = mae(model, y_train, preds_train)
    # mae_test = mae(model, y_test, preds_test)

    # print(f"MSE_train: {mse_train}")
    # print(f"MSE_test: {mse_test}")
    # print(f"MAE_train: {mae_train}")
    # print(f"MAE_test: {mae_test}")

    # MSE = {"train": mse_train,
    #        "test": mse_test}

    # MAE = {"train": mae_train,
    #        "test": mae_test}

    # plt.figure(figsize=(10,7))
    # plt.plot(y_train[:,0], label="Training")
    # plt.plot(preds_train[:,0], label="Prediction")
    # plt.ylabel("Stock Price", fontsize=14)
    # plt.title("Training")
    # plt.legend(fontsize=14)
    # plt.grid(True);

    # plt.figure(figsize=(10,7))
    # plt.plot(y_test[:,0], label="Test")
    # plt.plot(preds_test[:,0], label="Prediction")
    # plt.ylabel("Stock Price", fontsize=14)
    # plt.title("Test")
    # plt.legend(fontsize=14)
    # plt.grid(True);
  
    # return model, MAE, MSE
    return train_loss, val_loss

if st.button("Train Model"):
    now = datetime.now(tz=ZoneInfo("Asia/Bangkok"))
    dt_string = now.strftime("%d%m%Y_%H%M")

    if os.path.isdir(f"model/{selected_stock}") != True:
        os.mkdir(f"{path}/{selected_stock}")

    if os.path.isfile(f"model/{selected_stock}/information.pkl"):
        inform = pickle.load(open(f"model/{selected_stock}/information.pkl", "rb"))
        st.write("Have")
        st.write(inform)
    else:
        inform = {}

    SAVE_PATH = f"{path}/{selected_stock}/{dt_string}"

    inform[dt_string] = {}
    inform[dt_string]["model"] = selected_model
    inform[dt_string]["structure"] = structure_string
    inform[dt_string]["epoch"] = EPOCH
    inform[dt_string]["batch_size"] = BATCH_SIZE
    inform[dt_string]["window"] = WINDOWS
    inform[dt_string]["horizon"] = HORIZONS
    inform[dt_string]["test_size"] = test_split

    if selected_model == "Neural Network":
        st.write("NN")
        train_loss, val_loss = dense_model(structure=structure, data=data_train_test, model_name=model_name, epoch=EPOCH, batch_size=BATCH_SIZE, save_path=SAVE_PATH)
    
    elif selected_model == "LSTM":
        st.write("LSTM")

    inform[dt_string]["train_loss"] = train_loss
    inform[dt_string]["val_loss"] = val_loss
    pickle.dump(inform, open(f"model/{selected_stock}/information.pkl", "wb"))

st.write("")
st.markdown("## Model Evaluation")

if os.path.isfile(f"{path}/{selected_stock}/information.pkl"):
    informs = pickle.load(open(f"{path}/{selected_stock}/information.pkl", "rb"))
    date = []
    time = []
    model_type = []
    structure = []
    epoch = []
    batch = []
    window = []
    horizon = []
    test = []

    for dt, detail in informs.items():
        dt_split = dt.split("_")
        d = dt_split[0]
        t = dt_split[1]
        date.append(d)
        time.append(t)

        model_type.append(detail["model"])
        structure.append(detail["structure"])
        epoch.append(detail["epoch"])
        batch.append(detail["batch_size"])
        window.append(detail["window"])
        horizon.append(detail["horizon"])
        test.append(detail["test_size"])

    dict_compare = {"Model": np.arange(1,len(informs.keys())+1),
                    "Date": date,
                    "Time": time,
                    "Model Type": model_type,
                    "Model Structure": structure,
                    "Epoch": epoch,
                    "Batch Size": batch,
                    "Window Size": window,
                    "Horizon": horizon,
                    "Test Size": test}

    df_compare = pd.DataFrame(dict_compare)

    st.write(df_compare)