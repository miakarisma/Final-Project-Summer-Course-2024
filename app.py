from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from io import StringIO

app = Flask(__name__, static_folder='assets')

# Load the dataset
data_path = 'datasets/World_Malnutrition_Data_1970-2022.xlsx'
df = pd.read_excel(data_path)
df = df[['Geographic area','TIME_PERIOD','OBS_VALUE']]
dfs = {country: df[df['Geographic area'] == country] for country in df['Geographic area'].unique()}
countries = sorted(df['Geographic area'].unique())

# Function to calculate trend
def calculate_trend(series):
    differences = np.diff(series)
    avg_change = np.mean(differences)
    return avg_change

# Function to find extremes
def find_extremes(time, series):
    max_value = np.max(series)
    min_value = np.min(series)
    max_time = time[np.argmax(series)]
    min_time = time[np.argmin(series)]
    return max_value, max_time, min_value, min_time

# Function to display information about dataset
def display():
    head_html = df.head().to_html(classes='table table-striped')
    
    # Use StringIO to capture the output of df.info()
    buffer = StringIO()
    df.info(buf=buffer)
    info_html = buffer.getvalue().replace('\n', '<br>')
    
    return head_html, info_html

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/get_countries')
def get_countries():
    return jsonify(countries=countries)

@app.route('/get_data')
def get_data():
    country = request.args.get('country')
    df_v = dfs[country]
    df_v = df_v.assign(TIME_PERIOD=pd.to_datetime(df_v['TIME_PERIOD'], format='%Y'))
    df_v['OBS_VALUE'] = df_v['OBS_VALUE'].astype('float')
    time = df_v['TIME_PERIOD'].dt.strftime('%Y-%m-%d').tolist()
    values = df_v['OBS_VALUE'].tolist()
    return jsonify(time=time, values=values)

@app.route('/predict')
def predict():
    country = request.args.get('country')
    df_v = dfs[country]
    df_v = df_v.assign(TIME_PERIOD=pd.to_datetime(df_v['TIME_PERIOD'], format='%Y'))
    df_v['OBS_VALUE'] = df_v['OBS_VALUE'].astype('float')
    original_data = df_v['OBS_VALUE'].values
    time_step = df_v['TIME_PERIOD'].values

    min_val = np.min(original_data)
    max_val = np.max(original_data)
    series = (original_data - min_val) / (max_val - min_val)

    split_time_train = int(0.8 * len(series))
    time_train = time_step[:split_time_train]
    x_train = series[:split_time_train]
    time_valid = time_step[split_time_train:]
    x_valid = series[split_time_train:]

    window_size = 2
    batch_size = 1
    shuffle_buffer_size = 1

    def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
        dataset = tf.data.Dataset.from_tensor_slices(series)
        dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
        dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
        dataset = dataset.batch(batch_size).prefetch(1)
        return dataset

    train_set = windowed_dataset(x_train, window_size=window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)
    valid_set = windowed_dataset(x_valid, window_size=window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)

    model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(filters=16, kernel_size=3, strides=1, activation="relu", padding='causal', input_shape=[None, 1]),
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dense(20, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

    # Compile
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.SGD(momentum=0.9), metrics=["mae"])
    
    time_train = np.array(time_step)
    x_train = np.array(series)

    train_set = windowed_dataset(x_train, window_size=window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)

    model.fit(train_set, epochs=50, 
                callbacks=[tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch / 20))])
    
    # Predicting the next 10 years
    last_window = x_train[-window_size:]
    future_predictions = []

    for i in range(5):
        pred = model.predict(last_window[np.newaxis, :, np.newaxis])[0, 0]
        future_predictions.append(pred)
        last_window = np.append(last_window[1:], pred)

    future_predictions = np.array(future_predictions) * (max_val - min_val) + min_val

    last_date = time_step[-1]
    future_dates = pd.date_range(last_date, periods=6, freq='AS-JAN')[1:]
    
    # Ensure consistent date formats
    last_date = pd.to_datetime(time_step[-1])
    future_start_date = pd.to_datetime(future_dates[0])

    original_trend = calculate_trend(original_data)
    future_trend = calculate_trend(future_predictions)
    max_value, max_time, min_value, min_time = find_extremes(time_step, original_data)
    min_time = pd.to_datetime(min_time)
    max_time = pd.to_datetime(max_time)

    original_trend_desc = "Historical data shows a positive (upward) trend." if original_trend > 0 else "Historical data shows a negative (downward) trend." if original_trend < 0 else "Historical data shows a stable trend."
    future_trend_desc = "Future predictions shows a positive (upward) trend." if future_trend > 0 else "Future predictions shows a negative (downward) trend." if future_trend < 0 else "Future predictions shows a stable trend."

    return jsonify(
        time=time_step.astype(str).tolist(),
        last_time=last_date.strftime('%Y-%m-%d'),
        last_value=float(series[-1] * (max_val - min_val) + min_val),
        future_start_time=future_start_date.strftime('%Y-%m-%d'),
        values=(series * (max_val - min_val) + min_val).tolist(),
        future_time=future_dates.astype(str).tolist(),
        future_values=future_predictions.flatten().tolist(),
        original_trend_desc=original_trend_desc,
        future_trend_desc=future_trend_desc,
        max_value=max_value,
        max_time=max_time.strftime('%Y'),
        min_value=min_value,
        min_time=min_time.strftime('%Y')
    )
    
@app.route('/about')
def about():
    head_html, info_html = display()
    return render_template('about.html', table=head_html, info=info_html)

if __name__ == '__main__':
    # app.run()
    app.run(debug=True)
