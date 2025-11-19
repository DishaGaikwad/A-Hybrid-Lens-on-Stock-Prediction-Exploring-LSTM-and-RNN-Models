import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
import joblib
import os
import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

def create_sequences(data, look_back=60):
    x, y = [], []
    for i in range(look_back, len(data)):
        x.append(data[i - look_back:i, 0])
        y.append(data[i, 0])
    x, y = np.array(x), np.array(y)
    x = x.reshape((x.shape[0], x.shape[1], 1))
    return x, y

def build_model(model_type, input_shape):
    model = Sequential()
    if model_type == 'rnn':
        model.add(SimpleRNN(50, return_sequences=False, input_shape=input_shape))
    elif model_type == 'lstm':
        model.add(LSTM(50, return_sequences=False, input_shape=input_shape))
    elif model_type == 'hybrid':
        model.add(SimpleRNN(50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_save(file_path, model_basename, epochs=10, batch_size=32):
    if not os.path.exists('models'):
        os.makedirs('models')

    df = pd.read_csv(file_path)
    close_col = next((col for col in ['Close', 'close', 'CLOSE', 'Close Price', 'Closing Price', 'Close_Price', 'Adj Close', 'adj_close', 'ADJ_CLOSE'] if col in df.columns), None)
    if close_col is None:
        print(f"Close price column not found in {file_path}")
        return

    data = df[[close_col]].dropna().values
    scaler = MinMaxScaler((0,1))
    scaled = scaler.fit_transform(data)

    train_len = int(len(scaled)*0.8)
    train = scaled[:train_len]
    val = scaled[train_len-60:]

    x_train, y_train = create_sequences(train)
    x_val, y_val = create_sequences(val)

    models = {'rnn': None, 'lstm': None, 'hybrid': None}
    metrics = {}

    for model_type in models.keys():
        print(f"\nTraining {model_type.upper()} model...")
        model = build_model(model_type, (60,1))

        log_dir = f"logs/{model_basename}_{model_type}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        callbacks = [
            TensorBoard(log_dir=log_dir),
            CSVLogger(f"models/{model_basename}_{model_type}_log.csv"),
            ModelCheckpoint(f"models/best_{model_basename}_{model_type}.h5", save_best_only=True, monitor='val_loss')
        ]

        history = model.fit(x_train, y_train, validation_data=(x_val,y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)

        model.save(f"models/{model_basename}_{model_type}.h5")
        joblib.dump(scaler, f"models/{model_basename}_{model_type}_scaler.pkl")
        print(f"{model_type.upper()} model & scaler saved!")

        preds = model.predict(x_val)
        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, preds)
        metrics[model_type] = (mse, rmse, mae)

        # Save loss graph
        plt.figure(figsize=(6,4))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title(f"{model_type.upper()} Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"models/{model_basename}_{model_type}_loss.png")
        plt.close()

    # Combined metrics graph
    plt.figure(figsize=(8,6))
    for m in metrics:
        mse, rmse, _ = metrics[m]
        plt.bar(f'{m.upper()}-MSE', mse)
        plt.bar(f'{m.upper()}-RMSE', rmse)
    plt.title("MSE and RMSE Comparison")
    plt.ylabel("Error")
    plt.savefig(f"models/{model_basename}_metrics_comparison.png")
    plt.close()
    print("Combined MSE/RMSE graph saved.")

if __name__ == '__main__':
    files_to_process = [
        ('data/NIFTY IT_data.csv', 'nifty_it'),
        ('data/NIFTY FIN SERVICE_data.csv', 'nifty_fin_service'),
        ('data/NIFTY METAL_data.csv', 'nifty_metal')
    ]

    for file_path, model_basename in files_to_process:
        if os.path.exists(file_path):
            train_and_save(file_path, model_basename)
        else:
            print(f"File not found: {file_path}")