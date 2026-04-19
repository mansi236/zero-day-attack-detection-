import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input
import numpy as np
import matplotlib.pyplot as plt

def build_and_train_autoencoder(X_train, X_test):
    time_steps = X_train.shape[1]
    num_features = X_train.shape[2]
    print("\n--- PHASE 2: Building LSTM Autoencoder ---") 
    model = Sequential()
    model.add(Input(shape=(time_steps, num_features)))
    model.add(LSTM(units=64, activation='relu', return_sequences=False, name="Encoder_LSTM"))
    model.add(RepeatVector(time_steps, name="Bridge_Repeat"))
    model.add(LSTM(units=64, activation='relu', return_sequences=True, name="Decoder_LSTM"))
    model.add(TimeDistributed(Dense(units=num_features), name="Output_Reconstruction"))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    print("\n--- PHASE 3: Training the Model ---")
    
    history = model.fit(
        X_train, X_train,
        epochs=20,              
        batch_size=128,         
        validation_split=0.1,   
        verbose=1
    )
    print("\n--- PHASE 4: Calculating Anomaly Threshold ---")
    train_predictions = model.predict(X_train)
    train_mse = np.mean(np.power(X_train - train_predictions, 2), axis=(1, 2))
    threshold = np.mean(train_mse) + 3 * np.std(train_mse)
    print(f"Calculated Anomaly Threshold (MSE): {threshold:.5f}")
    
    print("\n--- PHASE 5: Detecting Zero-Day Attacks (Testing) ---")
    test_predictions = model.predict(X_test)
    test_mse = np.mean(np.power(X_test - test_predictions, 2), axis=(1, 2))
    anomalies = test_mse > threshold
    total_anomalies = np.sum(anomalies)
    print(f"Total Sequences Evaluated: {len(test_mse)}")
    print(f"Total Cyberattacks/Anomalies Detected: {total_anomalies}")
    
    return model, threshold, test_mse, anomalies


if __name__ == "__main__":
    import pandas as pd # Make sure pandas is imported to save the results later
    print("1. Loading Preprocessed 3D Tensors from disk...")

    try:
        X_train_tensor = np.load('X_train_tensor.npy')
        X_test_tensor = np.load('X_test_tensor.npy')
        print(f"Loaded Train Shape: {X_train_tensor.shape}")
        print(f"Loaded Test Shape:  {X_test_tensor.shape}")
    except FileNotFoundError:
        print("ERROR: Could not find the .npy files! Run data_preprocessing.py first.")
        exit()


    trained_model, final_threshold, mse_scores, anomaly_flags = build_and_train_autoencoder(X_train_tensor, X_test_tensor)
    
    print("\n--- PHASE 6: Saving the Trained AI ---")

    trained_model.save('zero_day_ids_model.h5')
    results_df = pd.DataFrame({
        'Reconstruction_Error': mse_scores,
        'Is_Attack': anomaly_flags
    })
    results_df.to_csv('model_results.csv', index=False)
    
    print("✅ SUCCESS: Model Trained and Results Saved! You can now run your Streamlit UI.")