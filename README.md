# 🛡️ Anomaly Detection in Multivariate Time Series using Autoencoder

An enterprise-grade AI Intrusion Detection System (IDS) designed to protect networks from Zero-Day cyberattacks. 

Unlike traditional signature-based firewalls, this project utilizes **Unsupervised Deep Learning** (LSTM Autoencoder) to learn the baseline mathematical behavior of normal network traffic. By calculating the Mean Squared Error (MSE) of real-time sequence reconstructions, the system dynamically flags malicious anomalies and isolates the exact compromised feature for rapid forensic analysis.

## 🚀 Key Features

* **Unsupervised Deep Learning:** Built with TensorFlow/Keras, requiring zero labeled attack data for training. It catches Zero-Day attacks by recognizing deviations from normal behavior.
* **Complex Data Engineering:** Transforms flat 2D network logs into 3D sequential tensors (50 time steps, 118 encoded features) using sliding window algorithms.
* **Real-Time SOC Dashboard:** A fully interactive Streamlit application simulating live network telemetry, complete with micro-batching and dynamic mathematical strictness controls.
* **Automated Root-Cause Analysis:** Algorithmic extraction (using `argmax`) of the exact multivariate feature responsible for an anomaly, logging it instantly to a Threat Intelligence Dataframe.

## 🛠️ Tech Stack

* **Core Language:** Python 3.x
* **Deep Learning Framework:** TensorFlow, Keras
* **Data Engineering & Math:** NumPy, Pandas, Scikit-Learn
* **UI & Visualization:** Streamlit, Plotly (WebGL)

## 📂 Project Structure

\`\`\`text
├── data/
│   ├── Train_data.csv          # Raw network traffic dataset (e.g., NSL-KDD)
│   ├── X_train_tensor.npy      # Preprocessed 3D tensors (Normal traffic)
│   └── X_test_tensor.npy       # Preprocessed 3D tensors (Mixed traffic)
├── dataProcessing.py           # Handles One-Hot encoding, MinMax scaling, and sliding windows
├── train_model.py              # Builds, trains, and tests the LSTM Autoencoder
├── real_time_app.py            # Streamlit dashboard for real-time inference
├── zero_day_ids_model.h5       # The trained neural network weights
└── README.md                   # Project documentation
\`\`\`

## ⚙️ Installation & Setup

1. **Clone the repository:**
   \`\`\`bash
   git clone https://github.com/YourUsername/Your-Repo-Name.git
   cd Your-Repo-Name
   \`\`\`

2. **Install the required dependencies:**
   \`\`\`bash
   pip install tensorflow numpy pandas scikit-learn streamlit plotly
   \`\`\`

3. **Run the Data Pipeline:**
   *(Ensure your raw dataset is in the `data/` folder first)*
   \`\`\`bash
   python dataProcessing.py
   \`\`\`

4. **Train the AI Model:**
   \`\`\`bash
   python train_model.py
   \`\`\`

5. **Launch the Real-Time Dashboard:**
   \`\`\`bash
   streamlit run real_time_app.py
   \`\`\`

## 🧠 How the AI Works

The system calculates the **Mean Squared Error (MSE)** between live incoming traffic and a perfectly normal AI reconstruction. By applying the Three-Sigma rule ($\mu + 3\sigma$) to the training errors, we created a statistical boundary. Any live sequence whose mathematical error exceeds that 99.7% confidence interval is automatically flagged as an anomaly.

---
*Project developed for academic defense / portfolio showcase.*