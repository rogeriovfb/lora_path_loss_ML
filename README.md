# Path Loss Prediction Using Machine Learning in LoRa Network
In this study, we aim to explore and compare different machine learning (ML) algorithms for predicting path loss in LoRa networks. By leveraging datasets collected from real-world deployments, we seek to develop robust predictive models that can accurately estimate path loss under different environmental conditions and network configurations. The datasets used in our experiments are available in the Data Folder of this repository. These datasets contain measurements obtained from stationary gateways and mobile GPS end nodes across urban environments

## Installation
To get started, you'll need to have Python and pip installed on your system. 

1) You can check if they're installed by running the following commands in your terminal:

```bash
python --version
pip --version
```
2) Clone this repository to your local machine:
```bash
git clone https://github.com/rogeriovfb/lora_path_loss_ML.git

```
3) Navigate to the project directory.

4) Install the Python dependencies using pip. It's recommended to use a virtual environment to avoid conflicts with other system dependencies:

```bash
# Create a virtual environment (optional)
python -m venv venv

# Activate the virtual environment (Windows)
venv\Scripts\activate

# Activate the virtual environment (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Data

### MCLAB Dataset
The MCLAB dataset, collected in Bonn, Germany, encompasses urban environment measurements from stationary gateways and mobile GPS end nodes. It consists of over 175,000 individual samples gathered over 230 days, covering an area of more than 200 km². The dataset underwent thorough post-processing to address GPS inaccuracies and was enriched through fusion with the EU-DEM elevation dataset. 

For access to the raw data, please refer to the original 
[GitHub repository](https://github.com/mclab-hbrs/lora-bonn).

### Medellin Dataset

The Medellin dataset is a comprehensive database spanning from October 2021 to March 2022, comprising a total of 990,750 observations. The sampling frequency was established at approximately 15 seconds, utilizing four End Nodes deployed at varying distances ranging from 2 to 8 km, communicating with a LoRaWAN Gateway.

The dataset encompasses various categories of data, including sample identification, physical and geometrical conditions of the installation (such as antenna distance and height), and experimental apparatus characteristics, facilitating the calculation of propagation losses, including transmitter power and antenna gain. Additionally, some environmental variables are recorded, such as temperature, humidity, barometric pressure, and particulate matter concentration in the air. Metrics related to radio wave propagation, such as Received Signal Strength Indicator (RSSI), Signal-to-Noise Ratio (SNR), and Time on Air (TOA), representing the time the EN remains transmitting data to the GW, are also provided.

Given the substantial volume of data, efforts were made to isolate the most relevant factor for prediction when associated with distance. This factor is the "Spreading Factor", so all algorithms were tested with two approaches, considering all data (all_data) and considering the most relevant factors (_relevant). All data are presented for both cases.

For access to the raw data, please refer to the original 
[GitHub repository](https://github.com/magonzalezudem/MDPI_LoRaWAN_Dataset_With_Environmental_Variables).


## Algorithms and Results

### Baseline

The log-distance model serves as a fundamental baseline for predicting path loss in wireless communication systems. It is based on the empirical observation that signal strength decays logarithmically with distance from the transmitter, often expressed as:

PL(d) = PL(d_0) + 10 * n * log_{10}(d/d_0) + X_f


Where:
- PL(d) is the path loss at distance d.
- PL(d_0) is the path loss at reference distance d_0.
- n is the path loss exponent.
- d is the distance from the transmitter.
- d_0 is the reference distance.
- X_f is a zero-mean Gaussian random variable representing shadowing effects.

The log-distance model assumes that the path loss exponent n and the reference path loss PL(d_0) are constants for a given environment. However, it does not account for specific environmental factors or variations in propagation characteristics.

To establish a baseline for comparison, we applied the log-distance model to both dataset and evaluated its performance in terms of Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

MCLAB dataset:
- MAE: 7.80 dBm
- RMSE: 9.78 dBm

Medellin dataset:
- MAE: 2.00 dBm
- RMSE: 2.47 dBm

Next, we present the results of the machine learning algorithms applied to the same dataset for a comprehensive comparative analysis.

### Decision Tree

MCLAB dataset:
- MAE: 5.06 dBm
- RMSE:7.05 dBm


Medellin dataset with all data:
- MAE:
- RMSE

Medellin dataset with relevant factors only:
- MAE:
- RMSE: 

### Lasso Regression

MCLAB dataset:
- MAE:7.43 dBm
- RMSE:9.41 dBm

Medellin dataset with all data:
- MAE:
- RMSE

Medellin dataset with relevant factors only:
- MAE:
- RMSE

### Neural network

MCLAB dataset:
- MAE: 6.25 dBm
- RMSE: 7.83 dBm

Medellin dataset with all data:
- MAE:
- RMSE

Medellin dataset with relevant factors only:
- MAE:
- RMSE

### Random Forest

MCLAB dataset:
- MAE: 5.23 dBm
- RMSE: 6.78 dBm

Medellin dataset with all data:
- MAE:
- RMSE

Medellin dataset with relevant factors only:
- MAE:
- RMSE

### Support Vector Regression (SVR)

MCLAB dataset (kernel RBF):
- MAE: 6.29 dBm
- RMSE:8.12 dBm

Medellin dataset with all data:
- MAE:
- RMSE:

Medellin dataset with relevant factors only:
- MAE:
- RMSE:

### XGBOOST

MCLAB dataset:
- MAE: 5.27 dBm
- RMSE: 6.79 dBm

Medellin dataset with all data:
- MAE:
- RMSE:

Medellin dataset with relevant factors only:
- MAE:
- RMSE:

