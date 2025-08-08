# HARFed: Federated Learning Experiment Similator
Heterogeneity, Attacks, and Robustness in Federated Learning
---

## Overview

The Harfed application provides a streamlined interface for setting up and conducting Federated Learning experiments, particularly focusing on:
1. **Heterogeneous data distributions**: Support for advanced dataset partitioning strategies like Dirichlet and Pathological.
2. **Robustness testing**: Configuring experiments with malfunctions or adversarial behavior (e.g., targeted or untargeted attacks).
3. **Monitoring and visualization**: GPU utilization tracking, dataset partition visualizations, accuracy trends, and attack success rates over training rounds.

---

## Features

### **Main Functionalities**:
- **Dataset Configuration**: 
  - Selection of datasets and subsets.
  - Customizable partitioning strategies (e.g., Dirichlet, IID, etc.).
- **Attack Configuration**:
  - Integration of Local Differential Privacy (LDP) settings.
  - Fraction of malicious clients.
- **Federated Learning Strategies**:
  - Popular FL algorithms like FedAvg, FedMedian, FedProx, etc.
  - Flexible settings for model architectures, client fractions, and training rounds.
- **Real-time Monitoring**:
  - GPU resource tracking (memory, utilization, temperature).
  - Live experiment logs.
- **Results Visualization**:
  - Accuracy and attack success rate (ASR) plots over FL rounds.
  - Dataset partition label distribution charts.

---

## Usage Instructions

1. **Install dependencies**:
   Ensure you have Python 3.10+ and install the required packages using `pip`:
   ```
   pip install -r requirements.txt
   ```

2. **Launch the Streamlit app**:
   Run the following command to start the Streamlit application:
   ```
   streamlit run run_experiment.py
   ```

3. **Configure the experiment**:
   - Use the sidebar to select the dataset, partitioner, and attack configurations.
   - Set Federated Learning strategies, the number of clients, and training rounds as needed.

4. **Visualize and monitor**:
   - Enable dataset partition visualization for insights into the data distribution.


   - Observe live GPU metrics and logs while the experiment is running.

5. **Run the experiment**:
   - Click on the "Run Experiment" button to execute the experiment.
   - The app generates logs and results at the specified output directory.

6. **View and download results**:
   - Check the results section for accuracy and ASR plots.
   - Download visualization graphs in PNG or PDF format.
   - Access raw results and GPU stats as CSV files.

---

## Screenshots

### **1. Dataset Partition Visualization**
![Screenshot 1 - Dataset Configuration](screenshots/Screenshot%20from%202025-08-06%2013-31-32.png)

![Screenshot 2 - Dataset Partition Visualization](screenshots/Screenshot%20from%202025-08-06%2013-32-52.png)

### **2. Real-time Monitoring**
![Screenshot 3 - Real-time Monitoring](screenshots/Screenshot%20from%202025-08-06%2013-37-49.png)

### **3. Results Visualization**
![Screenshot 4 - Results Visualization](screenshots/Screenshot%20from%202025-08-06%2013-40-47.png)
![Screenshot 5 - Results Visualization](screenshots/Screenshot%20from%202025-08-06%2013-41-07.png)
