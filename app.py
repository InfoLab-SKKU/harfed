import streamlit as st
import subprocess
import os
import time
import threading
import pandas as pd
import matplotlib.pyplot as plt
import io
import torch
from pynvml import *
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets.visualization import plot_label_distributions

class GPUTracker:
    def __init__(self, interval=1):
        self.interval = interval
        self.data = []
        self.running = False

    def _track(self):
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        while self.running:
            timestamp = time.time()
            mem = nvmlDeviceGetMemoryInfo(handle)
            util = nvmlDeviceGetUtilizationRates(handle)
            temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
            self.data.append({
                "time": timestamp,
                "memory_used_MB": mem.used // (1024 ** 2),
                "utilization_%": util.gpu,
                "temperature_C": temp
            })
            time.sleep(self.interval)
        nvmlShutdown()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._track)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def get_dataframe(self):
        return pd.DataFrame(self.data)

def dataset_config_panel():
    dataset_options = {
        "MNIST": "mnist",
        "CIFAR-10": "cifar10",
        "MedMNIST-v2": "albertvillanova/medmnist-v2",
        "Other Dataset": "OtherDataset",
    }
    subset_options = {
        "🩸 Blood MNIST": "bloodmnist",
        "🧵 Tissue MNIST": "tissuemnist",
        "🧫 Path MNIST": "pathmnist",
        "🧑‍⚕️ Derma MNIST": "dermamnist",
        "👁️ OCT MNIST": "octmnist",
        "🧠 Breast MNIST": "breastmnist",
        "🫁 Organ A MNIST": "organamnist",
        "🫀 Organ C MNIST": "organcmnist",
        "🧬 Organ S MNIST": "organsmnist",
        None: None,
    }
    with st.sidebar.expander("📊 Dataset Configuration"):
        selected_label = st.selectbox("Dataset", list(dataset_options.keys()))
        dataset = dataset_options[selected_label]
        selected_subset_label = st.selectbox("Subset", list(subset_options.keys())) if dataset == "albertvillanova/medmnist-v2" else None
        subset = subset_options[selected_subset_label]
        num_classes = {
            "mnist": 10,
            "cifar10": 10,
            "bloodmnist": 8,
            "tissuemnist": 8,
            "pathmnist": 9,
            "dermamnist": 7,
            "octmnist": 4,
            "breastmnist": 2,
            "organamnist": 11,
            "organcmnist": 11,
            "organsmnist": 11,
        }.get(subset or dataset, st.number_input("Number of Classes", min_value=1, max_value=100, value=10, step=1))
        partitioner = st.selectbox("Partitioner", ["DirichletPartitioner", "PathologicalPartitioner", "IiD"])
        if partitioner == "DirichletPartitioner":
            partition_param = "alpha"
            partition_value = st.slider("Alpha Value", 0.01, 5.0, 0.3, step=0.01)
        elif partitioner == "PathologicalPartitioner":
            partition_param = "num_classes_per_partition"
            partition_value = st.slider("Classes per Partition", 1, num_classes, 2, step=1)
        else:
            st.sidebar.info("No partition parameter required for IID.")
            partition_param = None
            partition_value = None
    return dataset, subset, num_classes, partitioner, partition_param, partition_value

def attack_config_panel():
    with st.sidebar.expander("🛡️ Attack Configuration"):
        attack = st.selectbox("Attack Type", ["targeted-lf", "adaptive-targeted-lf", "untargeted-lf", "none"])
        fraction_mal_cli = st.slider("Fraction of Malicious Clients", 0.0, 1.0, 0.3, step=0.01) if attack != "none" else 0.0
    return attack, fraction_mal_cli

def privacy_config_panel():
    with st.sidebar.expander("🔒 Differential Privacy", expanded=False):
        epsilon = st.slider("LDP Epsilon", 0.0, 1.0, 0.0, step=0.01)
    return epsilon

def strategy_config_panel(num_classes):
    with st.sidebar.expander("⚙️ Federated Strategy Configuration", expanded=False):
        strategy = st.selectbox("Strategy", ["FedAvg", "FedAvgM", "FedNova", "FedProx"])
        model = st.selectbox("Model", ["mobilenet_v3_small", "mobilenet_v3_large", "mobilenet_v2", "resnet18", "vgg16", "efficientnet_b0", "resnet101"])
        num_clients = st.number_input("Number of Clients", min_value=3, max_value=10000, value=50, step=1)
        fraction_train_clients = st.slider("Fraction of Training Clients", 0.01, 1.0, 0.1, step=0.01)
        num_rounds = st.number_input("Number of Rounds", min_value=3, max_value=10000, value=5)
    return strategy, model, num_clients, fraction_train_clients, num_rounds

def gpu_config_panel():
    def get_gpu_status():
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,nounits,noheader"])
            lines = output.decode("utf-8").strip().split("\n")
            return [tuple(map(int, line.split(","))) for line in lines]
        except Exception:
            return []
    with st.sidebar.expander("🖥️ GPU Configuration", expanded=True):
        gpu_stats = get_gpu_status()
        for gpu_id, mem_used, mem_total in gpu_stats:
            status = "🟢 In Use" if mem_used > 50 else "⚪ Idle"
            st.write(f"**GPU {gpu_id}** — {status} | Used: {mem_used} MB / {mem_total} MB")
        use_cuda = st.checkbox("Use CUDA", value=True)
        use_cuda = use_cuda and torch.cuda.is_available()
        if use_cuda:
            st.sidebar.success("CUDA is enabled. GPU will be used for training.")
            available_gpus = [f"GPU {i}" for i in range(torch.cuda.device_count())]
            selected_gpus = st.multiselect("Select GPUs", options=available_gpus, default=[f"GPU 0"])
            selected_gpu_ids = [gpu.split(" ")[1] for gpu in selected_gpus]
            cuda_visible_devices = ",".join(selected_gpu_ids)
            st.code(f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}", language="bash")
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
            st.success(f"Set CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
            current_device = torch.cuda.current_device()
            st.write(f"🚀 PyTorch is using: GPU {current_device} — {torch.cuda.get_device_name(current_device)}")
            num_gpus = st.slider("GPU Resources per Client", 0.01, 1.0, 1.0, step=0.01, 
                                help="Controls the fraction of GPU resources allocated to each client")
        else:
            num_gpus = 0
            st.sidebar.warning("CUDA is not available. Training will use CPU only.")
    return use_cuda, selected_gpus if use_cuda else [], num_gpus

def show_summary(dataset, subset, partitioner, partition_param, partition_value, attack, epsilon, fraction_mal_cli, strategy, fraction_train_clients, model, num_classes, num_clients, num_rounds):
    with st.expander("📝 Summary of Experiment Configuration"):
        # Create three columns for better organization
        col1, col2, col3 = st.columns(3)
        
        # Dataset configuration in first column
        with col1:
            st.markdown("### 📊 Dataset")
            st.markdown(f"**Dataset:** {dataset}")
            if subset:
                st.markdown(f"**Subset:** {subset}")
            st.markdown(f"**Partitioner:** {partitioner}")
            if partition_param:
                st.markdown(f"**{partition_param.capitalize()}:** {partition_value}")
            st.markdown(f"**Classes:** {num_classes}")
        
        # Attack and privacy configuration in second column
        with col2:
            st.markdown("### 🛡️ Security & Privacy")
            st.markdown(f"**Attack Type:** {attack}")
            st.markdown(f"**LDP Epsilon:** {epsilon}")
            st.markdown(f"**Malicious Fraction:** {fraction_mal_cli}")
        
        # Training configuration in third column
        with col3:
            st.markdown("### ⚙️ Training")
            st.markdown(f"**Strategy:** {strategy}")
            st.markdown(f"**Model:** {model}")
            st.markdown(f"**Clients:** {num_clients}")
            st.markdown(f"**Training Fraction:** {fraction_train_clients}")
            st.markdown(f"**Rounds:** {num_rounds}")

def visualize_partitions(dataset, subset, partitioner, partition_param, partition_value, num_clients):
    st.markdown("## 📊 Data Distribution Visualization")
    st.markdown("---")
    
    visualize = st.checkbox("Enable Dataset Partition Visualization", value=False)
    if not visualize:
        st.info("Enable the checkbox above to visualize how data is distributed across clients.")
        return
        
    with st.spinner("Generating partition visualization..."):
        num_partitions = int(num_clients)
    if dataset == "albertvillanova/medmnist-v2":
        if partitioner == "DirichletPartitioner":
            fds = FederatedDataset(
                dataset=dataset,
                subset=subset,
                partitioners={
                    "train": DirichletPartitioner(
                        num_partitions=num_partitions,
                        partition_by="label",
                        alpha=float(partition_value),
                        seed=42,
                        min_partition_size=0,
                    ),
                },
            )
        elif partitioner == "PathologicalPartitioner":
            from flwr_datasets.partitioner import PathologicalPartitioner
            fds = FederatedDataset(
                dataset=dataset,
                subset=subset,
                partitioners={
                    "train": PathologicalPartitioner(
                        num_partitions=num_partitions,
                        partition_by="label",
                        seed=42,
                        num_classes_per_partition=int(partition_value),
                    ),
                },
            )
        elif partitioner == "IiD":
            fds = FederatedDataset(
                dataset=dataset,
                subset=subset,
                partitioners={"train": num_partitions}
            )
        else:
            st.info("Label distribution visualization is not supported for this partitioner. Skipping..")
            return
    else:
        if partitioner == "DirichletPartitioner":
            fds = FederatedDataset(
                dataset=dataset,
                partitioners={
                    "train": DirichletPartitioner(
                        num_partitions=num_partitions,
                        partition_by="label",
                        alpha=float(partition_value),
                        seed=42,
                        min_partition_size=0,
                    ),
                },
            )
        elif partitioner == "PathologicalPartitioner":
            from flwr_datasets.partitioner import PathologicalPartitioner
            fds = FederatedDataset(
                dataset=dataset,
                partitioners={
                    "train": PathologicalPartitioner(
                        num_partitions=num_partitions,
                        partition_by="label",
                        seed=42,
                        num_classes_per_partition=int(partition_value),
                    ),
                },
            )
        elif partitioner == "IiD":
            fds = FederatedDataset(
                dataset=dataset,
                partitioners={"train": num_partitions}
            )
        else:
            st.info("Label distribution visualization is not supported for this partitioner. Skipping..")
            return
    partitionerp = fds.partitioners["train"]
    fig, ax, df = plot_label_distributions(
        partitionerp,
        label_name="label",
        plot_type="bar",
        size_unit="absolute",
        partition_id_axis="x",
        legend=True,
        verbose_labels=True,
        title="Per Partition Labels Distribution",
    )
    # Enhance the figure
    fig.set_size_inches(20, 6)
    plt.tight_layout()
    
    # Add grid lines and styling
    # for ax_i in ax:
    #     ax_i.grid(True, linestyle='--', alpha=0.7)
    #     ax_i.set_axisbelow(True)
    
    # Display the visualization in a container
    with st.container():
        st.markdown("### 📊 Label Distribution Across Clients")
        st.pyplot(fig, clear_figure=True, bbox_inches='tight')
        
        # Add download options
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        st.download_button("📥 Download Visualization (PNG)", buf, "partition_visualization.png", mime="image/png")
    
    # Show data distribution details in an expander
    with st.expander("📋 View Label Distribution Data"):
        st.dataframe(df, use_container_width=True)
        
        # Add CSV download option
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button("📥 Download Distribution Data (CSV)", csv_buffer.getvalue(), "label_distribution.csv", mime="text/csv")
    
    # Sample data section with better UI
    show_samples = st.checkbox("Show Sample Data from Client 0", value=False)
    if show_samples:
        with st.container():
            st.markdown("### 🖼️ Sample Data from Client 0")
            
            try:
                with st.spinner("Loading sample data..."):
                    sample_ds = fds.load_partition(partition_id=0, split="train")
                    sample_df = pd.DataFrame(sample_ds[:10])
                
                if "image" in sample_df.columns:
                    # Create a better image gallery
                    st.markdown("#### Sample Images")
                    
                    # Use 5 columns for better display on various screen sizes
                    num_cols = min(5, len(sample_df))
                    cols = st.columns(num_cols)
                    
                    for i, row in enumerate(sample_df.itertuples()):
                        with cols[i % num_cols]:
                            st.image(row.image, caption=f"Label: {row.label}", use_container_width=True)
                            
                    # Show tabular data in an expander
                    with st.expander("View Sample Data as Table"):
                        st.dataframe(sample_df, use_container_width=True)
                else:
                    # For non-image data, show as a table with styling
                    st.markdown("#### Sample Data")
                    st.dataframe(sample_df, use_container_width=True)
            except Exception as e:
                st.error(f"Could not load samples: {e}")

def run_experiment(strategy, model, dataset, partitioner, partition_value, subset, num_rounds, num_clients, fraction_train_clients, num_gpus, partition_param, num_classes, epsilon, fraction_mal_cli, attack):
    st.markdown("## 🚀 Launch Experiment")
    st.markdown("---")
    
    # Create a nice container for the launch controls
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Output directory display
            output_dir = f"outputs/{strategy}/{model}/{dataset}/{partitioner}/{partition_value or 'iid'}"
            st.info(f"📁 Results will be saved to: **{output_dir}**")
            
            # Command construction
            cmd = (
                f"python -m src.main "
                f"hydra.run.dir={output_dir} "
                f"dataset.name={dataset} "
                f"{f'dataset.subset={subset} ' if subset else ''}"
                f"strategy.name={strategy} "
                f"model.name={model} "
                f"dataset.partitioner.name={partitioner} "
                f"strategy.num_rounds={num_rounds} "
                f"client.count={num_clients} "
                f"strategy.fraction_train_clients={fraction_train_clients} "
                f"client.resources.num_gpus={num_gpus} "
                f"poisoning.name={attack} "
                f"poisoning.fraction={fraction_mal_cli} "
            )
            if partition_param:
                cmd += f"dataset.partitioner.{partition_param}={partition_value} "
            cmd += (
                f"model.num_classes={num_classes} "
                f"ldp.epsilon={epsilon} "
                f"poisoning.fraction={fraction_mal_cli} "
            )
            
            # Command display in expander
            with st.expander("🧾 View Command", expanded=False):
                st.code(cmd, language="bash")
        
        with col2:
            # Controls
            show_logs = st.checkbox("Show Live Logs", value=True)
            submitted = st.button("▶️ Run Experiment", use_container_width=True, type="primary")
    
    if not submitted:
        return
        
    # Progress indication
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.info("🔄 Starting experiment...")
    
    # Start GPU tracking
    tracker = GPUTracker(interval=1)
    tracker.start()
    
    try:
        # Create log placeholder
        log_container = st.container()
        with log_container:
            st.markdown("### 📋 Experiment Logs")
            log_placeholder = st.empty()
        
        # Run the process
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        if show_logs:
            logs = ""
            log_count = 0
            for line in process.stdout:
                if "[flwr][INFO]" in line:
                    logs += line
                    log_count += 1
                    # Update progress based on log count (approximate)
                    if "round" in line.lower() and "completed" in line.lower():
                        try:
                            current_round = int(line.split("round")[1].split("/")[0].strip())
                            total_rounds = int(line.split("round")[1].split("/")[1].split()[0].strip())
                            progress = min(0.95, current_round / total_rounds)
                            progress_bar.progress(progress)
                            status_text.info(f"🔄 Running round {current_round}/{total_rounds}...")
                        except:
                            # If parsing fails, just increment progress
                            progress_bar.progress(min(0.95, log_count / (num_rounds * 2 + 5)))
                    
                    log_placeholder.code(logs, language="bash", height=300)
        
        process.wait()
        
        # Final status
        if process.returncode == 0:
            progress_bar.progress(1.0)
            status_text.success("✅ Experiment completed successfully!")
        else:
            status_text.error("❌ Experiment failed. Check logs for details.")
    finally:
        tracker.stop()
        df = tracker.get_dataframe()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "gpu_stats.csv")
        df.to_csv(csv_path, index=False)
        
        # Show GPU stats
        with st.expander("📊 GPU Statistics", expanded=False):
            if not df.empty:
                # Plot GPU metrics
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
                
                # Memory usage
                ax1.plot(df['time'] - df['time'].iloc[0], df['memory_used_MB'], 'b-')
                ax1.set_ylabel('Memory (MB)')
                ax1.set_title('GPU Memory Usage')
                ax1.grid(True)
                
                # Utilization
                ax2.plot(df['time'] - df['time'].iloc[0], df['utilization_%'], 'g-')
                ax2.set_ylabel('Utilization (%)')
                ax2.set_title('GPU Utilization')
                ax2.grid(True)
                
                # Temperature
                ax3.plot(df['time'] - df['time'].iloc[0], df['temperature_C'], 'r-')
                ax3.set_xlabel('Time (seconds)')
                ax3.set_ylabel('Temperature (°C)')
                ax3.set_title('GPU Temperature')
                ax3.grid(True)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show raw data
                st.dataframe(df)
                
                # Download option
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button("📥 Download GPU Stats", csv_buffer.getvalue(), "gpu_stats.csv", mime="text/csv")
            else:
                st.warning("No GPU statistics were collected.")
    
    # Show results
    show_results(output_dir)

def show_results(output_dir):
    csv_path = os.path.join(output_dir, "results.csv")
    st.markdown("## 📊 Experiment Results")
    st.markdown("---")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # Display metrics summary if possible
        if {'Round', 'accuracy', 'asr'}.issubset(df.columns):
            # Get final metrics
            final_metrics = df.iloc[-1]
            final_round = int(final_metrics['Round'])
            final_accuracy = final_metrics['accuracy']
            final_asr = final_metrics['asr']
            
            # Create metrics display
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Final Round", final_round)
            with col2:
                st.metric("Final Accuracy", f"{final_accuracy:.4f}")
            with col3:
                st.metric("Final ASR", f"{final_asr:.4f}")
            
            # Create visualization
            st.markdown("### Training Progress")
            fig, ax = plt.subplots(figsize=(15, 5))
            ax.plot(df['Round'], df['accuracy'], 'b-', linewidth=2, label='Accuracy')
            ax.plot(df['Round'], df['asr'], 'r-', linewidth=2, label='ASR')
            ax.set_xlabel('Round', fontsize=12)
            ax.set_ylabel('Metric Value', fontsize=12)
            ax.set_ylim(0, 1)
            ax.set_title('Accuracy and Attack Success Rate (ASR) over Training Rounds', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=12)
            
            # Add some styling
            plt.tight_layout()
            st.pyplot(fig)
            
            # Add download options in a cleaner format
            col1, col2 = st.columns(2)
            
            # Prepare buffers for downloads
            buf_png = io.BytesIO()
            fig.savefig(buf_png, format='png', dpi=300, bbox_inches='tight')
            buf_png.seek(0)
            
            buf_pdf = io.BytesIO()
            fig.savefig(buf_pdf, format='pdf', bbox_inches='tight')
            buf_pdf.seek(0)
            
            with col1:
                st.download_button("📥 Download Graph (PNG)", buf_png, "training_progress.png", mime="image/png")
            with col2:
                st.download_button("📥 Download Graph (PDF)", buf_pdf, "training_progress.pdf", mime="application/pdf")
            
            # Display raw data with expander
            with st.expander("View Raw Data"):
                st.dataframe(df, use_container_width=True)
                
                # Add CSV download option
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button("📥 Download CSV", csv_buffer.getvalue(), "results.csv", mime="text/csv")
        else:
            st.warning("⚠️ CSV must include 'Round', 'accuracy', and 'asr' columns for visualization.")
            st.dataframe(df, use_container_width=True)
    else:
        st.warning("⚠️ No results.csv found. Experiment may not have logged output.")

# --- Main ---
st.set_page_config(page_title="Harfed - FL Experiment Similator", page_icon="🔄", layout="wide")
st.title("🔄 Harfed: Federated Learning Experiment Similator")
st.markdown("*Heterogeneity, Attacks, and Robustness in Federated Learning*")
st.markdown("---")
dataset, subset, num_classes, partitioner, partition_param, partition_value = dataset_config_panel()
attack, fraction_mal_cli = attack_config_panel()
epsilon = privacy_config_panel()
strategy, model, num_clients, fraction_train_clients, num_rounds = strategy_config_panel(num_classes)
use_cuda, selected_gpus, num_gpus = gpu_config_panel()
show_summary(dataset, subset, partitioner, partition_param, partition_value, attack, epsilon, fraction_mal_cli, strategy, fraction_train_clients, model, num_classes, num_clients, num_rounds)
visualize_partitions(dataset, subset, partitioner, partition_param, partition_value, num_clients)
run_experiment(strategy, model, dataset, partitioner, partition_value, subset, num_rounds, num_clients, fraction_train_clients, num_gpus, partition_param, num_classes, epsilon, fraction_mal_cli, attack)