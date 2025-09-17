# PerfKit Benchmarker: The Open-Source Standard for Cloud Benchmarking ([Original Repo](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker))

PerfKit Benchmarker (PKB) is an open-source project designed to provide a standardized, automated, and customizable framework for measuring and comparing the performance of cloud offerings. PKB automates the setup, execution, and analysis of a comprehensive suite of benchmarks across various cloud providers, allowing for consistent and reliable performance evaluations.

**Key Features:**

*   **Automated Benchmarking:** Automatically provisions VMs, installs benchmarks, and executes workloads.
*   **Vendor-Agnostic:** Supports major cloud providers (GCP, AWS, Azure, etc.) and custom environments.
*   **Extensive Benchmark Suite:** Includes a wide range of popular benchmarks (Aerospike, FIO, iperf, etc.) to assess different aspects of performance.
*   **Configuration Flexibility:**  Allows users to customize benchmarks using YAML configuration files or command-line overrides.
*   **Integration Testing:** Includes a suite of integration tests to ensure the proper functionality of the project

## Getting Started

### Installation and Setup
1.  **Prerequisites:** Ensure you have Python 3 (version 3.11 or later) installed.  It is recommended to install and run PKB within a virtual environment.
    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```
2.  **Clone the Repository:**
    ```bash
    cd $HOME
    git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    ```
3.  **Install Dependencies:**
    ```bash
    pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
    ```
    Install provider-specific dependencies for your target cloud(s), as needed.  For AWS, for example:
    ```bash
    cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/aws
    pip3 install -r requirements.txt
    ```

### Example Run

Here's a quick example to run the `iperf` benchmark on GCP:

```bash
./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
```

### Key Flags and Options:
*   `--benchmarks`:  Specify a comma-separated list of benchmarks to run (e.g., `--benchmarks=iperf,ping`) or a benchmark set (e.g., `--benchmarks="standard_set"`).
*   `--cloud`:  Select your cloud provider (e.g., `--cloud=GCP`, `--cloud=AWS`).
*   `--machine_type`:  Choose the machine type for your instances (e.g., `--machine_type=n1-standard-2`).
*   `--zones`:  Set the zone to run the benchmark in (e.g., `--zones=us-central1-a`).
*   `--data_disk_type`: Define the disk type to use (e.g., `--data_disk_type=pd-ssd`).

## Running Benchmarks

PKB can run benchmarks on a variety of clouds. Example runs on GCP, AWS, and Azure are as follows:

#### Example run on GCP
```bash
./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
```
#### Example run on AWS
```bash
cd PerfKitBenchmarker
./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro
```

#### Example run on Azure
```bash
./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf
```

## Extending PerfKit Benchmarker

Contribute new benchmarks, provider support, or features. See the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) file and the [wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki) for detailed documentation.  Issues can also be opened to request new features and documentation or to address any issues.

## Licenses

PKB leverages various open-source benchmark tools. Each benchmark's licensing terms apply.  Acceptance of these licenses is required before running benchmarks using the `--accept-licenses` flag.