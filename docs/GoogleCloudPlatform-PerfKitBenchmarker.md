# PerfKit Benchmarker: The Open-Source Standard for Cloud Performance Evaluation

**Quickly and accurately benchmark cloud offerings with PerfKit Benchmarker, the open-source solution for consistent and reliable performance analysis. Find the original repo [here](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker).**

PerfKit Benchmarker (PKB) is an open-source project designed to provide a standardized and automated way to measure and compare the performance of various cloud services. It leverages vendor-provided command-line tools and is designed for consistent results across different platforms.

## Key Features:

*   **Automated Benchmarking:** PKB automates the entire benchmarking process, including VM instantiation, benchmark installation, and workload execution, minimizing user interaction.
*   **Cross-Cloud Compatibility:** Run benchmarks on Google Cloud Platform (GCP), Amazon Web Services (AWS), Microsoft Azure, and other major cloud providers.
*   **Extensive Benchmark Suite:** Supports a wide range of benchmarks, including network performance (iperf, ping), storage performance (fio), database performance (YCSB), and CPU/memory-intensive workloads (Coremark, HPCG).
*   **Flexible Configuration:** Customize benchmark runs with a powerful YAML-based configuration system.  Define VM groups, specify machine types, data disks, and cloud provider specifics.
*   **Result Aggregation and Publishing:** Easily collect, analyze, and publish benchmark results. Supports Elasticsearch and InfluxDB for data visualization.
*   **Open Source and Community-Driven:**  Benefit from a collaborative, open-source project with active community support and continuous improvements.

## Getting Started

To begin using PerfKit Benchmarker, follow these steps:

### 1. Prerequisites

*   **Cloud Provider Accounts:**  You'll need accounts with the cloud providers you intend to benchmark (GCP, AWS, Azure, etc.).  
*   **Python 3:** Ensure Python 3 (at least version 3.11) and `pip` are installed.
*   **Virtual Environment:**  It's recommended to use a virtual environment to manage dependencies.
    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```

### 2. Installation

*   **Clone the Repository:**
    ```bash
    $ cd $HOME
    $ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    ```
*   **Install Dependencies:**
    ```bash
    $ pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
    ```
    *   You may need to install provider-specific dependencies (e.g., for AWS):
        ```bash
        $ cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/aws
        $ pip3 install -r requirements.txt
        ```

### 3. Basic Usage

Here are some examples of how to run benchmarks:

*   **GCP Example:**
    ```bash
    ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
    ```
*   **AWS Example:**
    ```bash
    cd PerfKitBenchmarker
    ./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro
    ```
*   **Azure Example:**
    ```bash
    ./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf
    ```

Refer to the  [examples](#example-runs)  section for examples on how to run on other providers.

### 4. Tutorials

For detailed instructions, refer to the tutorials:

*   [Beginner tutorial](tutorials/beginner_walkthrough)
*   [Docker tutorial](tutorials/docker_walkthrough)

### 5. Useful Flags

Here are some of the common flags. Check the detailed documentation to discover additional flags.

| Flag               | Description                                                                          |
| ------------------ | ------------------------------------------------------------------------------------- |
| `--benchmarks`     | Comma-separated list of benchmarks or benchmark sets (e.g., `--benchmarks=iperf,ping`) |
| `--cloud`          | The cloud provider to use (GCP, AWS, Azure, etc.)                                   |
| `--machine_type`   | The machine type to provision (provider-specific)                                     |
| `--zones`          | Override the default zone for the cloud provider                                      |
| `--data_disk_type` | Specify the type of data disk (e.g., `pd-ssd`, `gp3`, `Premium_LRS`)                 |

## Advanced Features and Customization

*   **Configurations and Configuration Overrides:** Customize benchmark behavior using YAML configuration files or the `--config_override` flag.  Define VM groups, machine specifications, and cloud-specific settings.
*   **Static (Pre-Provisioned) Machines:** Run benchmarks on existing machines or local workstations.
*   **Preprovisioned Data:**  Upload data to cloud storage and use it during benchmarks. Instructions on cloud specific bucket and data uploads are available in the documentation.
*   **Elasticsearch & InfluxDB Integration:** Publish benchmark results to Elasticsearch and InfluxDB for data visualization and analysis.
*   **Extending PerfKit Benchmarker:**  Learn how to add new benchmarks, OS types, and cloud providers.

## Licensing and Benchmarks

PerfKit Benchmarker includes wrappers and definitions for popular benchmark tools. You are responsible for reviewing and accepting the licenses of the individual benchmark tools before using them.  A full list of benchmarks and their licenses is provided in the original README, [here](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker).  Some benchmarks require additional setup, such as acquiring licenses or manually providing data.  Specific instructions can be found in the original README.

## Contribute

We welcome contributions!  Refer to the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md)  file for details on how to get involved.  
*   Open an [issue](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues) to discuss potential improvements.
*   Join us on #PerfKitBenchmarker on freenode to discuss issues, ask questions, or contribute to the project.