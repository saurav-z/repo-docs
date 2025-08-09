# PerfKit Benchmarker: The Open-Source Standard for Cloud Performance Measurement

**Tired of comparing cloud providers based on marketing hype? PerfKit Benchmarker automates performance testing, providing you with objective, reproducible results.** ([Original Repo](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker))

PerfKit Benchmarker (PKB) is an open-source project designed to provide a consistent and reliable way to measure and compare the performance of various cloud offerings. It achieves this by automating the deployment and execution of a wide range of industry-standard benchmarks across different cloud providers, using their native command-line tools.  PKB ensures consistent results, providing insights for informed cloud decisions.

## Key Features:

*   **Automated Benchmarking:** Automates the entire benchmarking process, from VM instantiation to result collection.
*   **Cloud Provider Agnostic:** Supports major cloud providers, including GCP, AWS, Azure, and others.
*   **Diverse Benchmark Suite:** Offers a comprehensive set of benchmarks, covering various workloads like CPU, storage, network, and databases.
*   **Reproducible Results:** Provides a consistent testing environment for reliable comparisons.
*   **Customizable Configurations:** Allows users to tailor benchmark settings and configurations to their specific needs.
*   **Detailed Reporting:** Generates comprehensive reports and supports publishing to Elasticsearch and InfluxDB.
*   **Open Source and Community-Driven:** Benefit from an active community and contribute to the project's evolution.

## Getting Started

### 1. Prerequisites

*   Python 3.11 or higher
*   Access to the desired cloud provider(s) accounts.
*   Basic knowledge of command-line tools.

### 2. Installation

1.  **Create a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    cd PerfKitBenchmarker
    ```

3.  **Install Dependencies:**
    ```bash
    pip3 install -r requirements.txt
    ```
    Install provider-specific requirements:
    ```bash
    cd perfkitbenchmarker/providers/aws  # replace aws with the cloud provider
    pip3 install -r requirements.txt
    ```

### 3. Running a Benchmark

PKB is designed to be run with a variety of cloud providers. Here are a few examples:
```bash
# Google Cloud Platform (GCP)
./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro

# Amazon Web Services (AWS)
./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro

# Microsoft Azure
./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf
```

*   **For more examples and complete instructions, refer to the sections on "Installation and Setup" and "Running a Single Benchmark" in the original README.**

## Useful Flags

| Flag               | Description                                                                                                                                |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `--helpmatch=pkb`  | Displays all global flags.                                                                                                                 |
| `--helpmatch=hpcc` | Displays flags specific to the "hpcc" benchmark (replace "hpcc" with any other benchmark name to see its associated flags).                |
| `--benchmarks`     | Comma-separated list of benchmarks or benchmark sets (e.g., `--benchmarks=iperf,ping`). Run `./pkb.py --helpmatch=benchmarks` for the full list. |
| `--cloud`          | Specifies the cloud provider (GCP, AWS, Azure, etc.).  See the table in the original README for available zones and their defaults.            |
| `--machine_type`   | Specifies the machine type.  See the table in the original README for examples.                                                           |
| `--zones`          | Overrides the default zone. See the table in the original README for provider-specific options.                                           |
| `--data_disk_type` | Specifies the data disk type (e.g., `pd-ssd`, `gp3`). See the original README for examples.                                                  |

## Configuration and Customization

PKB allows you to customize your benchmarks through YAML configuration files.  Use `--benchmark_config_file` to provide your own custom configs.  For a deep-dive, review the section on "Configurations and Configuration Overrides" in the original README.

## Advanced Usage

*   **Running Benchmarks Without Cloud Provisioning:**  Test performance locally using static VMs. See the "Advanced: How To Run Benchmarks Without Cloud Provisioning" section in the original README for detailed instructions.
*   **Extending PerfKit Benchmarker:**  Contribute to the project and add new benchmarks, providers, or OS support. See the "How to Extend PerfKit Benchmarker" and  [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) for more information.

## Licensing and Benchmarks

PerfKit Benchmarker incorporates various benchmark tools. Please review the licenses associated with each benchmark before use, as detailed in the original README.

## Additional Resources

*   [Wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki): Detailed documentation, FAQs, and design documents.
*   [Issues](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues): Report issues, request features, and contribute to the project.
*   [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md):  Guidance on contributing to the project.
*   [FAQ](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/FAQ)
*   [Tech Talks](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Tech-Talks)
*   [Governing rules](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Governing-Rules)
*   [Community meeting decks and notes](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Community-Meeting-Notes-Decks)
*   [Design documents](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Design-Docs)