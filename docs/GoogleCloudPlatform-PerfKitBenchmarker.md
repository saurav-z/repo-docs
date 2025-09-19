# PerfKit Benchmarker: Your Go-To Tool for Cloud Performance Benchmarking

**PerfKit Benchmarker is a powerful open-source tool designed to measure and compare the performance of various cloud offerings.** Leverage its automated capabilities to deploy benchmarks, collect results, and gain valuable insights into cloud performance. Check out the [original repo](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker) for more details!

## Key Features

*   **Automated Benchmarking:** Automates VM instantiation, benchmark installation, and workload execution across various cloud providers.
*   **Vendor-Agnostic:** Designed to operate with command-line tools, providing consistent results regardless of the cloud platform.
*   **Flexible Configuration:** Easily configure and customize benchmarks via YAML files and command-line overrides.
*   **Comprehensive Benchmark Suite:** Includes a wide range of benchmarks covering CPU, storage, network, and database performance.
*   **Extensible:** Add new benchmarks, cloud providers, and features with comprehensive documentation and developer support.
*   **Integration Testing:** Built in integration testing for comprehensive analysis and compatibility.

## Getting Started

### Installation and Setup

1.  **Prerequisites:**
    *   Python 3 (version 3.11 or higher recommended)
    *   Account(s) on the cloud provider(s) you want to benchmark.
    *   Necessary command-line tools and credentials for cloud access.

2.  **Virtual Environment (Recommended):**
    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```

3.  **Install PerfKit Benchmarker:**
    ```bash
    $ cd $HOME
    $ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    $ pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
    ```

4.  **Cloud-Specific Dependencies (if needed):** Install provider-specific dependencies if you are using them. For example for AWS:
    ```bash
    $ cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/aws
    $ pip3 install -r requirements.txt
    ```

### Running a Single Benchmark

Run iperf to test your network bandwidth and latency:

*   **Example: Google Cloud Platform (GCP)**
    ```bash
    ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
    ```

*   **Example: AWS**
    ```bash
    ./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro
    ```

*   **Example: Azure**
    ```bash
    ./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf
    ```

*   **Other Cloud Providers**: AliCloud, DigitalOcean, OpenStack, Kubernetes, Mesos, CloudStack, Rackspace, ProfitBricks.

### Useful Global Flags

*   `--helpmatch=pkb`: Displays all global flags.
*   `--helpmatch=<benchmark_name>`: Displays flags specific to a benchmark.
*   `--benchmarks`: Specifies a comma-separated list of benchmarks to run (e.g., `--benchmarks=iperf,ping`).
*   `--cloud`: Specifies the cloud provider (e.g., `--cloud=GCP`).
*   `--machine_type`: Specifies the virtual machine instance type.
*   `--zones`: Overrides the default zone for a cloud provider (e.g., `--zones=us-central1-a`).
*   `--data_disk_type`: Specifies the type of data disk to use.

### Configuration and Configuration Overrides

Customize benchmarks using YAML configuration files. Override default settings with the `--config_override` flag.

Example:

```bash
./pkb.py --benchmarks=cluster_boot --machine_type=n1-standard-2 --zones=us-central1-f --run_stage=provision,prepare,run
```

### Run all benchmarks:

```bash
./pkb.py --benchmarks="standard_set"
```

## Licensing

PerfKit Benchmarker utilizes various open-source benchmarks. Please review the licenses of each benchmark before use. A list of the benchmarks with licenses is detailed within the original README.

## Extending PerfKit Benchmarker

*   Contribute to the project by submitting pull requests and add new benchmarks and cloud providers.
*   Find more information and documentation via the [wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki).
*   Report any issues and contribute in the issues and pull requests.