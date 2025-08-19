# PerfKit Benchmarker: The Open-Source Standard for Cloud Performance Benchmarking

**Easily measure and compare cloud offerings with PerfKit Benchmarker, an open-source tool designed for consistent and automated benchmarking across platforms.** ([Original Repo](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker))

PerfKit Benchmarker (PKB) is your go-to solution for evaluating cloud performance. This powerful tool automates the entire benchmarking process, from VM provisioning to workload execution, providing standardized and repeatable results.

## Key Features:

*   **Automated Benchmarking:**  Instantiates VMs, installs benchmarks, and runs workloads without manual intervention.
*   **Vendor-Agnostic:** Designed to operate via vendor-provided command-line tools, ensuring consistent results across providers.
*   **Extensive Benchmark Suite:**  Supports a wide range of benchmarks, including Aerospike, Bonnie++, FIO, iperf, and more (full list below).
*   **Cloud Provider Support:**  Runs benchmarks on major cloud providers like GCP, AWS, Azure, DigitalOcean, and others, with Kubernetes and OpenStack support.
*   **Configuration Flexibility:**  Offers YAML-based configurations for fine-grained control over benchmarks, VM types, and disk settings.
*   **Data Pre-provisioning:**  Supports pre-provisioning of data for benchmarks requiring it.
*   **Advanced Features:** Includes options for static machine runs, Juju integration, selective stage execution, and result publishing to Elasticsearch and InfluxDB.

## Quick Start:

1.  **Prerequisites:**  Ensure you have Python 3 (version 3.11 or higher) and cloud provider accounts.
2.  **Setup Virtual Environment:** Create and activate a virtual environment:
    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```
3.  **Install PKB:** Clone the repository and install dependencies:
    ```bash
    git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    cd PerfKitBenchmarker
    pip3 install -r requirements.txt
    ```
4.  **Run a Benchmark:**  Choose a cloud provider and run a benchmark (e.g., iperf on GCP):
    ```bash
    ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
    ```
    See below for examples on other clouds.

## Available Benchmarks:

PKB provides wrappers and workload definitions for a wide array of popular benchmark tools.  **Important:**  You must accept the licenses of each benchmark before use (use the `--accept-licenses` flag).  A full list of the benchmarks supported by PerfKit Benchmarker is provided in the original README, but here is a summarized version with an attempt to group them logically:

**Storage and Disk Benchmarks:**

*   `bonnie++`
*   `copy_throughput`
*   `fio`

**Network Benchmarks:**

*   `iperf`
*   `mesh_network`
*   `netperf`
*   `ping`

**Database Benchmarks:**

*   `aerospike`
*   `cassandra_ycsb`
*   `cassandra_stress`
*   `mongodb_ycsb`
*   `pgbench`
*   `sysbench_oltp`
*   `silo`

**Compute and System Benchmarks:**

*   `coremark`
*   `hpcc`
*   `hpcg`
*   `scimark2`
*   `speccpu2006`
*   `unixbench`

**Application Benchmarks:**

*   `cluster_boot`
*   `cloudsuite3.0`
*   `hadoop_terasort`
*   `memtier_benchmark`
*   `oldisim`
*   `object_storage_service`
*   `TensorFlow`
*   `tomcat`
*   `wrk`
*   `ycsb`

**GPU Benchmarks**

*   `gpu_pcie_bandwidth`
*   `SHOC`

## Example Runs:

Below are example commands to get you started on different cloud providers:

*   **GCP:**
    ```bash
    ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
    ```
*   **AWS:**
    ```bash
    ./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro
    ```
*   **Azure:**
    ```bash
    ./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf
    ```
*   **DigitalOcean:**
    ```bash
    ./pkb.py --cloud=DigitalOcean --machine_type=16gb --benchmarks=iperf
    ```
    ... and many more! (See the original README for a complete list.)

## Advanced Usage:

*   **Configuration Files:** Customize your runs using YAML configuration files for detailed control over VM groups, machine types, and more.
*   **Preprovisioned Data:** Upload data to cloud storage for benchmarks requiring pre-provisioned data.
*   **Integration with Juju:** Use the `--os_type=juju` flag for Juju-managed deployments.
*   **Result Publishing:** Publish results to Elasticsearch and InfluxDB for analysis and visualization.
*   **Extend PerfKit:** Contribute new benchmarks, cloud provider support, and features. Refer to the `CONTRIBUTING.md` file.

## Useful Flags:

| Flag               | Description                                                                       |
| ------------------ | --------------------------------------------------------------------------------- |
| `--helpmatch=pkb`  | Displays all global flags.                                                       |
| `--helpmatch=<benchmark>` | Displays flags specific to a given benchmark.                                  |
| `--benchmarks`     | Comma-separated list of benchmarks or benchmark sets to run.                       |
| `--cloud`          | Specifies the cloud provider (GCP, AWS, Azure, etc.).                             |
| `--machine_type`   | Specifies the machine type. Provider-specific names or YAML expressions are accepted.|
| `--zones`          | Overrides the default zone for the cloud provider.                                 |
| `--data_disk_type` | Specifies the disk type (e.g., pd-ssd, gp3, Premium_LRS, etc.).                 |

## Additional Resources:

*   [Wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki) - Provides detailed information on various aspects of PKB.
*   [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) - Learn how to contribute to the project.
*   [Issues](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues) - Report issues or suggest improvements.