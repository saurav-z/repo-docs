# PerfKit Benchmarker: Cloud Benchmark Your Way to Performance Excellence

**PerfKit Benchmarker (PKB) is your go-to, open-source tool for consistently measuring and comparing the performance of cloud offerings.**  This README provides a comprehensive overview to get you started, with links to detailed resources within the [original repository](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker).

**Key Features:**

*   **Automated Cloud Benchmarking:** PKB automates the provisioning of VMs, installing benchmarks, and running workloads.
*   **Vendor-Agnostic:** Designed to operate with vendor-provided command-line tools, ensuring fair and consistent comparisons.
*   **Customizable:** Offers a wide array of configuration options and overrides to fine-tune benchmarks for specific needs.
*   **Comprehensive Benchmark Suite:** Includes a diverse set of popular benchmarks, covering various aspects of cloud performance.
*   **Flexible Deployment:** Supports benchmarking on multiple cloud providers (GCP, AWS, Azure, etc.) and local machines.
*   **Reporting & Visualization:**  Integrates with Elasticsearch and InfluxDB for powerful result analysis.

---

## Getting Started

### Prerequisites

*   **Account(s):**  Active accounts on the cloud provider(s) you plan to benchmark.
*   **Python 3:**  Install Python 3 (version 3.11 or higher) and the `venv` module (most Linux and recent macOS installations have this already).

### Installation

1.  **Create a Virtual Environment (Recommended):**

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

    Also install the requirements for specific cloud providers, like AWS:

    ```bash
    cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/aws
    pip3 install -r requirements.txt
    ```

### Running a Simple Benchmark (Example)

```bash
cd PerfKitBenchmarker
./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro
```

*   Replace `--cloud=AWS` with your chosen cloud provider (GCP, Azure, etc.).
*   Adjust `--machine_type` to a valid instance type for your provider.
*   Use `--benchmarks=iperf` for a simple network throughput test.

## Understanding Licensing

PerfKit Benchmarker utilizes various benchmark tools. You are responsible for accepting the licenses of the individual benchmarks before using PKB. You can view a complete list of the licenses for each benchmark within the original repository, and the most common benchmarks are:

*   `aerospike`
*   `bonnie++`
*   `cassandra_ycsb`
*   `cassandra_stress`
*   `cloudsuite3.0`
*   `cluster_boot`
*   `coremark`
*   `copy_throughput`
*   `fio`
*   `gpu_pcie_bandwidth`
*   `hadoop_terasort`
*   `hpcc`
*   `hpcg`
*   `iperf`
*   `memtier_benchmark`
*   `mesh_network`
*   `mongodb`
*   `mongodb_ycsb`
*   `multichase`
*   `netperf`
*   `oldisim`
*   `object_storage_service`
*   `pgbench`
*   `ping`
*   `silo`
*   `scimark2`
*   `speccpu2006`
*   `SHOC`
*   `sysbench_oltp`
*   `TensorFlow`
*   `tomcat`
*   `unixbench`
*   `wrk`
*   `ycsb`

## Preprovisioned Data

Some benchmarks need data to be uploaded to your cloud before the tests can run. See the instructions on preprovisioning data for:

*   Google Cloud
*   AWS

## Advanced Topics

*   **Configuration Overrides:** Customize benchmark behavior with the `--config_override` flag or by using configuration files.
*   **Static Machines:** Run benchmarks on local machines or existing cloud instances.
*   **Running Selective Stages:** Execute specific phases of a benchmark (provision, prepare, run, teardown) for detailed analysis.
*   **Named Sets:**  Run all benchmarks within a predefined set for streamlined testing.
*   **Elasticsearch & InfluxDB Integration:** Publish your results to Elasticsearch or InfluxDB for advanced data analysis and visualization.
*   **Extending PKB:**  Learn how to add new benchmarks, providers, and OS support.

## Useful Flags

Refer to the section on Useful Global Flags to see the various flags used when configuring PerfKit Benchmarker.

---

**Explore the comprehensive [PerfKit Benchmarker wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki) for detailed documentation, tutorials, and more.**