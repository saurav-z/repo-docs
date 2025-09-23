# PerfKit Benchmarker: The Open-Source Cloud Benchmark Suite

**PerfKit Benchmarker (PKB) is your go-to solution for consistent, automated benchmarking of cloud offerings.** Measure and compare cloud performance with ease using PKB's extensive suite of open-source benchmarks. Learn more at the [original repository](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker).

## Key Features

*   **Automated Benchmarking:** PKB automates the entire benchmarking process, from VM instantiation to workload execution.
*   **Vendor-Agnostic:** Designed to work across various cloud providers (GCP, AWS, Azure, and more) using vendor-provided command-line tools.
*   **Open-Source & Customizable:**  Leverage an open-source codebase and easily extend it with new benchmarks, cloud providers, and features.
*   **Standardized Benchmarks:**  Focus on consistent results with benchmark default settings designed for cross-service comparisons.
*   **Flexible Configuration:**  Customize benchmark runs using command-line flags and YAML configuration files for advanced control.
*   **Integration Testing:**  Test the integration using cloud resources.
*   **Publishing Results:**  Publish results to Elasticsearch or InfluxDB.

## Getting Started

### Installation

1.  **Prerequisites:** Ensure you have Python 3 (3.12 is recommended) and `pip` installed.
2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    cd PerfKitBenchmarker
    ```
3.  **Install Dependencies:**
    ```bash
    pip3 install -r requirements.txt
    ```
    Also install any provider-specific dependencies, such as for AWS:
    ```bash
    cd perfkitbenchmarker/providers/aws
    pip3 install -r requirements.txt
    ```

### Basic Usage

1.  **Choose Your Cloud:** PKB supports major cloud providers and local machines, allowing you to compare any of your cloud instances.
2.  **Example Runs:**

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
    *   **Local Machines**
        ```bash
        # Configure the static VM, then specify
        ./pkb.py --benchmarks=iperf --machine_type=f1-micro --benchmark_config_file=iperf.yaml --zones=us-central1-f --ip_addresses=EXTERNAL
        ```

    Replace placeholders with your actual cloud account details and desired machine types.  See the full list of providers and corresponding flags below.

###  Essential Flags & Configuration

*   **`--cloud`**: Specifies the cloud provider (GCP, AWS, Azure, etc.).
*   **`--benchmarks`**:  A comma-separated list of benchmarks to run (e.g., `iperf,ping` or `standard_set`). See below for the full list.
*   **`--machine_type`**: Defines the VM instance type.
*   **`--zones`**:  Specifies the cloud region.
*   **`--data_disk_type`**:  Select the disk type.
*   **`--benchmark_config_file`**:  Use this to load a YAML configuration file.
*   **`--config_override`**:  Override individual settings via the command line (e.g., `--config_override=cluster_boot.vm_groups.default.vm_count=100`).

##  Benchmarking Details

###  Benchmark Support

PKB supports a wide range of benchmarks, including:

*   aerospike, bonnie++, cassandra\_ycsb, cassandra\_stress, cloudsuite3.0, cluster\_boot, coremark, copy\_throughput, fio, gpu\_pcie\_bandwidth, hadoop\_terasort, hpcc, hpcg, iperf, memtier\_benchmark, mesh\_network, netperf, oldisim, object\_storage\_service, pgbench, ping, silo, scimark2, speccpu2006, shoc, sysbench\_oltp, tensorflow, tomcat, unixbench, wrk, ycsb

   See the original README for benchmark license details.

   *Please note that benchmarks may require license acceptance using the `--accept-licenses` flag.*

### Cloud Support

PKB supports the following clouds:

| Cloud name   | Default zone  | Notes                                       |
| ------------ | ------------- | ------------------------------------------- |
| GCP          | us-central1-a |                                             |
| AWS          | us-east-1a    |                                             |
| Azure        | eastus2       |                                             |
| IBMCloud     | us-south-1    |                                             |
| AliCloud     | West US       |                                             |
| DigitalOcean | sfo1          | You must use a zone that supports the features 'metadata' and 'private_networking'. |
| OpenStack    | nova          |                                             |
| CloudStack   | QC-1          |                                             |
| Rackspace    | IAD           | OnMetal machine-types are available only in IAD zone                                    |
| Kubernetes   | k8s           |                                             |
| ProfitBricks | AUTO          | Additional zones: ZONE_1, ZONE_2, or ZONE_3 |

### Preprovisioned Data

Some benchmarks require preprovisioned data. See original documentation for instructions.

###  Elasticsearch & InfluxDB Publishing

PKB supports publishing benchmark results to Elasticsearch and InfluxDB.

*   **Elasticsearch:** Install `elasticsearch` Python package. Use the `--es_uri`, `--es_index`, and `--es_type` flags to configure.
*   **InfluxDB:** Use the `--influx_uri` and `--influx_db_name` flags to configure.

## Extending PerfKit Benchmarker

*   **Contribution Guidelines:** Refer to `CONTRIBUTING.md` for information on contributing.
*   **Add New Benchmarks:** Easy to add via `--benchmarks=<new benchmark>`.
*   **Add New Providers:** Easy to add via `--cloud=<new provider>`.
*   **Documentation:** Additional documentation on the [wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki).

## Community

*   Join us on #PerfKitBenchmarker on freenode to discuss issues, pull requests, or anything else related to PerfKitBenchmarker.

## Planned Improvements

We are always looking for improvements. Please submit any requests via GitHub issues.