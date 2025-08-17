# PerfKit Benchmarker: The Open Source Tool for Cloud Performance Benchmarking

**Tired of guessing which cloud provider offers the best performance?** PerfKit Benchmarker, an open-source project from Google Cloud, provides a standardized and automated way to measure and compare cloud offerings.  See the original repo [here](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker).

## Key Features

*   **Automated Benchmarking:**  Automatically provisions resources, installs benchmarks, and runs workloads on various cloud providers.
*   **Vendor-Agnostic:** Designed to operate across different cloud platforms using their native command-line tools.
*   **Consistent Results:** Employs standardized settings for benchmark execution, promoting comparability across services.
*   **Extensive Benchmark Library:**  Supports a wide range of popular benchmarks, including Aerospike, FIO, iperf, and more (see full list below).
*   **Flexible Configuration:**  Supports various configuration options through YAML files, enabling complex setups and benchmark customization.
*   **Integration with Cloud Providers**: Support for GCP, AWS, Azure, DigitalOcean, OpenStack, Kubernetes, Mesos, CloudStack, Rackspace, ProfitBricks, IBMCloud and AliCloud
*   **Data Publishing:** Integrates with Elasticsearch and InfluxDB for easy result analysis and visualization.

## Getting Started

### Prerequisites

*   **Python 3:** Ensure you have Python 3.11 or later installed.  A virtual environment is recommended:

    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```

*   **Cloud Provider Accounts:**  You'll need accounts with the cloud providers you intend to benchmark.
*   **Dependencies:** Install the required Python packages:

    ```bash
    $ cd $HOME
    $ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    $ pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
    ```
    Additional provider-specific dependencies may also be required (e.g., for AWS, see `perfkitbenchmarker/providers/aws/requirements.txt`).

### Installation and Setup

1.  **Clone the Repository:**

    ```bash
    $ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    ```

2.  **Install Dependencies:**

    ```bash
    $ pip3 install -r PerfKitBenchmarker/requirements.txt
    ```

### Basic Usage

Run a benchmark by specifying your cloud provider, machine type, and the benchmark you want to execute.  Here are some examples:

*   **GCP:**

    ```bash
    ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
    ```

*   **AWS:**

    ```bash
    cd PerfKitBenchmarker
    ./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro
    ```

*   **Azure:**

    ```bash
    ./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf
    ```

For other providers like IBMCloud, AliCloud, DigitalOcean, OpenStack, CloudStack, Rackspace, ProfitBricks, and Kubernetes, see the full [README](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker) for example commands.

## Benchmark Listing & Licensing

PerfKit Benchmarker supports a comprehensive set of benchmarks.  **Before running any benchmark, you must accept the individual licenses associated with each tool.** See the [original readme](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker) for the full list and associated licenses. Benchmarks include:

*   aerospike
*   bonnie++
*   cassandra_ycsb
*   cassandra_stress
*   cloudsuite3.0
*   cluster_boot
*   coremark
*   copy_throughput
*   fio
*   gpu_pcie_bandwidth
*   hadoop_terasort
*   hpcc
*   hpcg
*   iperf
*   memtier_benchmark
*   mesh_network
*   mongodb
*   mongodb_ycsb
*   multichase
*   netperf
*   oldisim
*   object_storage_service
*   pgbench
*   ping
*   silo
*   scimark2
*   speccpu2006
*   SHOC
*   sysbench_oltp
*   TensorFlow
*   tomcat
*   unixbench
*   wrk
*   ycsb

### Important Notes on Licensing:

*   You are responsible for complying with the license terms of each benchmark you use.
*   You must run PKB with the `--accept-licenses` flag.
*   SPEC CPU2006 setup requires a separate license purchase.

## Advanced Usage

*   **Running Named Sets of Benchmarks:**  Use the `--benchmarks` parameter to run pre-defined sets (e.g., `--benchmarks="standard_set"`).
*   **Selective Stage Runs:** Run specific stages of a benchmark (e.g., provision, prepare, run) using the `--run_stage` flag.
*   **Configuration Files:**  Customize benchmark behavior using YAML configuration files. Override settings with the `--benchmark_config_file` and `--config_override` flags.
*   **Static Machines:** Run benchmarks on existing machines (e.g., local workstations) by configuring the `static_vms` setting in your configuration file.
*   **Publishing Results:** Integrate with Elasticsearch or InfluxDB for data visualization by setting the `--es_uri` (Elasticsearch) or `--influx_uri` (InfluxDB) flags.

## Extending PerfKit Benchmarker

*   **Contributions are welcome!** Follow the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) file guidelines.
*   Detailed documentation is available on the [wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki).
*   Open an [issue](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues) to request new features or report issues.