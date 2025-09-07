# PerfKit Benchmarker: Standardized Cloud Performance Benchmarking

**Measure and compare cloud offerings with PerfKit Benchmarker, the open-source tool designed for consistent and automated benchmarking.  Get started by visiting the original repository: [PerfKitBenchmarker on GitHub](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker)**

## Key Features:

*   **Automated Benchmarking:**  Automates VM instantiation, benchmark installation, and workload execution.
*   **Vendor-Agnostic:** Designed for consistent results across various cloud providers.
*   **Customizable:**  Offers flexibility through configurations and overrides to tailor benchmarks.
*   **Comprehensive Benchmark Suite:** Includes a wide array of benchmarks covering various workloads (see list below).
*   **Integration Testing:** Includes a suite of integration tests to help keep the project in working order.
*   **Flexible Deployment:** Supports multiple cloud platforms including GCP, AWS, Azure, DigitalOcean, and others.
*   **Detailed Reporting & Publishing:**  Supports publishing results to Elasticsearch and InfluxDB.

## Getting Started

To quickly get started running PKB, follow one of our tutorials:

*   [Beginner tutorial](./tutorials/beginner_walkthrough)
*   [Docker tutorial](./tutorials/docker_walkthrough)

For full installation & setup instructions, see below.

## Installation and Setup

### Prerequisites

*   Account(s) on the cloud provider(s) you want to benchmark.
*   Python 3 (at least 3.11) installed.

### Steps:

1.  **Create Virtual Environment (Recommended):**

    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```

2.  **Install PerfKit Benchmarker:**

    ```bash
    $ cd $HOME
    $ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    ```

3.  **Install Python Dependencies:**

    ```bash
    $ pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
    ```

4.  **Provider-Specific Dependencies:** Install additional dependencies if you are using a given cloud provider.  See example for AWS:

    ```bash
    $ cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/aws
    $ pip3 install -r requirements.txt
    ```

## Running a Single Benchmark

Execute benchmarks on various cloud providers.

### Example: Run iperf benchmark on GCP

```bash
$ ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
```

See the original README for example commands for AWS, Azure, IBMCloud, AliCloud, DigitalOcean, OpenStack, Kubernetes, Mesos, CloudStack, Rackspace, and ProfitBricks.

## Available Benchmarks

PerfKit Benchmarker provides a wide array of benchmark tools. **Important: You must accept the licenses of each benchmark before using PerfKitBenchmarker with the `--accept-licenses` flag.**

Benchmarks Include:

*   aerospike
*   bonnie++
*   cassandra\_ycsb
*   cassandra\_stress
*   cloudsuite3.0
*   cluster\_boot
*   coremark
*   copy\_throughput
*   fio
*   gpu\_pcie\_bandwidth
*   hadoop\_terasort
*   hpcc
*   hpcg
*   iperf
*   memtier\_benchmark
*   mesh\_network
*   mongodb (Deprecated)
*   mongodb\_ycsb
*   multichase
*   netperf
*   oldisim
*   object\_storage\_service
*   pgbench
*   ping
*   silo
*   scimark2
*   speccpu2006
*   SHOC
*   sysbench\_oltp
*   TensorFlow
*   tomcat
*   unixbench
*   wrk
*   ycsb
*   openjdk-7-jre
*   spec2006

## Configurations and Configuration Overrides

*   Each benchmark utilizes a configuration written in YAML.
*   Override default settings with `--benchmark_config_file` or `--config_override` flags.
*   See the full configuration file structure in the original README.

## Advanced: Static Machines (e.g., local workstation)

Run benchmarks on non-provisioned machines by configuring static VM details in a YAML file, including IP address, SSH key, and disk specifications.

## Publishing to Elasticsearch & InfluxDB

Optionally publish results.

*   **Elasticsearch:** Install `elasticsearch` Python package and use flags like `--es_uri`, `--es_index`, and `--es_type`.
*   **InfluxDB:** Use `--influx_uri` and `--influx_db_name` to publish results to an InfluxDB server.

## Extending PerfKit Benchmarker

Consult the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) file and the [wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki) for information on contributing to the project.

## Integration Testing

Run integration tests with `tox -e integration`. Ensure all cloud provider SDKs are installed.