# PerfKit Benchmarker: Cloud Performance Benchmarking Made Easy

**PerfKit Benchmarker (PKB) is your open-source solution for consistently measuring and comparing the performance of various cloud offerings; check out the original repo [here](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker).**

## Key Features

*   **Automated Benchmarking:** PKB automates the entire benchmarking process, from VM provisioning to benchmark execution, minimizing user interaction.
*   **Vendor-Agnostic:** Designed to work across multiple cloud providers, enabling fair comparisons.
*   **Comprehensive Benchmarks:** Supports a wide range of benchmarks covering various aspects of cloud performance, including network, storage, and compute.
*   **Configuration Flexibility:** Offers extensive configuration options and overrides, allowing you to tailor benchmarks to specific needs.
*   **Clear Results:** Provides comprehensive performance data, making it easy to analyze and compare cloud offerings.

## Getting Started

### Installation and Setup

1.  **Prerequisites:**  Ensure you have Python 3.11 or later installed and have accounts with the cloud providers you wish to benchmark.  You'll also need the provider-specific command-line tools.
2.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```
3.  **Install PerfKit Benchmarker:**
    ```bash
    $ cd $HOME
    $ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    ```
4.  **Install Dependencies:**
    ```bash
    $ pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
    ```
    Also install provider specific dependencies as needed:
    ```bash
    $ cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/aws
    $ pip3 install -r requirements.txt
    ```

### Running a Single Benchmark (Examples)

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

    ...and many more for IBMCloud, AliCloud, DigitalOcean, OpenStack, Kubernetes, Mesos, CloudStack, Rackspace, and ProfitBricks.

### Running All Standard Benchmarks
   Run with `--benchmarks="standard_set"` to run all benchmarks in the standard set.

## Advanced Usage

*   **Running Selective Stages:**  Run specific stages (provision, prepare, run, teardown) of a benchmark using the `--run_stage` flag.
*   **Running Windows Benchmarks:**  Run Windows benchmarks using the `--os_type=windows` flag.  Ensure `smbclient` is installed.
*   **Running Benchmarks with Juju:** Integrate with Juju using the `--os_type=juju` flag, streamlining deployment.
*   **Preprovisioned Data:** Learn how to use preprovisioned data with Google Cloud and AWS by using the appropriate flags.
*   **Configuration and Configuration Overrides:** Use YAML configuration files and the `--config_override` flag to customize benchmark behavior.
*   **Running on Local Workstations:** Run benchmarks without cloud provisioning by defining static VMs in your configuration.

## Useful Flags

See the full list with `./pkb.py --helpmatch=pkb`, or specific benchmark flags with `--helpmatch=<benchmark_name>`.

| Flag               | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `--benchmarks`     | Comma-separated list of benchmarks or benchmark sets to run. |
| `--cloud`          | Cloud provider (GCP, AWS, Azure, etc.). Default: GCP.        |
| `--machine_type`   | Machine type to provision.                                   |
| `--zones`          | Override the default zone.                                   |
| `--data_disk_type` | Type of data disk to use.                                    |

## Publishers

### Elasticsearch Publisher
Install the `elasticsearch` package to enable publishing results to an Elasticsearch server.
Use the following flags to configure:
- `--es_uri`
- `--es_index`
- `--es_type`

### InfluxDB Publisher
Publish PerfKit Benchmarker results to an InfluxDB server with the `--influx_uri` and `--influx_db_name` flags.

## Licensing

PerfKitBenchmarker provides wrappers and workload definitions around popular benchmark tools. You must agree to the licenses of each benchmark tool individually before running them. See the original README for a full list.

## Extending PerfKit Benchmarker

Explore the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) file for guidance on contributions. Add new benchmarks, packages, OS types, providers and more by adding comments to the code.

## Integration Testing

Run integration tests with `tox -e integration`. Ensure that the `PERFKIT_INTEGRATION` environment variable is set.

## Planned Improvements

Contribute to the evolution of PKB! Please submit new requests via GitHub issues.