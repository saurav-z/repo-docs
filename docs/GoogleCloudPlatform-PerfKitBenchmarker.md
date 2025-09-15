# PerfKit Benchmarker: Standardized Cloud Benchmarking and Performance Analysis

**Measure and compare cloud performance with PerfKit Benchmarker, an open-source tool that automates the process of benchmarking cloud offerings.** Explore the power of standardized benchmarks to drive informed decisions about cloud infrastructure. ([Original Repository](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker))

**Key Features:**

*   **Automated Benchmarking:** Easily deploy and run a comprehensive suite of benchmarks on various cloud providers.
*   **Vendor-Neutral:** Designed to provide consistent and reliable benchmark results across different cloud platforms.
*   **Wide Range of Benchmarks:** Includes popular benchmarks for CPU, storage, networking, and database performance.
*   **Customizable:** Offers flexible configuration options to tailor benchmarks to specific needs and workloads.
*   **Reporting and Analysis:** Provides tools to collect, analyze, and visualize benchmark results for easy comparison.
*   **Extensible:** Extend and customize benchmarks through an extensive ecosystem.

## Getting Started

### Installation and Setup

1.  **Python 3 & Virtual Environment**: Ensure Python 3.11+ is installed and create a virtual environment for PKB:

    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```

2.  **Clone the Repository**:

    ```bash
    $ cd $HOME
    $ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    ```

3.  **Install Dependencies**:

    ```bash
    $ pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
    ```
    *Install Provider Specific Dependencies*: For AWS, for instance:
    ```bash
    $ cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/aws
    $ pip3 install -r requirements.txt
    ```

### Running Benchmarks

#### Example Runs:
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

*   **DigitalOcean:**
    ```bash
    ./pkb.py --cloud=DigitalOcean --machine_type=16gb --benchmarks=iperf
    ```

*   **Other Providers**  (IBMCloud, AliCloud, OpenStack, CloudStack, Rackspace, ProfitBricks, Kubernetes, Mesos, and Rackspace)  can also be specified.

### Key Flags

| Flag                | Description                                              |
| ------------------- | -------------------------------------------------------- |
| `--benchmarks`      | Comma-separated list of benchmarks to run.              |
| `--cloud`           | The cloud provider (GCP, AWS, Azure, etc.).           |
| `--machine_type`    | The machine type to provision.                          |
| `--zones`           | Override the default zone.                               |
| `--data_disk_type`  | The type of data disk to use.                            |
| `--helpmatch`       | See all flags related to specific benchmarks             |

### Preprovisioned Data

*   Some benchmarks require preprovisioned data; such as Google Cloud Storage for GCP and AWS S3 for AWS
*   Utilize `gsutil cp` and `aws s3 cp` for data transfers.
*   The `--gcp_preprovisioned_data_bucket` and `--aws_preprovisioned_data_bucket` flags are used.

## Configuration and Configuration Overrides

*   Benchmarks use YAML configuration files.
*   Override settings via `--benchmark_config_file` or `--config_override`.
*   Use static machines with `static_vms` configuration.

## Advanced

*   **Running without Cloud Provisioning**: Configure static machines with IP address and SSH key.
*   **Specifying Flags in Configuration Files**: Use the `flags` key in config files to set defaults.

## Publishers

*   **Elasticsearch Publisher**: Configure with `--es_uri`, `--es_index`, and `--es_type`.
*   **InfluxDB Publisher**: Configure with `--influx_uri` and `--influx_db_name`.

## Extending PerfKit Benchmarker

*   Refer to the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) for contribution guidelines.
*   The [wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki) provides further documentation.
*   Add new benchmarks, package/OS type support, or providers.
*   Open an [issue](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues) to request missing documentation or to contribute.

## Integration Testing

*   Run integration tests with `tox -e integration` after configuring cloud provider SDKs.