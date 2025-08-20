# PerfKit Benchmarker: Cloud Performance Benchmarking and Comparison

**Measure, compare, and optimize cloud performance with PerfKit Benchmarker, your open-source solution for consistent and reliable benchmarking.** ([Original Repo](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker))

PerfKit Benchmarker (PKB) is an open-source tool designed to provide a standardized and automated approach to benchmarking cloud offerings. It simplifies the process of measuring and comparing performance across different cloud platforms. PKB leverages vendor-provided command-line tools to ensure consistency and reliability. This documentation provides essential information for getting started and working with the code.

## Key Features

*   **Automated Benchmarking:** PKB automates the entire benchmarking process, from VM instantiation to workload execution, minimizing user interaction.
*   **Vendor-Agnostic:** Supports a wide range of cloud providers (GCP, AWS, Azure, DigitalOcean, OpenStack, etc.) to enable cross-platform comparisons.
*   **Comprehensive Benchmarks:** Includes a diverse suite of benchmarks to evaluate various aspects of cloud performance, including storage, networking, compute, and more.
*   **Configuration Flexibility:** Allows users to customize benchmark settings, machine types, and cloud regions to tailor tests to specific needs.
*   **Result Analysis and Reporting:** Provides tools for collecting, analyzing, and visualizing benchmark results.
*   **Extensible:** PKB is designed to be easily extended. You can add new benchmarks, cloud providers, and configurations.

## Getting Started

### Installation and Setup

1.  **Prerequisites**: Ensure you have a working installation of Python 3.11 or later and a virtual environment.

    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```
2.  **Install PerfKit Benchmarker:** Clone the repository and install the required Python libraries:

    ```bash
    git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    cd PerfKitBenchmarker
    pip3 install -r requirements.txt
    ```

3.  **Cloud Provider Dependencies:** Install any cloud provider specific dependencies. For example, to install the dependencies for AWS:

    ```bash
    cd PerfKitBenchmarker/perfkitbenchmarker/providers/aws
    pip3 install -r requirements.txt
    ```
4.  **Cloud Account Setup:** Configure your cloud account(s) for the provider(s) you intend to benchmark. You will need the necessary credentials and permissions to create and manage resources within those clouds.

5.  **Accept Licenses:**  You must accept the licenses of each of the benchmarks individually before using the PerfKit Benchmarker, using the explicit flag `--accept-licenses`.

### Running a Single Benchmark

PKB can run benchmarks on various cloud providers as well as any machine you can SSH into.

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
*   **IBMCloud:**

    ```bash
    ./pkb.py --cloud=IBMCloud --machine_type=cx2-4x8 --benchmarks=iperf
    ```
*   **AliCloud:**

    ```bash
    ./pkb.py --cloud=AliCloud --machine_type=ecs.s2.large --benchmarks=iperf
    ```
*   **DigitalOcean:**

    ```bash
    ./pkb.py --cloud=DigitalOcean --machine_type=16gb --benchmarks=iperf
    ```
*   **OpenStack:**

    ```bash
    ./pkb.py --cloud=OpenStack --machine_type=m1.medium --openstack_network=private --benchmarks=iperf
    ```
*   **Kubernetes:**

    ```bash
    ./pkb.py --vm_platform=Kubernetes --benchmarks=iperf --kubeconfig=/path/to/kubeconfig --use_k8s_vm_node_selectors=False
    ```
*   **Mesos:**

    ```bash
    ./pkb.py --cloud=Mesos --benchmarks=iperf --marathon_address=localhost:8080
    ```
*   **CloudStack:**

    ```bash
    ./pkb.py --cloud=CloudStack --benchmarks=ping --cs_network_offering=DefaultNetworkOffering
    ```
*   **Rackspace:**

    ```bash
    ./pkb.py --cloud=Rackspace --machine_type=general1-2 --benchmarks=iperf
    ```
*   **ProfitBricks:**

    ```bash
    ./pkb.py --cloud=ProfitBricks --machine_type=Small --benchmarks=iperf
    ```

### Windows Benchmarks

To run Windows benchmarks, install dependencies as before and ensure `smbclient` is installed.  Then use the `--os_type=windows` flag.

### Running Benchmarks with Juju

PKB supports running benchmarks using [Juju](https://jujucharms.com/), a service orchestration tool. Specify the `--os_type=juju` flag.

### Running All Standard Benchmarks

Run with `--benchmarks="standard_set"` to execute every benchmark in the standard set serially.

### Running All Benchmarks in a Named Set

Run with `--benchmarks="<set name>"`.

### Running Selective Stages

Run with the `--run_stage` flag.  Example:  `./pkb.py --benchmarks=cluster_boot --machine_type=n1-standard-2 --zones=us-central1-f --run_stage=provision,prepare,run`

## Useful Global Flags

| Flag               | Notes                                                                                                                                                                                        |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--helpmatch=pkb`  | See all global flags.                                                                                                                                                                        |
| `--helpmatch=hpcc` | See all flags associated with the hpcc benchmark.  Replace `hpcc` with any benchmark name.                                                                                                     |
| `--benchmarks`     | A comma-separated list of benchmarks or benchmark sets to run (e.g., `--benchmarks=iperf,ping`). Run `./pkb.py --helpmatch=benchmarks` for a full list.                                         |
| `--cloud`          | Cloud provider. (e.g., GCP, AWS, Azure). See below table for options.                                                                                                                          |
| `--machine_type`   | VM instance type.  Use provider-specific names (e.g., `n1-standard-8`).  Can also use YAML expressions for custom VM specs (e.g., `--machine_type="{cpus\: 1, memory\: 4.5GiB}"` for a GCE VM). |
| `--zones`          | Override the default zone.  See below table.                                                                                                                                                  |
| `--data_disk_type` | Disk type (e.g., `pd-ssd`, `gp3`, `Premium_LRS`).  Also, use `local` to use the VM's built-in disk.                                                                                              |

*   **Default Zones**

    | Cloud name   | Default zone  |
    | ------------ | ------------- |
    | GCP          | us-central1-a |
    | AWS          | us-east-1a    |
    | Azure        | eastus2       |
    | IBMCloud     | us-south-1    |
    | AliCloud     | West US       |
    | DigitalOcean | sfo1          |
    | OpenStack    | nova          |
    | CloudStack   | QC-1          |
    | Rackspace    | IAD           |
    | Kubernetes   | k8s           |
    | ProfitBricks | AUTO          |

*   **Disk Types**

    | Cloud name | Network-attached SSD | Network-attached HDD |
    | ---------- | -------------------- | -------------------- |
    | GCP        | pd-ssd               | pd-standard          |
    | AWS        | gp3                  | st1                  |
    | Azure      | Premium_LRS          | Standard_LRS         |
    | Rackspace  | cbs-ssd              | cbs-sata             |

### Proxy Configuration for VM Guests
*   Configure proxy settings using the flags: `--http_proxy`, `--https_proxy`, and `--ftp_proxy`

### Preprovisioned Data
*   Upload preprovisioned data to cloud storage (e.g., GCS, S3) and provide bucket with the appropriate flag for each cloud.
    *   `--gcp_preprovisioned_data_bucket=<bucket>` for GCP
    *   `--aws_preprovisioned_data_bucket=<bucket>` for AWS

## Configurations and Configuration Overrides

*   Use YAML config files (`--benchmark_config_file`) to override default settings, including VM group specs and benchmark parameters.
*   Override settings with the `--config_override` flag, using dot notation for nested keys.

## Advanced: Running Benchmarks Without Cloud Provisioning

*   Configure static VMs in a YAML file.
*   Use `--ip_addresses=EXTERNAL` to avoid internal IP addresses.

## Specifying Flags in Configuration Files

*   Specify flags within your configuration files using the `flags` key.

## Using the Elasticsearch Publisher

*   Enable with the `--es_uri` flag.  Install the `elasticsearch` Python package.
*   Use the `--es_index` and `--es_type` flags to customize index and type.

## Using the InfluxDB Publisher

*   Enable with the `--influx_uri` flag.
*   Use the `--influx_db_name` flag to specify the database.

## Extending PerfKit Benchmarker

*   See [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) for information on contributing.
*   Add new benchmarks, cloud providers, and configurations.
*   For further documentation visit the [wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki).

## Integration Testing

*   Run integration tests using `tox -e integration` (requires `tox >= 2.0.0` and defined environment variable `PERFKIT_INTEGRATION`).

## Planned Improvements

*   [Contribute and add new requests via GitHub issues.](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues)