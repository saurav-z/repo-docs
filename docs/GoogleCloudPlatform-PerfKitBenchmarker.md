# PerfKit Benchmarker: The Open-Source Standard for Cloud Performance Measurement

**Easily benchmark and compare cloud offerings with PerfKit Benchmarker, an open-source tool designed for consistent and automated performance testing. [Visit the original repository](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker) for more details.**

## Key Features

*   **Automated Benchmarking:** Automatically provisions VMs on various cloud providers, installs benchmarks, and runs workloads with minimal user interaction.
*   **Vendor-Agnostic:** Utilizes vendor-provided command-line tools for consistent benchmark results across different cloud platforms.
*   **Extensive Benchmark Suite:** Supports a wide range of benchmarks, including Aerospike, Bonnie++, Cassandra, FIO, Hadoop, iperf, and many more, providing comprehensive performance insights.
*   **Flexible Configuration:** Customizable through YAML configuration files, allowing users to define complex setups, including running benchmarks across multiple clouds.
*   **Cloud Provider Support:** Supports major cloud providers: Google Cloud Platform (GCP), AWS, Azure, IBM Cloud, AliCloud, DigitalOcean, OpenStack, CloudStack, Rackspace, Kubernetes and ProfitBricks.
*   **Advanced Testing Options:** Includes support for pre-provisioned data, running selective benchmark stages, and integrating with Elasticsearch and InfluxDB for data publication.
*   **Extensible and Open Source:** Easily extend PerfKit Benchmarker to include new benchmarks, operating system types, and cloud providers.

## Getting Started

### Installation and Setup

1.  **Prerequisites:** You'll need accounts on the cloud providers you want to benchmark and the necessary command-line tools and credentials.
2.  **Python 3 and Virtual Environment:** It's recommended to use a Python 3 virtual environment:

    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```

3.  **Install PerfKit Benchmarker:** Clone or download the latest release from [GitHub](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/releases)

    ```bash
    git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    ```

4.  **Install Dependencies:** Install the required Python libraries.

    ```bash
    pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
    ```
    *   If using AWS:

        ```bash
        cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/aws
        pip3 install -r requirements.txt
        ```

### Running a Single Benchmark

You can run benchmarks on various cloud providers and even local machines. Example runs include:

*   **GCP:** `./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro`
*   **AWS:** `./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro`
*   **Azure:** `./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf`
*   **IBM Cloud:** `./pkb.py --cloud=IBMCloud --machine_type=cx2-4x8 --benchmarks=iperf`
*   **AliCloud:** `./pkb.py --cloud=AliCloud --machine_type=ecs.s2.large --benchmarks=iperf`
*   **DigitalOcean:** `./pkb.py --cloud=DigitalOcean --machine_type=16gb --benchmarks=iperf`
*   **OpenStack:** `./pkb.py --cloud=OpenStack --machine_type=m1.medium --openstack_network=private --benchmarks=iperf`
*   **Kubernetes:** `./pkb.py --vm_platform=Kubernetes --benchmarks=iperf --kubeconfig=/path/to/kubeconfig --use_k8s_vm_node_selectors=False`
*   **Mesos:** `./pkb.py --cloud=Mesos --benchmarks=iperf --marathon_address=localhost:8080`
*   **CloudStack:** `./pkb.py --cloud=CloudStack --benchmarks=ping --cs_network_offering=DefaultNetworkOffering`
*   **Rackspace:** `./pkb.py --cloud=Rackspace --machine_type=general1-2 --benchmarks=iperf`
*   **ProfitBricks:** `./pkb.py --cloud=ProfitBricks --machine_type=Small --benchmarks=iperf`

### Important Flags

| Flag               | Description                                                         |
| ------------------ | ------------------------------------------------------------------- |
| `--benchmarks`     | Comma-separated list of benchmarks or benchmark sets to run.         |
| `--cloud`          | The cloud provider to run the benchmarks on.                        |
| `--machine_type`   | The type of machine to provision.                                   |
| `--zones`          | Allows you to override the default zone.                          |
| `--data_disk_type` | The type of disk to use.                                            |

### Advanced Configuration

*   **Configuration Files:** Use YAML files to override default settings for more complex setups (e.g., `--benchmark_config_file=my_config.yaml`).
*   **Configuration Overrides:** Use `--config_override` for single setting changes (e.g., `--config_override=cluster_boot.vm_groups.default.vm_count=100`).
*   **Static Machines:** Run benchmarks on local workstations or other non-provisioned machines using the `static_vms` configuration.

### Other Features
*  **Running Windows Benchmarks:** By including the `--os_type=windows` flag, users can run the different set of benchmarks that target Windows Server 2012 R2.
*   **Benchmarking With Juju:** By including the `--os_type=juju` flag, users can automatically deploy supported benchmarks in an Juju-modeled service environment.
*   **Running All Benchmarks:** Run all benchmarks in the standard set via the `--benchmarks="standard_set"` parameter. Named set grouping also available.
*   **Selective Stage Running:** Use the `--run_stage` to run selective stages of a benchmark to examine different machine stages.
*   **Proxy configuration:** Configure proxy settings through `pkb.py` flags, including `--http_proxy`, `--https_proxy`, and `--ftp_proxy`.
*   **Preprovisioned Data:** Some benchmarks require preprovisioned data. Users can upload data to their respective clouds and configure the bucket/location using the `--<cloud>_preprovisioned_data_bucket=<bucket>` flags.
*   **Elasticsearch & InfluxDB Publishing:**  Publish performance data to Elasticsearch ( `--es_uri`, `--es_index`, `--es_type` ) and InfluxDB ( `--influx_uri`, `--influx_db_name` ).
### Extending PerfKit Benchmarker

*   Consult the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) file for guidance.
*   Extendable to include new benchmarks, OS types, and cloud providers.
*   Detailed documentation available on the [wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki).

### Integration Testing

Run integration tests with: `tox -e integration`. These tests require cloud provider SDKs to be installed and configured.