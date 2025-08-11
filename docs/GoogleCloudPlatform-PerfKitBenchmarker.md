# PerfKit Benchmarker: Measure and Compare Cloud Offerings

**PerfKit Benchmarker (PKB) is your open-source solution for standardized cloud performance benchmarking, offering a simple, automated way to evaluate and compare different cloud providers.**  [See the original repo](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker).

## Key Features

*   **Automated Benchmarking:**  PKB automates the entire benchmarking process: provisioning VMs, installing benchmarks, and running workloads without user interaction.
*   **Vendor Agnostic:** Operates across multiple cloud providers (GCP, AWS, Azure, DigitalOcean, etc.) for unbiased comparisons.
*   **Standardized Benchmarks:** Utilizes a curated set of benchmarks for consistent and comparable results.
*   **Flexible Configuration:** Offers extensive configuration options, including YAML-based configurations, to tailor benchmarks to specific needs.
*   **Detailed Reporting:** Provides comprehensive reports and supports publishing results to Elasticsearch and InfluxDB for in-depth analysis.
*   **Extensible:** Easily add new benchmarks, cloud providers, and OS types to customize your testing environment.

## Getting Started

### Prerequisites

*   **Python 3:** Install Python 3 (at least 3.11).
*   **Virtual Environment (Recommended):**

```bash
python3 -m venv $HOME/my_virtualenv
source $HOME/my_virtualenv/bin/activate
```

### Installation

1.  **Clone the Repository:**

```bash
$ cd $HOME
$ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
```

2.  **Install Dependencies:**

```bash
$ pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
```

   *   You may need to install additional dependencies for your cloud provider. See their specific `requirements.txt` files within the `perfkitbenchmarker/providers/` directory (e.g., `/perfkitbenchmarker/providers/aws/requirements.txt`).

### Basic Usage

Run a benchmark with the following command examples. Replace the bracketed values with your desired configuration.

#### Run Examples

*   **GCP:**

```bash
$ ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
```

*   **AWS:**

```bash
$ cd PerfKitBenchmarker
$ ./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro
```

*   **Azure:**

```bash
$ ./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf
```

*   **IBMCloud:**

```bash
$ ./pkb.py --cloud=IBMCloud --machine_type=cx2-4x8 --benchmarks=iperf
```

*   **AliCloud:**

```bash
$ ./pkb.py --cloud=AliCloud --machine_type=ecs.s2.large --benchmarks=iperf
```

*   **DigitalOcean:**

```bash
$ ./pkb.py --cloud=DigitalOcean --machine_type=16gb --benchmarks=iperf
```

*   **OpenStack:**

```bash
$ ./pkb.py --cloud=OpenStack --machine_type=m1.medium --openstack_network=private --benchmarks=iperf
```

*   **Kubernetes:**

```bash
$ ./pkb.py --vm_platform=Kubernetes --benchmarks=iperf --kubeconfig=/path/to/kubeconfig --use_k8s_vm_node_selectors=False
```

*   **Mesos:**

```bash
$ ./pkb.py --cloud=Mesos --benchmarks=iperf --marathon_address=localhost:8080
```

*   **CloudStack:**

```bash
./pkb.py --cloud=CloudStack --benchmarks=ping --cs_network_offering=DefaultNetworkOffering
```

*   **Rackspace:**

```bash
$ ./pkb.py --cloud=Rackspace --machine_type=general1-2 --benchmarks=iperf
```

*   **ProfitBricks:**

```bash
$ ./pkb.py --cloud=ProfitBricks --machine_type=Small --benchmarks=iperf
```

### Tutorials
To quickly get started running PKB, follow one of our tutorials:

*   [Beginner tutorial](./tutorials/beginner_walkthrough) for an in-depth but beginner friendly look at PKB's architectures, flags, and even data visualization, using GCP's Cloud Shell & netperf benchmarks.
*   [Docker tutorial](./tutorials/docker_walkthrough) to run PKB in just a few steps, using GCP & docker.
*   Continue reading below for installation & setup on all Clouds + discussion of many topics like flags, configurations, preprovisioned data, & how to make contributions.

## Licensing

PKB leverages existing open source benchmark tools.  Ensure you review and accept the licenses of the individual benchmarks.  You must use the `--accept-licenses` flag when running PKB.  See the original repo for license details and a full list of benchmarks: [https://github.com/GoogleCloudPlatform/PerfKitBenchmarker](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker).

## Preprovisioned Data

Some benchmarks require preprovisioned data. See the documentation in the original repo for more information on how to preprovision data for each cloud provider.

## Configuration and Customization

*   **Configuration Files:** PKB uses YAML-based configuration files for defining benchmark settings.  These files allow for complex setups and overrides of default values.
*   **Configuration Overrides:** Use the `--config_override` flag to modify specific settings directly on the command line.
*   **Static Machines:** Run benchmarks on existing machines by configuring static VM details.

## Advanced Usage

*   **Running Selective Stages:** Run only specific stages of a benchmark (provision, prepare, run, teardown) for troubleshooting and analysis.
*   **Global Flags:** Utilize various global flags for controlling cloud, machine type, zones, and disk type options. See the original README for a comprehensive list.

## Extending PerfKit Benchmarker

*   **CONTRIBUTING.md:**  Refer to the `CONTRIBUTING.md` file for detailed instructions on contributing to the project.
*   **Adding New Features:**  Easily extend PKB by adding new benchmarks, cloud providers, and OS types.  The code is well-commented to facilitate this.
*   **Wiki Documentation:**  Consult the [wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki) for additional documentation.

## Integration Testing

Run unit and integration tests with the `tox` command. Integration tests require the `PERFKIT_INTEGRATION` environment variable to be set.

## Planned Improvements

The project welcomes contributions and improvements. Please submit new requests via GitHub issues.