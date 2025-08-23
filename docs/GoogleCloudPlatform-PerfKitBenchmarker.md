# PerfKit Benchmarker: The Open-Source Standard for Cloud Performance Measurement

**Measure, compare, and optimize your cloud performance with PerfKit Benchmarker, the open-source tool that simplifies cloud benchmarking.** ([Original Repo](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker))

PerfKit Benchmarker (PKB) is an open-source project designed to provide a consistent and reliable way to benchmark cloud offerings. It leverages vendor-provided command-line tools to ensure accurate and comparable results across different cloud platforms.  PKB automates the process of deploying, configuring, and running popular benchmarks, enabling you to make informed decisions about your cloud infrastructure.

## Key Features:

*   **Automated Benchmarking:**  Easily run complex benchmarks with minimal user interaction.
*   **Cross-Platform Compatibility:** Supports benchmarking on major cloud providers like GCP, AWS, Azure, DigitalOcean, and more, as well as on-premise environments.
*   **Comprehensive Benchmark Suite:** Includes a wide range of benchmarks covering various aspects of cloud performance, including compute, storage, and network.
*   **Configuration Flexibility:**  Customize benchmark settings and VM configurations to suit your specific needs.
*   **Detailed Reporting:**  Generate comprehensive reports and integrate with Elasticsearch and InfluxDB for data analysis and visualization.
*   **Open Source & Community Driven:** Benefit from a collaborative ecosystem and contribute to improving cloud benchmarking practices.

## Getting Started:

To get started with PerfKit Benchmarker, you'll need to:

1.  **Set up your environment:** Ensure you have Python 3 (at least 3.11) and a virtual environment installed.
2.  **Install PerfKit Benchmarker:** Clone the repository and install dependencies using `pip3 install -r requirements.txt`.  You may need additional dependencies for specific cloud providers (e.g. AWS).
3.  **Configure Cloud Credentials:** Set up your cloud provider accounts and credentials as per the provider-specific instructions.
4.  **Run a Benchmark:** Execute benchmarks using the command-line interface.  See the examples below.

### Installation & Setup

**1. Python 3 and Virtual Environment**
```bash
python3 -m venv $HOME/my_virtualenv
source $HOME/my_virtualenv/bin/activate
```

**2. Clone Repository and Install Dependencies**
```bash
$ cd $HOME
$ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
$ pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
```

#### Provider Specific Requirements (Example)

```bash
$ cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/aws
$ pip3 install -r requirements.txt
```

### Example Runs:

**GCP:**
```bash
$ ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
```

**AWS:**
```bash
$ ./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro
```

**Azure:**
```bash
$ ./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf
```
(Similar examples are available for IBMCloud, AliCloud, DigitalOcean, OpenStack, Kubernetes, Mesos, CloudStack, Rackspace, and ProfitBricks.)

### Helpful Flags

| Flag               | Notes                                                 |
| ------------------ | ----------------------------------------------------- |
| `--helpmatch=pkb`  | See all global flags                                  |
| `--benchmarks`     | A comma separated list of benchmarks to run          |
| `--cloud`          | Cloud where the benchmarks are run. See the table below  |
| `--machine_type`   | Type of machine to provision                        |
| `--zones`          | Overrides the default zone. See the table below.  |
| `--data_disk_type` | Type of disk to use. Names are provider-specific. |

## Advanced Usage:

*   **Running Specific Benchmarks:**  Use the `--benchmarks` flag to specify a comma-separated list of benchmarks (e.g., `--benchmarks=iperf,ping`).
*   **Running Standard Sets:**  Use `--benchmarks="standard_set"` to run the default set of benchmarks.
*   **Using Named Sets:** Run all benchmarks within a named set with `--benchmarks="set_name"`.
*   **Running Selective Stages:** Use the `--run_stage` flag to run only the provisioning, preparation, or the benchmark itself.
*   **Configurations and Configuration Overrides:**  Customize benchmark settings and VM configurations using YAML configuration files or the `--config_override` flag.
*   **Running on Static Machines:** Run PKB on existing VMs by configuring the `static_vms` parameter.
*   **Integration Tests:** Run with the `tox -e integration` command to run the integration tests.

### Licensing and Benchmarks Included:
PerfKit Benchmarker includes wrappers and workload definitions around popular benchmark tools.  You are responsible for the individual licenses of each tool.  See the original README for a comprehensive list of benchmarks and their associated licenses.  You must accept the license of each of the benchmarks individually, and take responsibility for using them before you use the PerfKit Benchmarker by running PKB with the explicit flag `--accept-licenses`.

### Preprovisioned Data
Preprovisioned data allows for the use of data uploads with the PKB tool. Follow the instructions below for uploading with a bucket.

#### Google Cloud
```bash
gsutil cp <filename> gs://<bucket>/<benchmark-name>/<filename>
```
Run with the flag: `--gcp_preprovisioned_data_bucket=<bucket>`.

#### AWS
```bash
aws s3 cp <filename> s3://<bucket>/<benchmark-name>/<filename>
```
Run with the flag: `--aws_preprovisioned_data_bucket=<bucket>`.

## Extending PerfKit Benchmarker:

Explore the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) file and documentation on the wiki for detailed information on adding new benchmarks, OS types, providers, and more.

## Publishing Results

PKB offers optional integration with:

*   **Elasticsearch:** Install the `elasticsearch` Python package and use the `--es_uri`, `--es_index`, and `--es_type` flags.
*   **InfluxDB:** Use the `--influx_uri` and `--influx_db_name` flags (no extra packages needed).
```