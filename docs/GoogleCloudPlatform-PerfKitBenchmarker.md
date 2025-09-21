# PerfKit Benchmarker: The Open-Source Cloud Benchmarking Tool

**Tired of vendor lock-in and inconsistent cloud performance measurements?**  PerfKit Benchmarker is the open-source solution for comprehensive and standardized cloud benchmarking, allowing you to objectively compare the performance of various cloud providers and services.  [Get started with PerfKit Benchmarker](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker).

## Key Features

*   **Automated Benchmarking:**  Automates the entire benchmarking process, from provisioning virtual machines (VMs) on various cloud platforms to running benchmarks and collecting results.
*   **Vendor-Neutral:** Designed to provide consistent benchmark results across different cloud providers, offering a fair and objective comparison.
*   **Extensive Benchmark Library:**  Supports a wide range of industry-standard benchmarks, including network, storage, compute, and database workloads.
*   **Flexible Configuration:**  Offers extensive customization options, allowing users to tailor benchmarks to their specific needs and test various configurations.
*   **Result Visualization:**  Provides tools and integration options to visualize and analyze benchmark results, making it easy to identify performance bottlenecks and compare cloud offerings.
*   **Cloud Provider Support:** Supports major cloud providers: GCP, AWS, Azure, IBMCloud, AliCloud, DigitalOcean, OpenStack, CloudStack, Rackspace, ProfitBricks, Kubernetes and Mesos.
*   **Easy to Extend:**  Allows users to add new benchmarks, cloud providers, and features, fostering a collaborative open-source community.

## Getting Started

### 1. Installation & Setup

Before you run PerfKit Benchmarker, make sure you have the following:

*   **Python 3:** (Recommended minimum version: 3.11)
*   **Cloud Account(s):** Accounts with the cloud providers you want to benchmark (e.g., GCP, AWS, Azure).
*   **Python Virtual Environment:** Create and activate a virtual environment to manage dependencies:

```bash
python3 -m venv $HOME/my_virtualenv
source $HOME/my_virtualenv/bin/activate
```

*   **Install PerfKit Benchmarker:** Clone the repository and install dependencies:

```bash
$ cd $HOME
$ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
$ pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
```

*   **Cloud Provider Dependencies:**  Install provider-specific dependencies (e.g., for AWS):

```bash
$ cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/aws
$ pip3 install -r requirements.txt
```

### 2. Running a Benchmark

Here are some quick examples:

*   **GCP (iperf benchmark):**

```bash
./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
```

*   **AWS (iperf benchmark):**

```bash
$ cd PerfKitBenchmarker
$ ./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro
```

*   **Azure (iperf benchmark):**

```bash
./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf
```

**(See original README for more provider-specific examples)**

### 3. Running Benchmarks with Windows and Juju

*   **Windows Benchmarks:** Run with `--os_type=windows`.  See [`perfkitbenchmarker/windows_benchmarks/`](perfkitbenchmarker/windows_benchmarks).
*   **Juju:**  Run benchmarks using Juju orchestration with the `--os_type=juju` flag.

### 4. Running Standard Benchmarks and Named Sets

*   **Run all standard benchmarks:** `--benchmarks="standard_set"`
*   **Run all benchmarks in a named set:** `--benchmarks="<set_name>"` (e.g., `--benchmarks="google_set"`).

### 5. Useful Global Flags

| Flag               | Description                                      |
| ------------------ | ------------------------------------------------ |
| `--helpmatch=pkb`  | Display all global flags.                       |
| `--benchmarks`     | Comma-separated list of benchmarks/sets.        |
| `--cloud`          | Cloud provider (GCP, AWS, Azure, etc.).        |
| `--machine_type`   | VM machine type.                               |
| `--zones`          | Override default zone.                          |
| `--data_disk_type` | Disk type (e.g., pd-ssd, gp3, Premium_LRS).      |

**(See original README for detailed information on all flags)**

## Preprovisioned Data

Some benchmarks require pre-existing data.  You'll need to upload files to cloud storage (Google Cloud Storage, S3, Azure Blob Storage) and then configure PKB to use them via appropriate flags such as `--gcp_preprovisioned_data_bucket`.

## Configurations and Configuration Overrides

*   Use YAML configuration files to customize benchmark behavior, specify VM groups, and configure disk specifications.
*   Override settings via the `--benchmark_config_file` or `--config_override` flags.

## Advanced: Running Benchmarks Without Cloud Provisioning

Run benchmarks on local or static machines by:

1.  SSH access enabled.
2.  User has sudo access.
3.  Create a YAML config file (see example in original README) specifying the static machine's IP, SSH key, and other details.
4.  Run PKB with appropriate flags.

## Using Elasticsearch/InfluxDB Publishers

PerfKit Benchmarker supports publishing results to Elasticsearch and InfluxDB.  Install the necessary Python libraries (`pip install elasticsearch`) and use flags like `--es_uri`, `--influx_uri`, etc. to configure the publishers.

## How to Extend PerfKit Benchmarker

*   See the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) file for details on how to contribute.
*   Extensive comments and documentation are in the code and on the wiki.  Add a page to the wiki, or open an issue if you need more documentation.

## Integration Testing

Run integration tests with `tox -e integration`.  Requires configured cloud provider SDKs.

## License

PerfKit Benchmarker is licensed under [Apache v2](LICENSE).

##  Explore the [Wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki)

*   [FAQ](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/FAQ)
*   [Tech Talks](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Tech-Talks)
*   [Governing rules](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Governing-Rules)
*   [Community meeting decks and notes](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Community-Meeting-Notes-Decks)
*   [Design documents](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Design-Docs)