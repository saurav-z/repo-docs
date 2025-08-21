# PerfKit Benchmarker: Benchmark and Compare Cloud Offerings

**PerfKit Benchmarker (PKB) is a powerful open-source tool designed to measure and compare the performance of various cloud platforms by running a comprehensive suite of benchmarks.** ([Original Repo](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker))

## Key Features

*   **Automated Benchmarking:** Automatically provisions resources on various cloud providers, installs benchmarks, and runs workloads without manual intervention.
*   **Vendor-Neutral:** Designed for consistent benchmarking across different cloud offerings, using standardized settings.
*   **Extensive Benchmark Suite:** Includes a wide range of benchmarks covering CPU, storage, network, and more.
*   **Flexible Configuration:** Allows users to customize benchmark settings, machine types, and cloud providers through YAML configurations and command-line overrides.
*   **Cross-Cloud Support:** Run benchmarks on Google Cloud Platform (GCP), Amazon Web Services (AWS), Microsoft Azure, and other providers.
*   **Preprovisioned Data:** Supports benchmarks that require pre-loaded datasets.
*   **Data Visualization:** Supports publishing of the results to Elasticsearch and InfluxDB servers.
*   **Extensible:** Easy to add new benchmarks, package/OS type support and new providers.

## Getting Started

### Prerequisites

*   Python 3 (3.11 or later)
*   Account(s) on the cloud provider(s) you want to benchmark (see the [providers](perfkitbenchmarker/providers/README.md) for specific instructions and requirements.)
*   Required command-line tools and credentials for cloud account access.

### Installation

1.  **Create a Virtual Environment:**

    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```

2.  **Clone the Repository:**

    ```bash
    $ cd $HOME
    $ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    ```

3.  **Install Dependencies:**

    ```bash
    $ pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
    ```

    You may need to install additional provider-specific dependencies (e.g., AWS).
    ```bash
    $ cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/aws
    $ pip3 install -r requirements.txt
    ```

### Running a Benchmark

Run benchmarks using the `pkb.py` script with specific flags for your cloud provider, machine type, and benchmark:

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

*   **See the original [README](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker) for additional cloud examples.**

### Important Considerations

*   **Licensing:**  You are responsible for accepting the individual licenses of the benchmark tools used.  Use the `--accept-licenses` flag if prompted.
*   **Preprovisioned Data:**  Some benchmarks require pre-existing data.  See the original README for more details on how to upload to the required bucket (GCP and AWS are supported).

### Advanced Usage

*   **Configurations and Configuration Overrides:**  Customize benchmark behavior using YAML configuration files or the `--config_override` flag.
*   **Static Machine Usage:** Run benchmarks on local workstations or other non-provisioned machines (see original README).
*   **Integration Testing:** Run integration tests with the `tox -e integration` command (requires `tox >= 2.0.0`).
*   **Publishing Results:** Results can be published to Elasticsearch or InfluxDB.  See the original README for configuration instructions.

## Documentation and Support

*   [FAQ](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/FAQ)
*   [Tech Talks](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Tech-Talks)
*   [Governing rules](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Governing-Rules)
*   [Community meeting decks and notes](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Community-Meeting-Notes-Decks)
*   [Design documents](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Design-Docs)
*   Report issues or request features: [Open an issue](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues)
*   Join the community on freenode at #PerfKitBenchmarker

## Contributing

Review the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) file for guidelines on contributing to PerfKitBenchmarker.