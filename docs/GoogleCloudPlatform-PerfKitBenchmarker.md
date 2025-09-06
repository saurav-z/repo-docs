# PerfKit Benchmarker: The Definitive Cloud Benchmark Tool

**PerfKit Benchmarker (PKB) is an open-source tool that helps you rigorously measure and compare the performance of cloud offerings.** Get started with PKB and see how different cloud providers stack up! [Explore the original repository](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker) for more information.

## Key Features:

*   **Automated Benchmarking:** PKB automates the setup, execution, and teardown of benchmarks on various cloud platforms.
*   **Vendor-Neutral:** Designed to provide consistent results across different cloud providers.
*   **Extensive Benchmark Suite:** Includes a wide range of popular benchmarks for CPU, storage, networking, and more.
*   **Flexible Configuration:** Allows customization through YAML configuration files and command-line overrides.
*   **Cross-Cloud Comparison:** Supports running benchmarks on multiple cloud providers (GCP, AWS, Azure, DigitalOcean, IBMCloud, AliCloud, OpenStack, CloudStack, Rackspace, ProfitBricks, Kubernetes, and Mesos).
*   **Results Analysis & Reporting:** Data can be published to Elasticsearch and InfluxDB for detailed analysis.
*   **Community Driven:** Benefit from a collaborative environment, with active issue tracking and open contributions.

## Getting Started:

### Prerequisites:

*   A Python 3 environment (at least Python 3.11 is recommended).
*   Access to the cloud providers you intend to benchmark (GCP, AWS, Azure, etc.). You'll need command-line tools and credentials for each provider.

### Installation:

1.  **Set up a virtual environment:**

    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```

2.  **Clone the repository:**

    ```bash
    cd $HOME
    git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    ```

3.  **Install Python dependencies:**

    ```bash
    pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
    ```

    Install provider-specific requirements (e.g., for AWS):

    ```bash
    cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/aws
    pip3 install -r requirements.txt
    ```

### Running a Benchmark:

1.  **Basic Example (GCP):**

    ```bash
    ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
    ```

2.  **Other Cloud Providers:**  Follow the examples provided in the original README for other providers (AWS, Azure, etc.).

###  Key Flags and Configuration:

*   `--benchmarks`:  Specify the benchmark(s) to run (e.g., `iperf`, `ping`, `standard_set`).
*   `--cloud`:  Select the cloud provider (e.g., `GCP`, `AWS`, `Azure`).
*   `--machine_type`:  Define the virtual machine type.
*   `--zones`: Override the default zone.
*   `--data_disk_type`:  Specify the disk type.
*   `--benchmark_config_file`:  Provide a YAML configuration file for advanced customization.
*   `--config_override`:  Override specific configuration settings from the command line.

### Advanced Features:

*   **Running Specific Stages:**  Use `--run_stage` to run individual stages of a benchmark (provision, prepare, run, teardown).
*   **Static Machines:**  Run benchmarks on existing, non-provisioned machines (see the original README for details).
*   **Configuration Files:** Create YAML files for advanced customization of benchmarks.
*   **Integration Testing:** Run integration tests with the command `tox -e integration` (requires environment setup).
*   **Elasticsearch and InfluxDB Integration:** Publish results for analysis (see original README for flags).
*   **Extending PKB:** Extend PKB by adding new benchmarks, package/OS type support, and cloud providers.

**[See the original README](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker) for comprehensive details, including information on licensing, pre-provisioned data, Juju integration, and contributions.**