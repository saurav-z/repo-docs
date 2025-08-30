# PerfKit Benchmarker: The Open Source Standard for Cloud Benchmarking

**Tired of vendor lock-in and inconsistent performance claims?** PerfKit Benchmarker (PKB) provides a comprehensive, open-source solution for measuring and comparing cloud offerings, ensuring you make informed decisions based on reliable data.  [Explore the Original Repo](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker)

## Key Features:

*   **Vendor-Agnostic Benchmarking:**  PKB uses a standardized set of benchmarks to objectively evaluate cloud performance across different providers (GCP, AWS, Azure, DigitalOcean, and more).
*   **Automated & Reproducible:** PKB automates the setup, execution, and collection of results, ensuring consistent and reproducible benchmarking runs.
*   **Comprehensive Benchmark Suite:**  Run a diverse array of benchmarks, including I/O, network, CPU, and database performance tests.
*   **Flexible Configuration:** Customize benchmark settings and easily override default configurations to match your specific needs using YAML-based config files or command-line flags.
*   **Support for Preprovisioned Data:**  Easily manage and incorporate preprovisioned data to accurately simulate real-world workloads.
*   **Extensible Architecture:**  Extend PKB by adding new benchmarks, cloud providers, and features to adapt to evolving cloud technologies.
*   **Integrated Reporting and Analysis:** Optionally publish data to Elasticsearch or InfluxDB for detailed analysis and visualization of results.
*   **Integration Testing:** Includes integration tests to validate cloud resource creation (requires cloud provider SDKs).

## Getting Started:

### Installation and Setup

1.  **Prerequisites:** Ensure you have Python 3.11+ and Git installed. Create and activate a virtual environment:

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
    $ cd $HOME/PerfKitBenchmarker
    $ pip3 install -r requirements.txt
    ```
    Install provider-specific dependencies (e.g., for AWS) within the corresponding provider directory:

    ```bash
    $ cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/aws
    $ pip3 install -r requirements.txt
    ```

### Example Usage:
Run the `iperf` benchmark on a specific cloud provider:
```bash
./pkb.py --cloud=GCP --benchmarks=iperf --machine_type=f1-micro
```
See the full README for provider-specific setup and usage.

## Benchmarks and Licensing:

PKB provides wrappers for popular benchmark tools. Users are responsible for reviewing and accepting the licenses of each benchmark before use.  A comprehensive list of included benchmarks and their licenses can be found in the full README.

## Key Flags:

*   `--helpmatch=pkb`:  Display all global flags.
*   `--helpmatch=<benchmark_name>`:  Show flags specific to a benchmark.
*   `--benchmarks`:  Specify benchmarks to run (e.g., `--benchmarks=iperf,ping`).
*   `--cloud`:  Choose the cloud provider (e.g., `--cloud=AWS`).
*   `--machine_type`: Select the VM type (provider-specific).
*   `--zones`:  Override the default zone.
*   `--data_disk_type`:  Choose the data disk type (provider-specific).

## Advanced Features:

*   **Running Selective Stages:** Control the execution of benchmark phases (provision, prepare, run, teardown).
*   **Running on Local Machines:**  Run benchmarks on static (non-provisioned) machines.
*   **Configuration Files:** Override default configurations using YAML files (`--benchmark_config_file`) or the `--config_override` flag.
*   **Preprovisioned Data:** Upload data to cloud storage for benchmarks requiring it.
*   **Integration with Juju:** Deploy Juju-modeled services for supported benchmarks (`--os_type=juju`).
*   **Publishing Results:** Integrate with Elasticsearch (`--es_uri`) or InfluxDB (`--influx_uri`) for data visualization and analysis.

## Contributing:

Contributions are welcome! Refer to the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) file for guidance.  Join the community on #PerfKitBenchmarker on freenode or open an issue.

## Planned Improvements:

See GitHub issues for planned features and enhancements.