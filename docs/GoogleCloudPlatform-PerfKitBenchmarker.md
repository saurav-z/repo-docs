# PerfKit Benchmarker: Measure and Compare Cloud Offerings 

**PerfKit Benchmarker is an open-source tool that provides a standardized way to benchmark and compare the performance of various cloud platforms.** (Visit the original repo: [https://github.com/GoogleCloudPlatform/PerfKitBenchmarker](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker))

This README provides a comprehensive guide to using PerfKit Benchmarker, covering installation, running benchmarks, and extending the tool to meet your specific needs.

## Key Features:

*   **Automated Benchmarking:** Instantiates VMs on your chosen cloud provider, automatically installs benchmarks, and runs workloads without user interaction.
*   **Vendor-Agnostic:** Designed to use vendor-provided command-line tools for consistent results across different cloud platforms.
*   **Extensive Benchmark Suite:** Supports a wide array of benchmarks, including CPU, storage, network, and database tests.
*   **Configuration Flexibility:** Allows users to customize benchmark parameters and machine specifications via YAML configuration files.
*   **Multi-Cloud Support:** Runs benchmarks on various providers, including Google Cloud Platform (GCP), Amazon Web Services (AWS), Microsoft Azure, and others.
*   **Data Publishing:** Supports publishing results to Elasticsearch and InfluxDB for easy analysis and visualization.
*   **Extensible:** Easily add new benchmarks, cloud providers, and features to tailor the tool to your needs.

## Getting Started

1.  **Prerequisites:**
    *   Account(s) on the cloud provider(s) you wish to benchmark.
    *   Python 3 (at least version 3.11) and `pip`.
    *   Ensure your chosen cloud provider's CLI tools and dependencies are installed and configured.

2.  **Installation:**
    ```bash
    # Clone the repository
    git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    cd PerfKitBenchmarker

    # Create and activate a virtual environment (recommended)
    python3 -m venv .venv
    source .venv/bin/activate

    # Install dependencies
    pip3 install -r requirements.txt
    ```
    You might need to install additional dependencies based on your cloud provider; see the provider-specific `requirements.txt` files in the `perfkitbenchmarker/providers` directory.

3.  **Running a Benchmark:**
    ```bash
    ./pkb.py --cloud=<cloud provider> --benchmarks=<benchmark_name> --machine_type=<machine_type> <optional flags>
    ```
    Replace `<cloud provider>`, `<benchmark_name>`, and `<machine_type>` with your desired values. Use the ` --helpmatch=pkb` flag for a full list of global flags and `--helpmatch=<benchmark_name>` for flags associated with a specific benchmark.

    Example for running `iperf` on GCP:
    ```bash
    ./pkb.py --cloud=GCP --benchmarks=iperf --machine_type=f1-micro
    ```

## Key Concepts and Usage:

*   **Benchmarks:** Identify a specific workload/test to measure performance. (e.g. `iperf`, `fio`, `sysbench_oltp`)
*   **Cloud Providers:** Specify the target cloud platforms (e.g., GCP, AWS, Azure).
*   **Machine Types:** Defines the VM instance size on the selected cloud.
*   **Flags:** Command-line options for customizing benchmark behavior, cloud configuration, and more.
*   **Configuration Files:** YAML files for defining complex setups, overrides, and custom VM configurations (See the [Configurations and Configuration Overrides](#configurations-and-configuration-overrides) section.)
*   **Static VMs:** You can utilize pre-existing, non-provisioned VMs for benchmarking by using the `--benchmark_config_file`.

## Advanced Usage:

*   **Running Windows Benchmarks:** Run benchmarks using the `--os_type=windows` flag, with supported benchmarks in the `perfkitbenchmarker/windows_benchmarks/` directory.
*   **Integration with Juju:** Benchmarks can be deployed with Juju orchestration by setting the `--os_type=juju` flag. (See the [How to Run Benchmarks with Juju](#how-to-run-benchmarks-with-juju) section.)
*   **Running Selective Stages:** Run only specific stages of the benchmark (provision, prepare, run, teardown) via the `--run_stage` flag.
*   **Preprovisioned Data:** Some benchmarks require preprovisioned data, which needs to be uploaded to the cloud (e.g. Google Cloud Storage or AWS S3) and specified in your command. (See the [Preprovisioned Data](#preprovisioned-data) section.)
*   **Elasticsearch and InfluxDB Publishing:** Send benchmark results to Elasticsearch and InfluxDB for data analysis, via `--es_uri` and `--influx_uri` flag configurations respectively.
*   **Extending PerfKit Benchmarker:** Contribute to the project by adding new benchmarks, cloud providers, or features.

## Licensing
Please see the [original README](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker) for complete details.

## Contributing

Refer to the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) for how to work with PerfKitBenchmarker, and how to submit your pull requests. You can also review [the wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki) for more documentation.

## Further Resources

*   [Wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki)
*   [Issues](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues) -  For questions, reporting issues, or feature requests