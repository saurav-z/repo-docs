# PerfKit Benchmarker: Your Cloud Benchmarking Toolkit

**Effortlessly compare cloud offerings with PerfKit Benchmarker, an open-source tool designed for consistent and reliable performance analysis. ([Original Repo](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker))**

## Key Features:

*   **Automated Benchmarking:**  Instantiates VMs, installs benchmarks, and runs workloads with minimal user interaction.
*   **Vendor Agnostic:**  Operates across multiple cloud providers (GCP, AWS, Azure, etc.) using command-line tools for consistent results.
*   **Extensive Benchmark Suite:** Supports a wide range of benchmarks, including `iperf`, `fio`, `cluster_boot`, and many more.
*   **Flexible Configuration:**  Customize benchmark runs with extensive flag options and YAML-based configuration files for advanced scenarios.
*   **Preprovisioned Data Support:**  Facilitates benchmarks requiring pre-loaded data for accurate performance measurement.
*   **Integration Testing:** Comprehensive integration tests ensure the quality and reliability of the benchmarking process.
*   **Elasticsearch and InfluxDB Integration:**  Publish results to Elasticsearch or InfluxDB for easy data analysis and visualization.
*   **Extensible Architecture:**  Easily add new benchmarks, cloud providers, and OS support.

## Getting Started

1.  **Installation:**

    *   **Prerequisites:** Python 3 (at least 3.11) and cloud provider accounts.
    *   Create a virtual environment:

        ```bash
        python3 -m venv $HOME/my_virtualenv
        source $HOME/my_virtualenv/bin/activate
        ```

    *   Clone the repository:

        ```bash
        $ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
        ```
    *   Install dependencies:

        ```bash
        $ pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
        ```

        Install provider-specific dependencies (e.g., for AWS):
        ```bash
        $ cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/aws
        $ pip3 install -r requirements.txt
        ```

2.  **Running a Benchmark:**

    *   **Example:**  Run `iperf` on GCP:

        ```bash
        ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
        ```

    *   See the documentation for examples on other clouds and more in-depth tutorials.

## Important Information

*   **Licensing:**  You are responsible for adhering to the licenses of the individual benchmarks used by PerfKit Benchmarker.
*   **Accepting Licenses:** Run PKB with the `--accept-licenses` flag.
*   **SPEC CPU2006 Setup:** Requires manual download and configuration, see the original README.
*   **Windows Benchmarks:** Run with `--os_type=windows`.
*   **Juju Support:** Deploy and manage services using Juju with the `--os_type=juju` flag.
*   **Preprovisioned Data:**  Follow the instructions for uploading data to the appropriate cloud.
*   **Configuration Overrides:** Use `--benchmark_config_file` or `--config_override` to customize benchmark behavior.
*   **Running Benchmarks Without Provisioning:**  Configure static VMs for local testing.

## Extend and Contribute

*   See the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) for how to contribute.
*   Extend functionality by adding new benchmarks, cloud provider support, and OS types.
*   Detailed documentation is available on the [wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki).

## Community and Support

*   Report issues and request features via [GitHub issues](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues).
*   Discuss with the community on #PerfKitBenchmarker on freenode.