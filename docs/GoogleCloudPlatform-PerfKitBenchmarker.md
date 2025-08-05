# PerfKit Benchmarker: The Open Source Cloud Benchmark Tool

**Measure and compare cloud offerings with ease using PerfKit Benchmarker, an open-source tool designed for consistent and automated benchmarking. [Get started now!](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker)**

PerfKit Benchmarker (PKB) is a comprehensive benchmarking framework designed to simplify the process of evaluating cloud performance. PKB automates the deployment, configuration, and execution of popular benchmarks on various cloud platforms, enabling you to objectively compare cloud offerings.

## Key Features:

*   **Automated Benchmarking:** Instantiates VMs on your chosen cloud provider, automatically installs benchmarks, and runs workloads without user interaction.
*   **Vendor-Neutral:** Designed to operate using vendor-provided command-line tools, ensuring consistent results across services.
*   **Extensive Benchmark Suite:** Supports a wide array of benchmarks, including those for storage, networking, compute, and database performance.
*   **Flexible Configuration:** Offers robust configuration options through YAML files, enabling customization for specific testing needs.
*   **Cloud Provider Support:** Runs benchmarks on major cloud providers, including Google Cloud Platform (GCP), Amazon Web Services (AWS), Microsoft Azure, and more.
*   **Detailed Reporting:** Produces comprehensive reports and supports publishing results to Elasticsearch and InfluxDB for further analysis.
*   **Open Source & Community Driven:** Benefit from a collaborative effort, with opportunities to contribute and shape the future of cloud benchmarking.

## Getting Started

To quickly begin using PerfKit Benchmarker, follow the steps below. Detailed tutorials are available in the `tutorials` directory of the repository (see the original README for links).

1.  **Install Python 3 and Virtual Environment:**
    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    cd PerfKitBenchmarker
    ```

3.  **Install Dependencies:**
    ```bash
    pip3 install -r requirements.txt
    ```
    Install provider-specific dependencies as needed (e.g., `pip3 install -r perfkitbenchmarker/providers/aws/requirements.txt` for AWS).

4.  **Run a Benchmark (Example for iperf on GCP):**
    ```bash
    ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
    ```

**Note:** Ensure you have accepted the licenses of the individual benchmarks before running them. Use the `--accept-licenses` flag as needed.

## Advanced Usage and Configurations

*   **Running Specific Benchmarks:** Use the `--benchmarks` flag to specify a comma-separated list of benchmarks.
*   **Cloud Selection:** Use the `--cloud` flag to specify the target cloud provider (e.g., `--cloud=AWS`).
*   **Machine Type:** Use `--machine_type` to specify the instance type.
*   **Configuration Files:** Utilize YAML configuration files for advanced customization and overrides using `--benchmark_config_file` or `--config_override`.
*   **Preprovisioned Data:** Learn how to prepare data in the cloud for benchmarks that need it.
*   **Elasticsearch/InfluxDB Publishing:** Configure data publishing using the appropriate flags (`--es_uri`, `--influx_uri`).
*   **Windows Benchmarks:** Windows benchmarking is available via the `--os_type=windows` flag.
*   **Static Machine Benchmarking:** Configure PKB to target static machines to benchmark.

For more detailed information, including FAQs, design documents, and community resources, please consult the project's [wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki).

## Extend PerfKit Benchmarker

Contribute to the project by adding new benchmarks, providers, and features. Refer to [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) for guidelines.