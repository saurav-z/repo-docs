# PerfKit Benchmarker: The Open-Source Standard for Cloud Performance Benchmarking

**Quickly and easily measure and compare cloud performance with PerfKit Benchmarker, a flexible and automated benchmarking tool.  Check out the original repo [here](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker)!**

## Key Features

*   **Automated Cloud Resource Provisioning:** Automatically sets up VMs on various cloud providers (GCP, AWS, Azure, etc.).
*   **Comprehensive Benchmark Suite:** Supports a wide range of benchmarks covering compute, storage, network, and database performance.
*   **Flexible Configuration:**  Customize benchmarks and cloud configurations using YAML files and command-line overrides.
*   **Multi-Cloud Support:**  Compare performance across different cloud providers with a consistent set of benchmarks.
*   **Detailed Reporting & Publishing:**  Generate comprehensive reports and publish results to Elasticsearch and InfluxDB.
*   **Extensible Architecture:** Easily add new benchmarks, cloud providers, and operating system support.

## Introduction

PerfKit Benchmarker (PKB) is an open-source project designed to provide a standardized approach to measuring and comparing the performance of cloud offerings.  It simplifies the benchmarking process by automating the provisioning of cloud resources, installing and running benchmarks, and collecting results. PKB uses vendor-provided command-line tools to execute benchmarks, ensuring consistency across different cloud platforms. Benchmarks are designed with recommended default settings for consistency, but are also easily configurable. 

## Getting Started

### Installation and Setup

1.  **Prerequisites:**
    *   Python 3.11 or later
    *   Access to cloud provider accounts (GCP, AWS, Azure, etc.)
    *   Necessary command-line tools and credentials for cloud access.

2.  **Set up a Virtual Environment:**
    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```

3.  **Clone the Repository:**
    ```bash
    cd $HOME
    git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    ```

4.  **Install Dependencies:**
    ```bash
    pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
    ```
    *   Install additional provider specific requirements as necessary from the `perfkitbenchmarker/providers/<provider>/requirements.txt` file for your chosen cloud.

5.  **Example Run (GCP):**
    ```bash
    ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
    ```

    *   Refer to the "Example Runs" section for more specific examples for different cloud providers.

### Running Benchmarks

*   **Use `--benchmarks`:**  Specify a comma-separated list of benchmarks or benchmark sets to execute.
*   **Use `--cloud`:**  Select the cloud provider to use (GCP, AWS, Azure, etc.).
*   **Use `--machine_type`:** Define the instance type to use for your benchmarks.

## Advanced Topics

*   **Licensing:** PerfKit Benchmarker uses wrappers and workload definitions for popular benchmark tools, each with its own license. You must accept the licenses of the individual benchmarks before use. See the full list of licenses in the original README.
*   **Configurations and Configuration Overrides:** Customize benchmark settings via YAML config files or the `--config_override` flag.
*   **Running Selective Stages:** Provision, prepare, and run benchmarks in stages for debugging.
*   **Preprovisioned Data:**  Upload and configure preprovisioned data for benchmarks that require it (e.g., databases).
*   **Running Without Cloud Provisioning:** Run benchmarks on local machines or other static environments.
*   **Integration with Juju:** Deploy benchmarks using the Juju orchestration tool with the `--os_type=juju` flag.
*   **Publishing Results:** Publish benchmark results to Elasticsearch or InfluxDB using the appropriate flags.
*   **Extending PerfKit Benchmarker:** Learn how to contribute and add new benchmarks, cloud providers, and features.
*   **Flag Summary:**  Explore the many command line flags for detailed control.

## Useful Resources
*   [Wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki)
*   [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md)

## Contributing

We welcome contributions!  Please see the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) file for guidelines on how to contribute.

## Integration Testing

Run integration tests to test your changes. Ensure you have `tox >= 2.0.0` installed and define the `PERFKIT_INTEGRATION` environment variable.

```bash
$ tox -e integration
```

## Planned Improvements

We're always looking to improve PerfKit Benchmarker. Please submit feature requests and suggestions via GitHub issues.