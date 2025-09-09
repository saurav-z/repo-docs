# PerfKit Benchmarker: The Open-Source Standard for Cloud Performance Benchmarking

**Measure, compare, and optimize your cloud infrastructure with PerfKit Benchmarker, the open-source tool from Google for comprehensive performance analysis. [Explore the repository on GitHub](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker).**

## Key Features

*   **Automated Benchmarking:** Easily set up, run, and automate benchmark workloads on various cloud providers.
*   **Cross-Cloud Comparisons:** Compare performance across GCP, AWS, Azure, and more, using a consistent methodology.
*   **Extensive Benchmark Suite:** Access a wide range of industry-standard benchmarks, including iperf, fio, Hadoop Terasort, and many more.
*   **Flexible Configuration:** Customize benchmark settings and machine configurations to meet your specific needs.
*   **Detailed Results:** Generate comprehensive performance reports and easily integrate with data visualization tools like Elasticsearch and InfluxDB.
*   **Open-Source & Community Driven:** Contribute to and benefit from a community-driven project with ongoing improvements and support.

## Getting Started

### 1. Prerequisites

*   **Python 3:** (3.11 or higher recommended) Ensure you have Python 3 installed.
*   **Cloud Provider Accounts:** Access to the cloud providers you wish to benchmark (GCP, AWS, Azure, etc.).
*   **Dependencies:** Install the necessary Python libraries and cloud provider-specific tools.

### 2. Installation

1.  **Create a virtual environment:**

    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```

2.  **Clone the repository:**

    ```bash
    $ cd $HOME
    $ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    ```

3.  **Install Python dependencies:**

    ```bash
    $ cd $HOME/PerfKitBenchmarker
    $ pip3 install -r requirements.txt
    ```

4.  **Install Cloud Provider-Specific Dependencies:** Refer to the [AWS](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/perfkitbenchmarker/providers/aws/requirements.txt) and other cloud provider requirements.txt files as needed.

### 3. Running Benchmarks

1.  **Basic Run Example (GCP):**

    ```bash
    $ ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
    ```

2.  **Basic Run Example (AWS):**

    ```bash
    $ ./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro
    ```

3.  **Additional Examples:**  See the original README for examples of running on Azure, IBMCloud, AliCloud, DigitalOcean, OpenStack, Kubernetes, Mesos, CloudStack, Rackspace, and ProfitBricks.

## Key Concepts & Features

*   **Licensing:** Review and accept licenses for the benchmarks you intend to run.  Use the `--accept-licenses` flag.
*   **Benchmarks:** Run a single benchmark or a named set (e.g., `--benchmarks=standard_set`).
*   **Flags:** Utilize global flags for configuration; see the README for a list of common flags.
*   **Configuration Files:** Override default settings with YAML configuration files using the `--benchmark_config_file` or `--config_override` flag.
*   **Preprovisioned Data:** Learn how to configure and upload data for benchmarks that require pre-provisioned resources.

## Advanced Usage

*   **Running Selective Stages:** Use `--run_stage` to provision, prepare, or teardown.
*   **Static Machines:** Run benchmarks on local or pre-existing machines.
*   **Windows Benchmarks:** Run Windows benchmarks by using `--os_type=windows`.
*   **Juju Integration:** Deploy and benchmark services using Juju with `--os_type=juju`.
*   **Elasticsearch & InfluxDB Integration:** Publish results to Elasticsearch or InfluxDB for analysis.  See the README for setup instructions and relevant flags.

## Extending PerfKit Benchmarker

*   **Contributions:**  See the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) file.
*   **Documentation:**  Refer to the [wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki) for detailed documentation.

## Integration Testing

*   Run integration tests with `tox -e integration`.

**Visit the [GitHub repository](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker) to learn more, contribute, and stay updated on the latest features and improvements.**