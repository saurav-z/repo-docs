# PerfKit Benchmarker: Standardized Cloud Benchmark Suite

**PerfKit Benchmarker (PKB) is an open-source tool for measuring and comparing the performance of various cloud offerings, providing a consistent benchmark framework.** [Visit the Original Repo](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker)

## Key Features

*   **Automated Benchmarking:** PKB automates the process of deploying, configuring, and running benchmarks on various cloud platforms.
*   **Cloud Agnostic:** Supports major cloud providers like GCP, AWS, Azure, and others, allowing for cross-platform comparisons.
*   **Extensible:** Easily add new benchmarks, support for operating systems, and cloud providers.
*   **Customizable:** Provides flexibility through configuration files and command-line flags to tailor benchmarks to specific needs.
*   **Comprehensive Benchmark Suite:** Includes a wide range of benchmarks covering various aspects of cloud performance, including compute, storage, and networking.
*   **Detailed Reporting:** Generates comprehensive results and supports publishing data to Elasticsearch and InfluxDB.

## Getting Started

1.  **Environment Setup:**  Install Python 3 (at least 3.11) and create a virtual environment.
    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```
2.  **Install PerfKit Benchmarker:** Clone the repository and install dependencies.
    ```bash
    git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    cd PerfKitBenchmarker
    pip3 install -r requirements.txt
    ```
3.  **Cloud Provider Setup:** Configure your cloud provider credentials and tools (e.g., AWS CLI, gcloud).
4.  **Run a Benchmark:**  Use the command-line interface to execute benchmarks.
    ```bash
    ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
    ```
    (or adapt the example to your cloud provider and desired tests)

## Key Usage Flags
*   `--cloud`: Specify the cloud provider. (GCP, AWS, Azure, etc.)
*   `--machine_type`: Define the instance type for your tests.
*   `--benchmarks`: Choose specific benchmarks to run (e.g., `iperf`, `ping`, `standard_set`).

## Documentation and Support

*   **Detailed documentation:** The [wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki) contains more in-depth information.
*   **Contribute:** Check out the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) file
*   **Discussions:** Join us on `#PerfKitBenchmarker` on freenode
*   **Issues:** Report issues or request features on [GitHub](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues)

## License

PerfKit Benchmarker utilizes various open-source benchmark tools. The licenses for individual benchmarks are available in the original README.