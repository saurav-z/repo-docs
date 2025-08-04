# PerfKit Benchmarker: The Open Source Standard for Cloud Performance Testing

**Ready to compare cloud offerings and optimize your infrastructure?** PerfKit Benchmarker is a powerful, open-source tool designed to benchmark and compare cloud services using a standardized set of tests. [Explore the PerfKit Benchmarker Repository](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker).

## Key Features

*   **Automated Benchmarking:** Automates VM instantiation, benchmark installation, and workload execution on various cloud providers.
*   **Vendor Agnostic:** Uses command-line tools provided by cloud vendors, promoting consistent results across platforms.
*   **Configurable:** Offers flexible configuration options, including support for YAML configurations, to tailor benchmarks to specific needs.
*   **Extensible:** Easily add new benchmarks, providers, and operating system support.
*   **Comprehensive:** Provides a wide range of benchmarks, from basic network tests to complex database and application workloads.
*   **Detailed Documentation:** Includes extensive documentation, including a wiki and a beginner-friendly tutorial, to help you get started.
*   **Reporting and Analysis:**  Integrates with Elasticsearch and InfluxDB for data publishing and visualization.

## Getting Started

1.  **Install Python 3 & Virtualenv:**  Ensure you have Python 3 installed and create a virtual environment.
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
3.  **Configure Cloud Access:**  Set up access to your chosen cloud provider (GCP, AWS, Azure, etc.). See provider-specific documentation in the README.
4.  **Run Your First Benchmark:**  Use the provided examples to run benchmarks on your cloud provider.

    ```bash
    ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
    ```

## Licensing and Benchmarks

PerfKit Benchmarker wraps various open-source benchmark tools. Before using, review and accept the individual licenses of the benchmarks you intend to run. A comprehensive list of supported benchmarks and their licenses is included in the README.

## Advanced Usage

*   **Configuration Files:** Use YAML files for advanced benchmark configurations.
*   **Static Machines:** Run benchmarks on existing, non-provisioned machines.
*   **Integration Testing:** Execute integration tests to validate cloud configurations.
*   **Extending PerfKit Benchmarker:** Easily contribute new benchmarks and provider support by following our documented guidelines.

**Contribute and Collaborate:**  Join the PerfKit Benchmarker community by opening issues, contributing code, and participating in discussions.