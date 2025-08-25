# PerfKit Benchmarker: The Open-Source Standard for Cloud Performance Measurement

**PerfKit Benchmarker (PKB) is a powerful, open-source framework for benchmarking and comparing cloud offerings using a standardized set of tools; [explore the original repository](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker) to understand the framework in more detail.**

## Key Features:

*   **Automated Benchmarking:** PKB automates the process of provisioning VMs on various cloud providers, installing benchmarks, and running workloads, eliminating the need for manual intervention.
*   **Vendor-Agnostic:** PKB is designed to operate via vendor-provided command-line tools, ensuring consistency across different cloud platforms.
*   **Comprehensive Benchmark Suite:** PKB provides a wide range of benchmarks covering diverse aspects of cloud performance, including compute, storage, and network.
*   **YAML Configuration:** Users can easily customize benchmark runs through YAML configuration files, allowing for complex setups and cloud-spanning tests.
*   **Flexible Deployment:** PKB supports various cloud providers (GCP, AWS, Azure, DigitalOcean, etc.) and can also run on local machines, Kubernetes clusters, and other environments.
*   **Extensible:** The framework is designed for easy extension, allowing users to add new benchmarks, cloud providers, and features.
*   **Result Publishing:** Data can be published to various platforms, including Elasticsearch and InfluxDB, for easy analysis and comparison.
*   **Integration Testing:** Included integration tests that create real cloud resources to assess the accuracy of testing.

## Getting Started:

1.  **Prerequisites:**  Ensure you have accounts on your target cloud providers and the necessary software dependencies.
2.  **Install Python 3 and Create Virtual Environment:**

    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```
3.  **Install PerfKit Benchmarker:**

    ```bash
    $ cd $HOME
    $ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    $ cd PerfKitBenchmarker
    $ pip3 install -r requirements.txt
    ```

4.  **Run a benchmark**: Select your cloud provider and machine type, then run a benchmark. Some examples include:

    ```bash
    # Example run on GCP
    ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro

    # Example run on AWS
    ./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro
    ```
    Refer to the original repository for more detailed examples.
5.  **Explore and Customize:**  Review the available flags and benchmark configurations in the original README to tailor your tests.

## Benchmarks & Licensing
PerfKit Benchmarker provides wrappers and workload definitions around popular
benchmark tools. Please review the licenses of the benchmarks used before using. You can review the full list in the original README.

## Advanced Usage:

*   **Running Specific Stages:** Control the execution stages of a benchmark (provision, prepare, run, teardown) to troubleshoot or analyze specific steps.
*   **Configurations and Configuration Overrides:** Customize benchmarks through YAML config files or command-line overrides.
*   **Running on Local Machines:** Run benchmarks without cloud provisioning by specifying static VM details in a YAML config.
*   **Integrating with Elasticsearch/InfluxDB:** Publish results to Elasticsearch and InfluxDB for data visualization and analysis.
*   **Running Windows Benchmarks:** Run Windows benchmarks using `--os_type=windows`.
*   **Running Benchmarks with Juju:** Run benchmarks with the `--os_type=juju` flag.

## Extending PerfKit Benchmarker:

Contribute to PKB by adding new benchmarks, cloud providers, or features. Start by reviewing the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) file and exploring the code.

## Resources:

*   [Wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki) for detailed information.
*   [Open an issue](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues) to ask questions or report issues.
*   Join the #PerfKitBenchmarker channel on freenode.
*   See the original [README](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker) for further information.