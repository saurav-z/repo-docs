# PerfKit Benchmarker: Cloud Performance Benchmarking Made Easy

**Measure and compare cloud offerings with PerfKit Benchmarker, an open-source tool designed for consistent and automated performance testing. Get started today at [https://github.com/GoogleCloudPlatform/PerfKitBenchmarker](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker)!**

PerfKit Benchmarker (PKB) is a robust, open-source project designed to help you benchmark and compare the performance of various cloud providers. It simplifies the process of evaluating cloud offerings by automating the deployment, configuration, and execution of a wide range of popular benchmarks. This README provides the essential information to get you started, along with links to more in-depth documentation.

## Key Features

*   **Automated Benchmarking:** PKB automates the entire benchmarking process, from provisioning VMs on your chosen cloud provider to installing and running workloads.
*   **Vendor-Neutral Approach:** PKB utilizes command-line tools provided by cloud vendors, ensuring consistency and facilitating cross-provider comparisons.
*   **Extensive Benchmark Suite:** PKB supports a comprehensive set of benchmarks, covering various aspects of cloud performance, including compute, storage, and networking.
*   **Flexible Configuration:** Customize your benchmarking runs with flexible configurations, including support for YAML-based configuration files and command-line overrides.
*   **Integration with Various Clouds:**  PKB supports a wide array of cloud providers including Google Cloud Platform (GCP), Amazon Web Services (AWS), Azure, DigitalOcean, and more.
*   **Elasticsearch and InfluxDB Publishing:** Easily publish and analyze benchmark results using Elasticsearch and InfluxDB for comprehensive reporting and visualization.
*   **Extensibility:** PKB is designed for easy extensibility, allowing you to add new benchmarks, cloud providers, and features.

## Quickstart

*   **Installation and Setup:**

    1.  **Prerequisites:** Ensure you have Python 3 (version 3.11 or higher) installed and a virtual environment set up.
    2.  **Clone the Repository:** `git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git`
    3.  **Install Dependencies:** Navigate to your PerfKitBenchmarker directory and run `pip3 install -r requirements.txt`.  Also, install provider-specific requirements (e.g. for AWS `cd perfkitbenchmarker/providers/aws && pip3 install -r requirements.txt`).
    4.  **Cloud Provider Accounts:** You'll need accounts on the cloud provider(s) you want to benchmark.  Configure your credentials appropriately.
*   **Example Run:**

    *   To run `iperf` on Google Cloud with an f1-micro instance:
        ```bash
        ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
        ```
    *   For AWS:
         ```bash
        ./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro
        ```
        (Replace `t2.micro` with the appropriate AWS machine type for your use case)

    *   For Azure:
         ```bash
        ./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf
        ```

    *   See the rest of the README for options on other providers like IBMCloud, AliCloud, DigitalOcean, OpenStack, Kubernetes, Mesos, CloudStack, Rackspace, ProfitBricks.

## Benchmarks and Licensing

PerfKit Benchmarker includes a variety of benchmarks.  Due to licensing, users must accept the license of each benchmark before use. A full list of included benchmarks and their licenses can be found in the original [README](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker#licensing) in the original repo.

## Configuration and Advanced Usage

*   **Running Selective Stages:** Provision, prepare, and run stages of `cluster_boot`.

    ```
    ./pkb.py --benchmarks=cluster_boot --machine_type=n1-standard-2 --zones=us-central1-f --run_stage=provision,prepare,run
    ```

*   **Configuration Files:** Use YAML config files to customize benchmark parameters, including machine types, zones, and disk configurations. Example:
    ```yaml
    iperf:
        flags:
            machine_type: n1-standard-2
            zone: us-central1-b
            iperf_sending_thread_count: 2
    ```

*   **Command-Line Overrides:**  Use `--config_override` to change configuration settings directly from the command line.  Example: `--config_override=cluster_boot.vm_groups.default.vm_count=100`

*   **Static Machines:** Run benchmarks on pre-existing machines.  Configure the `static_vms` section in your YAML config.

## Learn More

*   [Wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki) - Comprehensive documentation.
*   [FAQ](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/FAQ) - Frequently asked questions.
*   [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) -  Learn how to contribute.
*   [Issues](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues) - Report issues or suggest improvements.