# PerfKit Benchmarker: The Open-Source Standard for Cloud Performance Measurement ([View on GitHub](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker))

**PerfKit Benchmarker (PKB) provides a standardized and automated way to measure and compare cloud performance across different providers.** This open-source project offers a comprehensive suite of benchmarks, enabling you to evaluate compute, storage, and network performance in a consistent and reproducible manner.

## Key Features:

*   **Automated Benchmarking:** PKB automates the entire benchmarking process, from provisioning VMs on your chosen cloud provider to running workloads and collecting results, minimizing manual intervention.
*   **Vendor-Agnostic:**  PKB is designed to be cloud-agnostic, supporting major providers like GCP, AWS, Azure, and others, allowing for cross-cloud comparisons.
*   **Extensible:** Easily add new benchmarks, cloud providers, and operating system support to meet evolving performance measurement needs.
*   **Configurable:**  Customize benchmark parameters and configurations using YAML files and command-line overrides to fine-tune your performance testing.
*   **Reproducible Results:**  PKB's standardized approach ensures consistent and reproducible results, enabling reliable performance comparisons.
*   **Comprehensive Benchmark Suite:** Includes a wide array of benchmarks covering various performance aspects, from CPU and memory to storage and network throughput.
*   **Open-Source & Community-Driven:** Benefit from an open-source project with a strong community, providing ongoing improvements, updates, and support.
*   **Licensing Compliance:** PKB includes detailed licensing information for all benchmarks used, ensuring compliance with open-source licenses.
*   **Data Publishing:**  Integrate with Elasticsearch and InfluxDB for powerful result visualization and analysis.
*   **Flexible Deployment:** Run on Cloud Providers (GCP, AWS, Azure, DigitalOcean), and even on any "machine" you can SSH into.

## Getting Started:

### 1. Installation and Setup

Before you begin, ensure you have a cloud provider account. You will need the following:

*   Python 3.11 or higher installed, and a virtual environment.
*   Clone the PerfKit Benchmarker repository:
    ```bash
    $ cd $HOME
    $ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    ```
*   Install the required Python dependencies:
    ```bash
    $ cd $HOME/PerfKitBenchmarker
    $ pip3 install -r requirements.txt
    ```
    Additional provider-specific dependencies can be found in their respective directories (e.g., `perfkitbenchmarker/providers/aws/requirements.txt`).

### 2. Running Your First Benchmark

PKB can run benchmarks on various cloud platforms. Here are a few examples:

*   **GCP:**
    ```bash
    $ ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
    ```
*   **AWS:**
    ```bash
    $ ./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro
    ```
*   **Azure:**
    ```bash
    $ ./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf
    ```
*   **DigitalOcean:**
    ```bash
    $ ./pkb.py --cloud=DigitalOcean --machine_type=16gb --benchmarks=iperf
    ```
*   **Example Run on Kubernetes**
    ```bash
    $ ./pkb.py --vm_platform=Kubernetes --benchmarks=iperf \
               --kubeconfig=/path/to/kubeconfig --use_k8s_vm_node_selectors=False
    ```
    _For the remaining platforms, see the original README._

### 3. Key Flags and Configuration:

*   `--benchmarks`:  Specify a comma-separated list of benchmarks to run (e.g., `--benchmarks=iperf,ping`). Use `--helpmatch=benchmarks` to see the full list.
*   `--cloud`:  Select the cloud provider (e.g., `--cloud=GCP`, `--cloud=AWS`). Defaults to GCP.
*   `--machine_type`: Define the VM instance type (e.g., `--machine_type=n1-standard-8`).
*   `--zones`: Specify the cloud zone (e.g., `--zones=us-east-1a`).
*   `--data_disk_type`: Set the disk type (e.g., `--data_disk_type=pd-ssd`).
*   `--os_type`: Specifies the operating system (e.g. Windows, Juju, etc.)

For a comprehensive guide to available flags and customization options, see the original [README](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker)

### 4. Configurations and Customization

PKB offers flexibility through YAML-based configuration files. You can customize benchmark settings by providing a configuration file using the `--benchmark_config_file` flag or by overriding individual settings with the `--config_override` flag. See the original documentation for a full explanation of configuration options, including the ability to pre-provision data.

### 5. Advanced Usage and Extensions

*   **Running on Static Machines:** Run PKB on your local machines or non-provisioned cloud instances using static VM configurations.
*   **Benchmark Sets:** Run benchmark sets for more comprehensive coverage (e.g., `--benchmarks="standard_set"`).
*   **Extending PKB:**  Contribute to PKB by adding new benchmarks, cloud provider support, and more. See the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) file.
*   **Integration Testing:**  Run integration tests using `tox -e integration` after setting the `PERFKIT_INTEGRATION` variable in your environment.

## License Information

Please see the original README for license details of each benchmark.

## Community and Support

*   Join us on [#PerfKitBenchmarker](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues) on freenode to discuss issues, pull requests, or anything else related to PerfKitBenchmarker
*   Open an [issue](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues)
*   Check out the [wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki) for more detailed information.