# PerfKit Benchmarker: Benchmarking and Comparing Cloud Offerings (Enhance Your Cloud Decisions!)

**PerfKit Benchmarker** is an open-source tool designed to measure and compare the performance of cloud services.  With automated testing and consistent benchmark settings, you can make informed decisions about your cloud infrastructure.  Learn more and contribute at the [original repository](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker).

## Key Features:

*   **Automated Benchmarking:** Simplifies the process of running benchmarks on various cloud providers.
*   **Cross-Platform Compatibility:** Supports a wide range of cloud platforms, including GCP, AWS, Azure, and others.
*   **Consistent Settings:** Provides standardized benchmark configurations for reliable comparison.
*   **Extensive Benchmark Library:** Includes a variety of popular benchmarks to evaluate different aspects of cloud performance.
*   **Customizable Configurations:** Allows users to tailor benchmark settings for specific needs and scenarios.
*   **Reporting and Analysis:** Provides output in a format that's easily reported, and integrates with Elasticsearch and InfluxDB for advanced visualization.

## Getting Started

### Installation & Setup
1.  **Environment Setup**: Ensure you have Python 3.11+ installed. Create and activate a virtual environment:
    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```
2.  **Clone the Repository**:
    ```bash
    $ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    $ cd PerfKitBenchmarker
    ```
3.  **Install Dependencies**:
    ```bash
    $ pip3 install -r requirements.txt
    ```
    *   Install cloud-specific dependencies (e.g. for AWS):
        ```bash
        $ cd perfkitbenchmarker/providers/aws
        $ pip3 install -r requirements.txt
        ```
4.  **Cloud Account**: You'll need accounts on the cloud providers you plan to benchmark.
5.  **Configuration**:  Familiarize yourself with the flags; and the YAML config options
6.  **Accept Licenses**:  Use `--accept-licenses` when running benchmarks.

### Example Runs

Quickly start running benchmarks:

*   **GCP**:
    ```bash
    ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
    ```
*   **AWS**:
    ```bash
    ./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro
    ```
*   **Azure**:
    ```bash
    ./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf
    ```

*(Refer to the original README for more specific examples on running against different clouds and benchmarks)*

### Key Flags and Options

*   `--benchmarks`: Specify benchmarks to run (e.g., `iperf,ping` or `"standard_set"`).
*   `--cloud`:  Choose your cloud provider (GCP, AWS, Azure, etc.).
*   `--machine_type`: Select the instance/VM type.
*   `--zones`: Specify the zone for your resources.
*   `--data_disk_type`: Select the type of disk.
*   `--os_type`: Run tests with Windows (windows), or Juju (juju)
*   `--benchmark_config_file`: Override the settings of the benchmark with a YAML file.

*(Refer to the original README for a complete list of flags.)*

## Benchmarks and Licenses

PerfKit Benchmarker wraps many popular benchmark tools.  Ensure you review and accept the licenses for each individual benchmark before use. Use the `--accept-licenses` flag when running tests.  Benchmarks include: Aerospike, Bonnie++, Cassandra YCSB/Stress, Coremark, FIO, Hadoop Terasort, HPCG, Iperf, and many more.

### Preprovisioned Data

Some benchmarks require data uploaded to a cloud storage bucket prior to running.  See the original documentation for details on preprovisioning data in GCP and AWS.

## Advanced Usage

*   **Configurations**:  Use YAML configuration files or the `--config_override` flag for advanced setups.
*   **Static Machines**: Run benchmarks on existing machines (workstations, etc.) with the `static_vms` configuration.
*   **Elasticsearch Publisher:** Send results to an Elasticsearch server for analysis.
*   **InfluxDB Publisher**: Send results to an InfluxDB server for analysis.
*   **Integration Testing**:  Run integration tests with `tox -e integration`.

## Contributing

We welcome contributions! See the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) file for details on how to get involved.

## Planned Improvements

*(Refer to the original README for a list of planned improvements.)*