# PerfKit Benchmarker: The Open Source Standard for Cloud Performance Benchmarking

**Need to compare cloud performance across providers?** PerfKit Benchmarker (PKB) is your go-to, open-source solution, providing a standardized and automated way to measure and compare cloud offerings.  Explore the features and get started today with PKB's ability to run on a variety of platforms, and benchmark offerings from cloud providers like GCP, AWS, Azure, and more!  [Check out the original repository](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker) for the latest updates.

**Key Features:**

*   **Automated Benchmarking:** Automatically provisions VMs on your chosen cloud provider, installs benchmarks, and runs workloads without user interaction.
*   **Comprehensive Benchmarking:** Supports a wide array of popular benchmarks, including Aerospike, Bonnie++, FIO, iperf, and many more.
*   **Cross-Platform Compatibility:** Runs benchmarks on a variety of cloud providers (GCP, AWS, Azure, etc.) and even on local machines.
*   **Customizable Configurations:**  Utilizes YAML configurations for flexible and advanced setup, allowing for custom VM specifications, disk configurations, and more.
*   **Preprovisioned Data Support:**  Allows for efficient testing with benchmarks that require pre-existing data in cloud storage.
*   **Integration Testing:** Provides integration tests to ensure functionality and compatibility across cloud providers.
*   **Elasticsearch and InfluxDB Publishing:**  Optionally publishes performance data to Elasticsearch and InfluxDB for visualization and analysis.
*   **Easy Extension:** Designed for extensibility, enabling the addition of new benchmarks, providers, and operating system support.

## Getting Started

### Installation and Setup

1.  **Prerequisites:** You will need to have an account on the cloud provider(s) you want to benchmark.  You also need the software dependencies, which are mostly command line tools and credentials to access your accounts without a password.

2.  **Python 3 Installation:**
    *   Ensure you have Python 3.11 or later installed. Most systems have this installed at `/usr/bin/python3`.

        ```bash
        python3 -m venv $HOME/my_virtualenv
        source $HOME/my_virtualenv/bin/activate
        ```

3.  **Install PerfKit Benchmarker:**
    *   Clone the repository or download a release:

        ```bash
        $ cd $HOME
        $ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
        ```

    *   Install the Python library dependencies:

        ```bash
        $ pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
        ```

    *   Install provider-specific dependencies:
        ```bash
        $ cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/<cloud provider>
        $ pip3 install -r requirements.txt
        ```
### Running a Single Benchmark

Choose your preferred Cloud and run the following with your project and desired machine type.  Example:

```bash
$ ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
```

(Refer to the original README for additional examples.)

### Advanced Configuration

*   **Configuration Files:** Use YAML files for complex setups, including specifying VM groups, cloud providers, and disk configurations. Override default settings using the `--benchmark_config_file` or `--config_override` flags.

*   **Static Machines:** Benchmark local or externally managed machines using static VM configurations, which you specify in a YAML file

*   **Flags in Configuration Files**: Store flags in configuration files and use them as defaults to simplify benchmarking

## Licensing

PerfKit Benchmarker provides wrappers and workload definitions around popular benchmark tools. Due to the automation, you must accept the license of each of the benchmarks individually, and take responsibility for using them before you use the PerfKit Benchmarker. Run PKB with `--accept-licenses`. See the original README for a complete list of licenses.

## Extending PerfKit Benchmarker

Explore the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) file for contributions.  Add new benchmarks, cloud providers, and operating system support by following the instructions in the code comments and the wiki.

## Learn More

*   [Wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki)
*   [FAQ](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/FAQ)
*   [Tutorials](./tutorials/beginner_walkthrough)
*   [Open an Issue](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues)