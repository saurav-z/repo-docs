# PerfKit Benchmarker: Cloud Benchmarking for Consistent Performance Analysis

**PerfKit Benchmarker (PKB) is an open-source tool designed to measure and compare cloud offerings, providing a standardized approach to performance evaluation. [Explore the PerfKitBenchmarker repo](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker) for the latest updates!**

## Key Features

*   **Automated Benchmarking:** Automates the setup, execution, and tear-down of benchmarks across multiple cloud providers.
*   **Vendor-Agnostic:** Utilizes command-line tools from various cloud providers, ensuring a consistent comparison.
*   **Comprehensive Benchmarks:** Includes a wide array of benchmarks covering various aspects of cloud performance.
*   **Flexible Configuration:** Allows for customizable configurations to match your specific needs and scenarios.
*   **Data Visualization:** Supports publishing results to Elasticsearch and InfluxDB for data analysis and visualization.
*   **Extensible:** Open-source and easy to extend with new benchmarks, providers, and features.

## Getting Started

### Installation and Setup

1.  **Prerequisites:**
    *   Python 3.12 or higher is recommended.
    *   Cloud provider accounts (e.g., GCP, AWS, Azure) and necessary permissions are needed for the providers you want to benchmark (see [providers](perfkitbenchmarker/providers/README.md)).

2.  **Install Python 3.12**
    ```bash
    # install pyenv to install python on persistent home directory
    curl https://pyenv.run | bash

    # add to path
    echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

    # update bashrc
    source ~/.bashrc

    # install python 3.12 and make default
    pyenv install 3.12
    pyenv global 3.12
    ```
3.  **Clone the Repository:**
    ```bash
    $ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    $ cd PerfKitBenchmarker
    ```
4.  **Install Dependencies:**
    ```bash
    $ pip3 install -r requirements.txt
    ```

    Install provider-specific dependencies as needed within the respective provider directories (e.g., `cd perfkitbenchmarker/providers/aws && pip3 install -r requirements.txt`).

### Running a Benchmark

To run a benchmark, use the following format:

```bash
$ ./pkb.py --cloud=<cloud provider> --benchmarks=<benchmark name> --machine_type=<machine type>
```

**Examples:**

*   **GCP:** `./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro`
*   **AWS:** `./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro`

Refer to the [examples section](#running-a-single-benchmark) in the original README for running on other providers.

### Useful Flags

*   `--benchmarks`: Comma-separated list of benchmarks (e.g., `iperf,ping` or `standard_set`).
*   `--cloud`: Cloud provider (e.g., `GCP`, `AWS`, `Azure`).  Defaults to `GCP`.
*   `--machine_type`: Specifies the virtual machine instance type.
*   `--zones`: Overrides the default zone for a provider.
*   `--data_disk_type`: Specifies the type of disk to use (e.g., `pd-ssd`, `gp3`).

## Advanced Usage

*   **Preprovisioned Data:** Learn how to upload and use preprovisioned data for specific benchmarks. See [Preprovisioned Data](#preprovisioned-data) for details.
*   **Configurations:** Leverage YAML-based configuration files for complex setups and overrides. See [Configurations and Configuration Overrides](#configurations-and-configuration-overrides) for details.
*   **Static VMs:** Run benchmarks on existing, non-provisioned machines.  See [Advanced: How To Run Benchmarks Without Cloud Provisioning (e.g., local workstation)](#advanced-how-to-run-benchmarks-without-cloud-provisioning-e.g-local-workstation) for details.
*   **Elasticsearch & InfluxDB Publishing:**  Publish your results. See [Using Elasticsearch Publisher](#using-elasticsearch-publisher) and [Using InfluxDB Publisher](#using-influxdb-publisher).

## Licensing

PerfKit Benchmarker uses a variety of open-source benchmarks with individual licensing terms.  You are responsible for reviewing and adhering to the licenses of the benchmarks used.  You must use the `--accept-licenses` flag when running PKB.

## Extending and Contributing

Contribute to the project by adding new benchmarks, providers, or features!  See [How to Extend PerfKit Benchmarker](#how-to-extend-perfkit-benchmarker) for details.  We welcome contributions!

## Additional Resources

*   [Wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki):  Contains detailed information and documentation.
*   [Issues](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues):  Report issues and request features.