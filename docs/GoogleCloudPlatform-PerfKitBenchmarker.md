# PerfKit Benchmarker: The Open-Source Standard for Cloud Performance Benchmarking

**Measure and compare cloud offerings with PerfKit Benchmarker, the open-source tool that automates performance testing across multiple cloud providers.** (Original Repo: [https://github.com/GoogleCloudPlatform/PerfKitBenchmarker](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker))

PerfKit Benchmarker (PKB) provides a standardized approach to benchmarking cloud infrastructure, enabling you to make data-driven decisions about your cloud deployments. PKB automates the process of running popular benchmarks, installing required software, and collecting results. This allows for easy and consistent comparisons across various cloud providers and instance types.

**Key Features:**

*   **Automated Benchmarking:** Simplifies the execution of a wide range of benchmarks.
*   **Vendor-Agnostic:** Supports major cloud providers like GCP, AWS, Azure, and more.
*   **Configurable:** Offers extensive configuration options to customize benchmarks.
*   **Standardized Results:** Provides consistent and comparable performance data.
*   **Open Source:**  Freely available for use, modification, and contribution.

## Getting Started

### Prerequisites

*   A cloud provider account (GCP, AWS, Azure, etc.).
*   Python 3.11 or later installed and configured.
*   pip3 installed.

### Installation

1.  **Create a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```

2.  **Clone the Repository:**
    ```bash
    cd $HOME
    git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    ```

3.  **Install Dependencies:**
    ```bash
    pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
    ```
    *   Install cloud provider specific requirements as needed, for example:
        ```bash
        cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/aws
        pip3 install -r requirements.txt
        ```

### Running a Benchmark

Here are some examples of running benchmarks:
   ```bash
    ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
    ./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro
    ./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf
   ```
    See original README for additional example runs on more providers.

### Key Flags

*   `--benchmarks`:  Specify the benchmark(s) to run (e.g., `iperf`, `ping`, or named benchmark sets like `standard_set`).
*   `--cloud`: Select your cloud provider (GCP, AWS, Azure, etc.).
*   `--machine_type`: Choose the instance type for your benchmarks.
*   `--zones`: Override the default zone for a cloud provider.
*   `--data_disk_type`: Specify the disk type for your machines (e.g., `pd-ssd` for GCP).
*   `--config_override`: Override configuration parameters (e.g., `--config_override=cluster_boot.vm_groups.default.vm_count=100`).

## Advanced Usage

*   **Configurations and Configuration Overrides:** Use YAML configuration files for complex setups and customization.
*   **Running on Local Machines:** Run PKB on your local workstation or other non-cloud machines.
*   **Preprovisioned Data:**  Learn how to pre-load data for certain benchmarks.
*   **Extend PerfKit Benchmarker:** Contribute to PKB by adding new benchmarks, providers, and features. See the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) file.
*   **Integration Testing:** Run integration tests (requires `tox` and defined `PERFKIT_INTEGRATION` environment variable).

## Licensing and Benchmarks
PKB uses a variety of benchmarks that each require acceptance of their associated licenses. You must accept the license of each of the benchmarks individually.
See the original README for more details.

## Additional Resources

*   [Wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki) (FAQ, Tech Talks, Design Docs, etc.)
*   [GitHub Issues](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues) (Report issues, request features)
*   [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) (How to contribute)

## Planned Improvements
The project is always looking for improvements. Please submit issues with any ideas.