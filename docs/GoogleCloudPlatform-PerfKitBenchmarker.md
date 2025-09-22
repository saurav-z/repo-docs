# PerfKit Benchmarker: The Open-Source Standard for Cloud Performance Benchmarking

**Measure and compare cloud offerings with ease using PerfKit Benchmarker, an open-source toolkit for standardized performance testing.** ([Original Repo](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker))

PerfKit Benchmarker (PKB) is a powerful tool designed to provide a consistent and automated way to benchmark cloud services.  It helps you understand the performance of various cloud platforms and configurations by running a comprehensive set of benchmarks.  PKB leverages vendor-provided command-line tools, automating VM instantiation, benchmark installation, and workload execution, providing reliable results.

## Key Features:

*   **Automated Benchmarking:**  PKB automates the entire benchmarking process, from cloud resource provisioning to result collection.
*   **Standardized Workloads:**  Offers a canonical set of benchmarks to ensure consistent and comparable results across different cloud providers.
*   **Cloud Provider Support:** Supports multiple cloud platforms, including GCP, AWS, Azure, and more.
*   **Flexible Configuration:** Easily customize benchmarks and machine types through configuration files and command-line flags.
*   **Extensible:**  Supports adding new benchmarks, cloud providers, and OS types, allowing you to customize the tool to your needs.
*   **Licensing Information**: Detailed licensing information to make sure your use case is compliant.

## Getting Started

### 1. Installation and Setup

Follow these steps to get PerfKit Benchmarker up and running:

*   **Python 3 and Virtual Environment:**
    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```
*   **Install PerfKit Benchmarker:**
    ```bash
    $ cd $HOME
    $ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    ```
*   **Install Dependencies:**
    ```bash
    $ pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
    ```
    *   Also install the provider's specific requirements:
    ```bash
    $ cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/aws
    $ pip3 install -r requirements.txt
    ```

### 2. Running a Benchmark

PKB provides a simple command-line interface for running benchmarks. Here are a few examples:

*   **GCP:**
    ```bash
    ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
    ```
*   **AWS:**
    ```bash
    ./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro
    ```
*   **Azure:**
    ```bash
    ./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf
    ```

(See the original README for more provider examples.)

### 3. Additional Resources

*   **Comprehensive Documentation:** The [wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki) provides detailed information on various aspects of the tool.
*   **Contribute:**  We welcome contributions!  See [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) for more information.
*   **Community:** Join the discussion on #PerfKitBenchmarker on freenode.
*   **Report Issues:**  [Open an issue](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues) if you encounter any problems.

---

**For detailed information on specific benchmarks, configurations, and advanced usage, please refer to the original README and the PerfKit Benchmarker wiki.**