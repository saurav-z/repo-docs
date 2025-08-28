# PerfKit Benchmarker: Cloud Benchmarking Made Easy

**PerfKit Benchmarker (PKB) is an open-source tool that simplifies benchmarking cloud offerings, allowing you to compare performance with ease.  [Explore the repo!](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker)**

PKB streamlines the process of measuring and comparing cloud performance by automating benchmark execution across various cloud providers.  It leverages vendor-provided command-line tools and offers a consistent benchmarking experience across platforms.

**Key Features:**

*   **Automated Benchmarking:** PKB automates VM instantiation, benchmark installation, and workload execution.
*   **Cloud Provider Support:**  Compatible with GCP, AWS, Azure, IBMCloud, AliCloud, DigitalOcean, OpenStack, CloudStack, Rackspace, ProfitBricks, Kubernetes, and Mesos.
*   **Diverse Benchmarks:** Supports a wide range of popular benchmarks, including iperf, FIO, and YCSB.
*   **Configuration Flexibility:**  Uses YAML-based configurations for customizing benchmarks and supporting multi-cloud setups.
*   **Result Publishing:** Integrates with Elasticsearch and InfluxDB for data visualization and analysis.
*   **Extensible and Customizable:** Easily add new benchmarks, cloud providers, and OS types.

**Getting Started:**

1.  **Setup:**
    *   **Python 3:** Ensure Python 3.11+ and `venv` are installed.
        ```bash
        python3 -m venv $HOME/my_virtualenv
        source $HOME/my_virtualenv/bin/activate
        ```
    *   **Clone and Install:**
        ```bash
        cd $HOME
        git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
        pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
        ```
        (Install cloud-specific requirements if needed, e.g., for AWS:
        `cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/aws && pip3 install -r requirements.txt`)

2.  **Run a Benchmark (Example - iperf on GCP):**
    ```bash
    ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
    ```
    (Adapt `--cloud`, `--machine_type`, and other flags as needed for your chosen provider and benchmark.)

3.  **Installation and Setup**
    See below for additional installation and setup instructions.

**Licensing:**

PKB utilizes various benchmark tools, each with its own license. Ensure you review and accept the licenses of the individual benchmarks before use.  Use the `--accept-licenses` flag when running PKB.  The README provides a complete list of benchmarks and their corresponding licenses.

**Additional Resources:**

*   [Wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki) - Detailed documentation, FAQs, and design documents.
*   [Contributing Guide](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) - Learn how to contribute.
*   [Issues](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues) - Report issues or suggest improvements.
*   [FAQ](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/FAQ)
*   [Tech Talks](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Tech-Talks)
*   [Governing rules](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Governing-Rules)
*   [Community meeting decks and notes](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Community-Meeting-Notes-Decks)
*   [Design documents](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Design-Docs)

**Complete Instructions**

*   [Installation and Setup](#installation-and-setup)
*   [Running a Single Benchmark](#running-a-single-benchmark)
*   [How to Run Windows Benchmarks](#how-to-run-windows-benchmarks)
*   [How to Run Benchmarks with Juju](#how-to-run-benchmarks-with-juju)
*   [How to Run All Standard Benchmarks](#how-to-run-all-standard-benchmarks)
*   [How to Run All Benchmarks in a Named Set](#how-to-run-all-benchmarks-in-a-named-set)
*   [Running selective stages of a benchmark](#running-selective-stages-of-a-benchmark)
*   [Useful Global Flags](#useful-global-flags)
*   [Proxy configuration for VM guests](#proxy-configuration-for-vm-guests)
*   [Preprovisioned Data](#preprovisioned-data)
*   [Configurations and Configuration Overrides](#configurations-and-configuration-overrides)
*   [Advanced: How To Run Benchmarks Without Cloud Provisioning (e.g., local workstation)](#advanced-how-to-run-benchmarks-without-cloud-provisioning-eg-local-workstation)
*   [Specifying Flags in Configuration Files](#specifying-flags-in-configuration-files)
*   [Using Elasticsearch Publisher](#using-elasticsearch-publisher)
*   [Using InfluxDB Publisher](#using-influxdb-publisher)
*   [How to Extend PerfKit Benchmarker](#how-to-extend-perfkit-benchmarker)
*   [Integration Testing](#integration-testing)
*   [Planned Improvements](#planned-improvements)