# PerfKit Benchmarker: Cloud Benchmarking for Everyone

**Want to accurately compare cloud offerings? PerfKit Benchmarker, a Google Cloud project, provides a standardized framework for measuring and comparing the performance of various cloud platforms, making it easier than ever to make informed decisions. Explore the [original repo](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker) for more details.**

## Key Features:

*   **Comprehensive Benchmarks:** Run a wide range of industry-standard benchmarks (listed below), covering various aspects of cloud performance.
*   **Automated Execution:**  Automates VM provisioning, benchmark installation, and workload execution, minimizing user interaction.
*   **Vendor-Agnostic:** Designed to work with various cloud providers, enabling cross-platform comparisons (GCP, AWS, Azure, and more).
*   **Configuration Flexibility:** Offers robust configuration options, including YAML-based configuration files, for customizing benchmark parameters.
*   **Extensible Architecture:** Easily add new benchmarks, support new cloud providers, and integrate with different performance reporting tools.
*   **Licensing Compliance:** Ensures compliance by requiring users to accept the licenses of the underlying benchmark tools.

## Getting Started

### Installation

1.  **Prerequisites:**
    *   Python 3.12+ (or a compatible version).  Installation instructions are included in the original README.
    *   Account(s) with your desired cloud provider(s) (GCP, AWS, Azure, etc.).
    *   Necessary command-line tools and credentials for cloud access.
2.  **Install PerfKit Benchmarker:**
    ```bash
    $ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    $ cd PerfKitBenchmarker
    $ pip3 install -r requirements.txt
    ```
    *   Install provider-specific dependencies (e.g., for AWS):
    ```bash
    $ cd PerfKitBenchmarker/perfkitbenchmarker/providers/aws
    $ pip3 install -r requirements.txt
    ```
3.  **Configure Cloud Provider Access:** Set up the necessary credentials (e.g., API keys, service accounts) for your chosen cloud provider(s).

### Running Benchmarks

1.  **Basic Run Example (GCP):**
    ```bash
    $ ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
    ```
2.  **Provider-Specific Examples:** The original README provides detailed examples for running benchmarks on various cloud platforms (AWS, Azure, IBMCloud, AliCloud, DigitalOcean, OpenStack, Kubernetes, CloudStack, Rackspace, ProfitBricks, and others).  Refer to the original README for the full command line examples.
3.  **Advanced Options:**
    *   **Run a specific set of benchmarks:** Use the `--benchmarks` flag.  Example: `--benchmarks=iperf,ping`
    *   **Specify cloud provider:** Use the `--cloud` flag (e.g., `--cloud=AWS`).
    *   **Choose machine type:** Use the `--machine_type` flag (e.g., `--machine_type=t2.micro`).
    *   **Use configuration files:**  Utilize YAML configuration files for complex setups and configuration overrides, using the `--benchmark_config_file` and `--config_override` flags.

## Benchmarks Included (Partial List - See README for Full List)

*   aerospike
*   bonnie++
*   cassandra_ycsb
*   cassandra_stress
*   cloudsuite3.0
*   cluster_boot
*   coremark
*   copy_throughput
*   fio
*   gpu_pcie_bandwidth
*   hadoop_terasort
*   hpcc
*   hpcg
*   iperf
*   memtier_benchmark
*   mesh_network
*   netperf
*   oldisim
*   object_storage_service
*   pgbench
*   ping
*   silo
*   scimark2
*   speccpu2006
*   SHOC
*   sysbench_oltp
*   TensorFlow
*   tomcat
*   unixbench
*   wrk
*   ycsb
*   ...and more!

## Preprovisioned Data

Some benchmarks require preprovisioned data. Instructions for provisioning data for Google Cloud and AWS, and associated flags, are detailed in the original README.

## Customization & Extension

*   **Extend PerfKit Benchmarker:** Follow the guidelines in the `CONTRIBUTING.md` file. You can add new benchmarks, support new OS types, or add new cloud providers.  Refer to the [wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki) for more documentation.
*   **Integration Testing:** Run unit and integration tests to validate your changes (requires `tox >= 2.0.0`).

## Support & Community

*   **Issues:**  Report issues and suggest improvements through [GitHub Issues](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues).
*   **Community:** Join the conversation on `#PerfKitBenchmarker` on freenode.
*   **Tutorials:**  The original README and [wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki) provide extensive documentation, including tutorials and design documents.