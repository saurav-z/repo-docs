# PerfKit Benchmarker: Benchmarking Cloud Offerings Made Easy

**PerfKit Benchmarker empowers you to objectively measure and compare the performance of various cloud providers using a consistent and automated approach. Explore the [original repository](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker) for comprehensive benchmarking tools.**

## Key Features:

*   **Automated Cloud Benchmarking:** Automatically provisions resources, installs benchmarks, and runs workloads across multiple cloud providers.
*   **Vendor-Agnostic Approach:** Designed for consistency across services, utilizing vendor-provided command-line tools to minimize platform-specific tuning.
*   **Extensive Benchmark Suite:** Includes popular benchmarks like iperf, fio, and Hadoop TeraSort, providing a wide range of performance metrics.
*   **Flexible Configuration:** Supports YAML-based configuration and overrides for customized benchmark runs, including support for static (pre-provisioned) machines.
*   **Integration with Cloud Providers:** Supports major cloud providers like GCP, AWS, Azure, and others.
*   **Data Publishing:** Easily publish benchmark results to Elasticsearch and InfluxDB for analysis and visualization.
*   **Extensible Architecture:** Open-source and designed for community contributions, allowing for the addition of new benchmarks, providers, and features.

## Getting Started

1.  **Prerequisites:** Ensure you have Python 3 (3.11+ recommended) installed and set up a virtual environment.
2.  **Installation:** Clone the repository from GitHub: `git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git` and install dependencies: `pip3 install -r requirements.txt`.  Install provider-specific dependencies if needed (e.g., for AWS:  `pip3 install -r perfkitbenchmarker/providers/aws/requirements.txt`).
3.  **Configuration:** Configure your cloud provider accounts with necessary credentials. See documentation for your provider.
4.  **Running a Benchmark:** Use the `pkb.py` script to run benchmarks.

    *   **Example:**  `./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro`

## Important Considerations:

*   **Licensing:**  Be aware of the licenses associated with each benchmark.  The `pkb.py` script requires `--accept-licenses` to run after initial setup.
*   **Preprovisioned Data:** Some benchmarks require preprovisioned data in cloud storage. See the README for Google Cloud and AWS instructions.
*   **SPEC CPU2006:** Due to licensing restrictions, the SPEC CPU2006 benchmark requires manual setup.
*   **Windows Benchmarks:**  Run Windows benchmarks with the `--os_type=windows` flag.
*   **Juju Support:** Run benchmarks with the `--os_type=juju` flag for automated service orchestration using Juju.

## Core Flags:

*   `--benchmarks`: Comma-separated list of benchmarks or named sets to run (e.g., `iperf,ping` or `standard_set`).
*   `--cloud`: Specifies the cloud provider (e.g., `GCP`, `AWS`, `Azure`). Default: `GCP`.
*   `--machine_type`: Specifies the virtual machine type (e.g., `n1-standard-8`).
*   `--zones`: Override the default zone.
*   `--data_disk_type`: Specifies the disk type (e.g., `pd-ssd`, `gp3`).

## Further Information:

*   **Tutorials:** Explore the [Beginner tutorial](./tutorials/beginner_walkthrough) and the [Docker tutorial](./tutorials/docker_walkthrough) for step-by-step guides.
*   **Documentation:**
    *   [FAQ](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/FAQ)
    *   [Tech Talks](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Tech-Talks)
    *   [Governing rules](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Governing-Rules)
    *   [Community meeting decks and notes](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Community-Meeting-Notes-Decks)
    *   [Design documents](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Design-Docs)
*   **Contributions:**  Review the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) file for instructions.  Join the community on `#PerfKitBenchmarker` on freenode or [open an issue](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues) for questions or help.