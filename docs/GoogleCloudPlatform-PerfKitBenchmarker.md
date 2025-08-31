# PerfKit Benchmarker: Cloud Performance Benchmarking Made Easy

**Measure and compare cloud offerings with ease using PerfKit Benchmarker, an open-source tool designed for consistent and reliable performance testing.  [Explore the Original Repo](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker)**

## Key Features:

*   **Automated Benchmarking:**  Automates VM instantiation, benchmark installation, and workload execution on various cloud platforms.
*   **Vendor-Agnostic:**  Uses a canonical set of benchmarks, promoting consistency across different cloud providers.
*   **Extensible:**  Allows you to add new benchmarks, package/OS types, and cloud providers with ease.
*   **YAML-Based Configuration:** Easily configure and customize benchmarks using YAML configuration files, including support for complex setups and multi-cloud testing.
*   **Flexible Deployment:** Supports running benchmarks on various cloud providers (GCP, AWS, Azure, etc.), local machines, and Kubernetes.
*   **Preprovisioned Data Support:** Simplifies the process of using preprovisioned data for benchmarks on Google Cloud and AWS.
*   **Comprehensive Documentation:**  Detailed documentation, including tutorials and a wiki, to help you get started and use the tool effectively.
*   **Data Publishing:** Publish results to Elasticsearch and InfluxDB for advanced analysis and visualization.
*   **Integration Testing:** Run integration tests to create actual cloud resources and ensure that everything works as expected.

## Getting Started

1.  **Prerequisites:**  Ensure you have Python 3 (at least 3.11) and necessary cloud provider accounts/credentials.
2.  **Installation:**
    ```bash
    # Create and activate a virtual environment (recommended)
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate

    # Clone the repository
    git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    cd PerfKitBenchmarker

    # Install dependencies
    pip3 install -r requirements.txt
    ```

3.  **Choose a Tutorial:**

    *   [Beginner tutorial](./tutorials/beginner_walkthrough)
    *   [Docker tutorial](./tutorials/docker_walkthrough)

4.  **Run a Benchmark:**

    *   Use the provided examples for GCP, AWS, Azure, and other providers (see below).

    *   Example: GCP
        ```bash
        ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
        ```
    *   Example: AWS
        ```bash
        ./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro
        ```

## Core Concepts:

*   **Licensing:** Be aware of the licenses associated with the benchmarks.  Use the `--accept-licenses` flag after reviewing the license of each benchmark.
*   **Flags:** Use flags to customize benchmark runs. Common flags include `--benchmarks`, `--cloud`, `--machine_type`, and `--zones`. See the "Useful Global Flags" section below for more details.
*   **Configuration Files:**  Utilize YAML configuration files to specify machine specs, cloud provider settings, and benchmark parameters. Override configurations using the `--config_override` flag.

## Useful Global Flags:

(See the original README for details on flag options.)

| Flag               | Description                                                        |
| ------------------ | ------------------------------------------------------------------ |
| `--helpmatch=pkb`  | Display all global flags.                                          |
| `--benchmarks`     | Comma-separated list of benchmarks (e.g., `iperf,ping`).          |
| `--cloud`          | Specifies the cloud provider (e.g., GCP, AWS, Azure).             |
| `--machine_type`   | Specifies the machine type for the VMs.                          |
| `--zones`          | Specifies the zone/region for the VMs.                            |
| `--data_disk_type` | Specifies the type of data disk (e.g., `pd-ssd`, `gp3`, etc.). |

## Advanced Usage:

*   **Proxy Configuration:** Configure proxy settings for VM guests using flags like `--http_proxy`, `--https_proxy`, and `--ftp_proxy`.
*   **Preprovisioned Data:** Use preprovisioned data for benchmarks that require it by uploading to Google Cloud Storage or S3 and using appropriate flags.
*   **Running Benchmarks Without Cloud Provisioning:**  Run benchmarks on local machines by configuring static VMs in a YAML config file.
*   **Configuration File Flags:** Define default flag values within your configuration files using the `flags` key.
*   **Publishing Results:**  Publish benchmark data to Elasticsearch or InfluxDB.
*   **Extending PerfKit Benchmarker:** Learn how to add new benchmarks, cloud providers, and more by following the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) guide and documentation on the [wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki).
*   **Integration Testing:**  Run the integration tests to ensure that everything works as expected using `tox -e integration`

## License Information

PerfKit Benchmarker is licensed under open source licenses. Please see the original repository for details about licenses related to individual benchmarks.

## Further Exploration:

*   [FAQ](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/FAQ)
*   [Tech Talks](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Tech-Talks)
*   [Governing rules](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Governing-Rules)
*   [Community meeting decks and notes](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Community-Meeting-Notes-Decks)
*   [Design documents](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Design-Docs)
*   [Open an issue](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues)