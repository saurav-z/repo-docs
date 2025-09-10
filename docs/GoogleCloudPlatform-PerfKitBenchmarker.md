# PerfKit Benchmarker: Measure, Compare, and Optimize Your Cloud Performance

**PerfKit Benchmarker (PKB) is an open-source tool that helps you measure and compare the performance of various cloud offerings, empowering you to make informed decisions about your infrastructure. Explore the original repository on [GitHub](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker).**

## Key Features:

*   **Automated Benchmarking:** PKB automates the entire benchmarking process, from provisioning virtual machines to running workloads.
*   **Vendor-Neutral:**  Designed for consistency, PKB leverages common practices and vendor-provided command-line tools for reliable results across different cloud platforms.
*   **Extensive Benchmark Suite:** PKB provides wrappers and workload definitions for a wide array of popular benchmark tools.
*   **Cloud Provider Support:** Supports benchmarking across major cloud providers like GCP, AWS, Azure, DigitalOcean, and more.
*   **Customizable Configurations:**  Flexible configuration options with YAML files and command-line flags, allowing tailored tests.
*   **Integration Testing:** The integration tests can be run by defining the `PERFKIT_INTEGRATION` variable.

## Getting Started

1.  **Prerequisites:** You'll need accounts on the cloud provider(s) you want to benchmark, along with necessary command-line tools and credentials for access.

2.  **Setup:**

    *   **Python 3:**  Recommended to use Python 3.11+ within a virtual environment.
    *   **Install PerfKit Benchmarker:**
        ```bash
        $ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
        $ cd PerfKitBenchmarker
        $ pip3 install -r requirements.txt
        ```
        Also install provider-specific dependencies (e.g., for AWS):
        ```bash
        $ cd perfkitbenchmarker/providers/aws
        $ pip3 install -r requirements.txt
        ```

3.  **Running a Simple Benchmark (Example: iperf on GCP):**
    ```bash
    $ ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
    ```
    Adjust the flags as necessary for other clouds, machine types, and benchmarks.

4.  **Licensing:**  Before running benchmarks, review and accept the licenses for the individual benchmark tools.  Use the `--accept-licenses` flag.  See the full list of benchmarks and their licenses below.

## Advanced Usage and Configuration

*   **Running Specific Benchmarks:** Use the `--benchmarks` flag to select benchmarks.
*   **Running All Standard Benchmarks:**  Run with `--benchmarks="standard_set"`.
*   **Named Sets:** Use named sets to run groups of benchmarks (e.g., `--benchmarks="standard_set"`).
*   **Selective Stages:** Run provision, prepare, and run stages of any benchmark.
*   **Configuration Files:**  Use YAML config files (`--benchmark_config_file`) or the `--config_override` flag to customize benchmark settings.
*   **Static Machines:**  Run benchmarks on your own pre-existing machines (see documentation).
*   **Elasticsearch Publisher:** Publish your PerfKit data to Elasticsearch (install `elasticsearch` package and use `--es_uri`).
*   **InfluxDB Publisher:** Publish your PerfKit data to InfluxDB (use `--influx_uri`).
*   **Extending PKB:**  See the `CONTRIBUTING.md` file and the wiki for instructions on contributing new benchmarks, OS types, providers, etc.

## Licensing Information
See the [original README](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker) for a full list of benchmarks and licensing.

## Useful Global Flags (Partial List)

| Flag               | Description                                            |
| ------------------ | ------------------------------------------------------ |
| `--helpmatch=pkb`  | Show all global flags.                                |
| `--helpmatch=<benchmark>` | Show flags specific to a benchmark.                 |
| `--benchmarks`     | Comma-separated list of benchmarks (e.g., `iperf,ping`). |
| `--cloud`          | Cloud provider (e.g., GCP, AWS).                      |
| `--machine_type`   | VM machine type (provider-specific).                    |
| `--zones`          | Zone to run the benchmark in.                          |
| `--data_disk_type` | Type of data disk to use (provider-specific).          |

## Preprovisioned Data

Some benchmarks require preprovisioned data, such as for `sample_preprovision` or `SPEC CPU2006`.  Follow instructions for each cloud (Google Cloud and AWS) to upload data to a bucket.  See the original README for details.

## Running Windows Benchmarks

Run with `--os_type=windows`.

## Running Benchmarks with Juju

Run with `--os_type=juju`.