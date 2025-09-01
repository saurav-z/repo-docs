# PerfKit Benchmarker: The Ultimate Cloud Benchmarking Tool

**Tired of guesswork? PerfKit Benchmarker (PKB) helps you objectively compare cloud offerings by automating benchmark execution and providing consistent results.**  [Visit the original repository](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker)

PerfKit Benchmarker is an open-source tool designed to measure and compare the performance of various cloud providers. It simplifies the process of benchmarking by automating VM provisioning, benchmark installation, and workload execution using vendor-provided command-line tools.  PKB offers a standardized approach to benchmarking, ensuring consistent results across different services and platforms.

## Key Features:

*   **Automated Benchmarking:**  PKB streamlines the benchmarking process, eliminating manual setup and execution.
*   **Vendor-Agnostic:** Supports multiple cloud providers, enabling cross-platform performance comparisons.
*   **Consistent Results:**  Uses standardized benchmark settings to ensure reliable and comparable data.
*   **Extensible:**  Easily add new benchmarks, cloud providers, and OS types to expand PKB's capabilities.
*   **Flexible Configuration:**  Customize benchmarks using command-line flags, YAML configuration files, and configuration overrides.
*   **Pre-provisioned Data Support:** Facilitates benchmarking scenarios that require pre-existing datasets.
*   **Integration with Popular Tools:**  Publish results to Elasticsearch and InfluxDB for detailed analysis and visualization.
*   **Comprehensive Documentation:**  Detailed [wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki) with FAQs, tutorials, and design documents.

## Quick Start:

1.  **Prerequisites:**
    *   A Python 3 environment (virtualenv recommended).
    *   Cloud provider accounts (GCP, AWS, Azure, etc.).
    *   Required cloud provider and benchmark dependencies.

2.  **Installation:**

    ```bash
    # Clone the repository
    git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    cd PerfKitBenchmarker

    # Create and activate a virtual environment
    python3 -m venv .venv
    source .venv/bin/activate

    # Install dependencies
    pip3 install -r requirements.txt
    ```

3.  **Running a Benchmark (Example: iperf on GCP):**

    ```bash
    ./pkb.py --project=<YOUR_GCP_PROJECT_ID> --benchmarks=iperf --machine_type=f1-micro
    ```

    Adapt the command with appropriate flags for your desired cloud, machine type, and benchmarks. Refer to the detailed documentation for specific instructions on each cloud provider.

## Licensing

PerfKit Benchmarker uses wrappers and workload definitions for popular benchmark tools. You must accept the license of each benchmark you use.  The licenses for the included benchmarks are detailed in the original README. You must also run PKB with the explicit flag `--accept-licenses` to acknowledge the licenses of included benchmarks.

## Running Benchmarks on Windows

Run with `--os_type=windows` to test on Windows Server 2012 R2 VMs.

## Integration with Juju

Supported benchmarks deploy a Juju-modeled service automatically using the `--os_type=juju` flag.

## Advanced Usage:

*   **Running All Standard Benchmarks:**  `--benchmarks="standard_set"`
*   **Running a Named Set of Benchmarks:**  `--benchmarks="<set_name>"`
*   **Running Selective Stages:** `--run_stage=<stage1>,<stage2>,...` (e.g., `--run_stage=provision,prepare`)
*   **Running on Static/Local Machines:** Configure in YAML and reference static VMs.
*   **Configuration Files:** Create YAML files to specify benchmark parameters and overrides.
*   **Elasticsearch Publisher:**  Publish results to Elasticsearch using the `--es_uri`, `--es_index` and `--es_type` flags after installing the `elasticsearch` python package.
*   **InfluxDB Publisher:** Publish results to InfluxDB using the `--influx_uri` and `--influx_db_name` flags

## Contributing

Contributions are welcome!  Refer to the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) file for guidelines. You can also add new benchmarks, package/os type support, providers, and more.

## Get Involved:

*   Open an [issue](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues) to report issues, request features, or ask questions.
*   Join the community on #PerfKitBenchmarker on freenode.
*   Explore the extensive documentation on the [wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki).