# PerfKit Benchmarker: The Open Source Cloud Benchmarking Tool

**PerfKit Benchmarker (PKB) is your one-stop-shop for standardized cloud performance testing, enabling you to compare and evaluate various cloud offerings effectively.  [View on GitHub](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker)**

## Key Features:

*   **Automated Benchmarking:** Automatically provisions VMs on various cloud providers, installs benchmarks, and runs workloads without user interaction.
*   **Cross-Cloud Comparison:** Designed to run benchmarks consistently across multiple cloud platforms (GCP, AWS, Azure, etc.) for unbiased comparisons.
*   **Extensive Benchmark Suite:** Provides a comprehensive collection of popular benchmarks covering compute, storage, network, and database performance.
*   **Configuration Flexibility:** Allows customization through YAML configuration files and command-line overrides for tailored testing.
*   **Open Source and Extensible:**  Open-source tool, making it easy to add new benchmarks, support new cloud providers, and contribute to the community.
*   **Flexible Deployment Options:** Supports running benchmarks on various platforms including VMs on cloud providers, local machines, and Kubernetes.
*   **Result Publishing:** Supports publishing the result metrics to Elasticsearch and InfluxDB.

## Getting Started

### Installation and Setup

1.  **Prerequisites:**  Ensure you have accounts on the cloud providers you intend to benchmark.  Install Python 3 (at least version 3.11) and a virtual environment is recommended.

```bash
python3 -m venv $HOME/my_virtualenv
source $HOME/my_virtualenv/bin/activate
```

2.  **Clone and Install:** Clone the PerfKit Benchmarker repository and install dependencies.

```bash
git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
cd PerfKitBenchmarker
pip3 install -r requirements.txt
```

3.  **Provider-Specific Dependencies:** Install dependencies for the cloud providers you'll be using (e.g., for AWS, install requirements from `perfkitbenchmarker/providers/aws/requirements.txt`).

### Running a Single Benchmark

Use the command-line interface to run benchmarks.  Here are a few examples:

*   **GCP:**  `./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro`
*   **AWS:**  `./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro`
*   **Azure:** `./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf`

(See the full README for other examples)

### Additional Information:
*   **Tutorials:**  Beginner and Docker tutorials are available in the `tutorials` directory to guide you.
*   **Advanced Configurations:**  Utilize YAML configuration files for complex setups, and override settings with the `--config_override` flag.
*   **Licensing:**  Review the licenses of the included benchmark tools before using.  You must accept the licenses with the `--accept-licenses` flag.

## Advanced Features

*   **Preprovisioned Data:** Some benchmarks require data uploads to your cloud provider's object storage. See the README for details.
*   **Static Machines:** Run benchmarks on local or non-provisioned machines using the `static_vms` configuration.
*   **Configuration File Flags:** Specify command-line flags in configuration files.
*   **Elasticsearch/InfluxDB Publishing:** Publish benchmark results to Elasticsearch or InfluxDB for analysis and visualization.

## Extending PerfKit Benchmarker

*   **Contribution Guidelines:** Refer to `CONTRIBUTING.md` for details on how to contribute.
*   **Documentation:** Explore the [wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki) for detailed documentation.

## Integration Testing

Run integration tests with `tox -e integration`.  Requires proper configuration of cloud provider SDKs.