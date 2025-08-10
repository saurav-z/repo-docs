# PerfKit Benchmarker: Cloud Performance Benchmarking and Comparison

**Effortlessly measure and compare cloud offerings with PerfKit Benchmarker, an open-source tool designed for consistent and automated performance evaluations.** [Visit the original repo](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker) to dive deeper.

## Key Features:

*   **Automated Benchmarking:** Automates the entire process, from VM instantiation to benchmark execution, minimizing user interaction.
*   **Vendor-Agnostic:** Designed to work across multiple cloud providers, enabling fair comparison.
*   **Comprehensive Benchmarks:** Includes a wide array of benchmarks covering various aspects of cloud performance.
*   **Configurable:** Offers flexible configurations to tailor benchmarks to specific needs, supporting YAML-based configurations for fine-grained control.
*   **Reporting & Analysis:** Supports publishing results to Elasticsearch and InfluxDB for in-depth analysis.
*   **Open Source & Community Driven:** Benefit from a collaborative effort with continuous improvements and community support.

## Quick Start

1.  **Prerequisites:**  Ensure you have Python 3 (at least 3.11) and a virtual environment set up, with access to the cloud providers you want to benchmark.

```bash
python3 -m venv $HOME/my_virtualenv
source $HOME/my_virtualenv/bin/activate
```

2.  **Installation:**

```bash
git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
cd PerfKitBenchmarker
pip3 install -r requirements.txt
```
(Install any provider-specific requirements as needed, e.g. AWS:  `pip3 install -r perfkitbenchmarker/providers/aws/requirements.txt`)

3.  **Run a Benchmark (Example: iperf on GCP):**

```bash
./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
```

## Core Concepts and Usage:

*   **Providers:** PerfKit Benchmarker supports benchmarking on various cloud providers like GCP, AWS, Azure, and others. Use the `--cloud` flag to specify your target.
*   **Benchmarks:** Choose from a diverse set of benchmarks with the `--benchmarks` flag. Run a single benchmark with `--benchmarks=iperf` or a set with `--benchmarks=standard_set`.
*   **Configurations:**  Customize benchmark settings and machine specifications using YAML configuration files or the `--config_override` flag. Explore detailed examples in the [Advanced section](#advanced-how-to-run-benchmarks-without-cloud-provisioning-e.g-local-workstation).
*   **Preprovisioned Data:** Some benchmarks need pre-existing data.  Follow instructions in the readme for pre-provisioning data to cloud buckets.

## Advanced Topics:

*   **[Licensing](#licensing)**: Understand the licensing requirements for individual benchmarks.
*   **[Installation and Setup](#installation-and-setup)**: Provides further instruction for installation.
*   **[Getting Started](#getting-started)**: Tutorial for the architectture, flags, and data visualization of PKB.
*   **[How to Run Benchmarks Without Cloud Provisioning](#advanced-how-to-run-benchmarks-without-cloud-provisioning-e.g-local-workstation)**: Run benchmarks on local machines or using static IPs.
*   **[Running Selective Stages of a Benchmark](#running-selective-stages-of-a-benchmark)**: Control which steps of a benchmark are executed.
*   **[Specifying Flags in Configuration Files](#specifying-flags-in-configuration-files)**:  Set flag defaults within your configuration files.
*   **[Publishing Results to Elasticsearch and InfluxDB](#using-elasticsearch-publisher)**:  Integrate with your data analysis tools.
*   **[Extending PerfKit Benchmarker](#how-to-extend-perfkit-benchmarker)**: Contribute to the project by adding new benchmarks, providers, or features.

**For comprehensive information and guidance, refer to the official documentation on the [wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki).**