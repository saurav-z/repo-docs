# PerfKit Benchmarker: Cloud Performance Benchmarking Made Easy

**Measure and compare cloud offerings with ease using PerfKit Benchmarker, a powerful and automated open-source tool.**  [Explore the PerfKit Benchmarker Repository](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker)

PerfKit Benchmarker (PKB) is an open-source project designed to provide a consistent and reliable way to benchmark cloud performance. It uses vendor-provided command-line tools to automate the process of measuring and comparing various cloud offerings. PKB simplifies the complexities of performance evaluation by automating VM instantiation, benchmark installation, and workload execution, allowing users to focus on the results.

**Key Features:**

*   **Automated Benchmarking:** PKB automates the entire benchmarking process, from VM creation to result collection.
*   **Vendor-Agnostic:**  Works with multiple cloud providers (GCP, AWS, Azure, and others) using a unified framework.
*   **Consistent Results:**  Uses standardized benchmark settings to ensure comparability across different platforms.
*   **Extensible:**  Supports a wide range of benchmarks and is easily extensible to include new ones.
*   **Flexible Configuration:** Allows for detailed customization through configuration files and command-line flags.
*   **Comprehensive Documentation:**  Includes extensive documentation, tutorials, and community support to help you get started.

**Getting Started:**

1.  **Prerequisites:**
    *   Python 3.11 or higher with a virtual environment is highly recommended.
    *   Account(s) on the cloud provider(s) you want to benchmark.
    *   Required command-line tools and credentials for accessing your cloud accounts.

2.  **Installation:**

    ```bash
    # Create and activate a virtual environment (optional but recommended)
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate

    # Clone the repository
    git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git

    # Install dependencies
    cd PerfKitBenchmarker
    pip3 install -r requirements.txt

    # Install provider-specific dependencies (e.g., for AWS)
    cd perfkitbenchmarker/providers/aws
    pip3 install -r requirements.txt
    ```

3.  **Basic Usage:**

    Run a benchmark on a specific cloud:

    ```bash
    ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
    # or
    ./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro
    ```

**Key Considerations**

*   **Licensing:**  Be aware that PKB includes wrappers for popular benchmark tools, each with its own license. Review and accept the licenses before using any of the benchmarks. PKB includes a flag `--accept-licenses` to handle this.
*   **Preprovisioned Data:** Some benchmarks require data to be preprovisioned. See the documentation in the repository for specific instructions.

**Explore Further:**

*   [Wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki): Comprehensive information, including FAQs, tutorials, design documents, and more.
*   [Contribute](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md): Learn how to contribute to the project.
*   [Issues](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues): Report issues, request features, or join the discussion.
*   **Useful Global Flags**: `--helpmatch=pkb` (see all global flags), `--benchmarks` (specify which benchmarks to run), `--cloud` (select the cloud provider), `--machine_type` (choose the VM instance type), `--zones` (specify the zone), `--data_disk_type` (select the type of data disk).

**Enhancements:**

*   [Configurations and Configuration Overrides](#configurations-and-configuration-overrides) Use configuration files to setup more complex benchmarks.
*   [Advanced: Running Without Cloud Provisioning](#advanced-how-to-run-benchmarks-without-cloud-provisioning-e-g-local-workstation) Use static machines for benchmarking.

**Contribute and Get Involved:**

PerfKit Benchmarker is an open-source project, and we welcome contributions from the community. Whether you're an experienced cloud user, a developer, or a performance enthusiast, there are many ways to contribute.