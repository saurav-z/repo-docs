# PerfKit Benchmarker: The Open-Source Cloud Benchmarking Tool

**Measure, compare, and optimize your cloud performance with PerfKit Benchmarker, a versatile tool for evaluating cloud offerings.  Explore the [original repo](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker) for the latest updates and contributions.**

## Key Features

*   **Automated Benchmarking:**  Easily run a wide range of benchmarks on various cloud providers with minimal user interaction.
*   **Vendor-Agnostic:**  Compare performance across Google Cloud Platform (GCP), Amazon Web Services (AWS), Microsoft Azure, and more.
*   **Extensive Benchmark Suite:** Includes popular benchmarks such as iperf, fio, Hadoop Terasort, and many more, with detailed information on licensing.
*   **Flexible Configuration:**  Customize your tests with YAML configuration files to fine-tune machine types, disk options, and more.
*   **Preprovisioning Support:**  Easily handle benchmarks that require pre-existing data by leveraging Google Cloud Storage and AWS S3.
*   **Elasticsearch and InfluxDB Integration:**  Visualize your benchmark results by publishing them to Elasticsearch or InfluxDB.
*   **Juju Integration:**  Automated benchmark runs for Juju environments.
*   **Extensible and Customizable:**  Extend PerfKit Benchmarker to support new benchmarks, cloud providers, and operating systems.

## Getting Started

### Installation and Setup

1.  **Prerequisites:**
    *   A cloud provider account (GCP, AWS, Azure, etc.).  See [providers](perfkitbenchmarker/providers/README.md) for details.
    *   Python 3 (3.11 recommended) and pip.
    *   Install `git` if you do not have it.
2.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```
3.  **Clone the Repository:**
    ```bash
    cd $HOME
    git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    ```
4.  **Install Dependencies:**
    ```bash
    pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
    ```
    *   Install provider-specific requirements (e.g., AWS):
        ```bash
        cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/aws
        pip3 install -r requirements.txt
        ```

### Running Benchmarks

*   **Basic Example (GCP):**
    ```bash
    ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
    ```

*   **Cloud-Specific Examples:**  Examples are provided for AWS, Azure, IBMCloud, AliCloud, DigitalOcean, OpenStack, Kubernetes, Mesos, CloudStack, Rackspace, and ProfitBricks in the original README.  Adjust the `--cloud` and `--machine_type` flags accordingly.

*   **Running All Standard Benchmarks:**
    ```bash
    ./pkb.py --benchmarks="standard_set"
    ```

*   **Running Benchmarks on a Local Workstation:**  See instructions under "Advanced" in the original README.

## Licensing

PerfKit Benchmarker uses various open-source benchmark tools. **You are responsible for reviewing and accepting the licenses of the individual benchmarks before running them.**  A list of included benchmarks and their licenses can be found in the original README.

## Configuration & Advanced Usage

*   **Configurations and Configuration Overrides:**  Use YAML config files or the `--config_override` flag to customize benchmark settings. See the original README for details.
*   **Running Selective Stages:** Use the `--run_stage` option to run provision, prepare, or teardown the machines.
*   **Preprovisioning Data:**  Upload data to Google Cloud Storage or AWS S3 for benchmarks that require it. See the "Preprovisioned Data" section in the original README.
*   **Using Publishers:** Utilize the Elasticsearch and InfluxDB publishers to store data and visualize results.
*   **Extending PerfKit Benchmarker:**  See the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) file and the original README's "How to Extend" section.

## Additional Resources

*   [Wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki) - Detailed information, FAQs, and design documents.
*   [Issues](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues) -  Report issues or suggest improvements.
*   Community: Join us on #PerfKitBenchmarker on freenode.