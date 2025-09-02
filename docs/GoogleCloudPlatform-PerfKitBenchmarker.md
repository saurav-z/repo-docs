# PerfKit Benchmarker: The Open Source Standard for Cloud Performance Testing

**Want to compare cloud offerings and optimize your infrastructure?** PerfKit Benchmarker (PKB) provides a standardized, automated way to measure and compare cloud performance using a wide array of benchmarks, enabling data-driven decisions about your cloud resources. Check out the [original repo here](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker)!

**Key Features:**

*   **Automated Benchmarking:** PKB automates the entire benchmarking process, from provisioning VMs on various cloud providers (GCP, AWS, Azure, and more) to installing and running benchmarks.
*   **Standardized Benchmarks:** Utilizes a canonical set of industry-standard benchmarks to ensure consistent and reliable performance comparisons.
*   **Cross-Cloud Compatibility:** Supports benchmarking across multiple cloud providers, allowing for direct comparisons and workload portability analysis.
*   **Flexible Configuration:** Offers extensive configuration options through YAML files and command-line flags, enabling fine-grained control and customization.
*   **Extensible Architecture:** Designed to easily accommodate new benchmarks, cloud providers, and operating systems.
*   **Data Publishing:** Integrates with Elasticsearch and InfluxDB for powerful data visualization and analysis.

## Getting Started

### Installation

1.  **Prerequisites:**  Ensure you have Python 3.11 or later installed and a virtual environment set up.
    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```
2.  **Clone the Repository:**
    ```bash
    $ cd $HOME
    $ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    ```
3.  **Install Dependencies:**
    ```bash
    $ pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
    ```
    Install provider-specific requirements, e.g., for AWS:
    ```bash
    $ cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/aws
    $ pip3 install -r requirements.txt
    ```

### Running Benchmarks

Quickly get started using these examples:

*   **GCP:**  `./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro`
*   **AWS:**  `./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro`
*   **Azure:** `./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf`
*   (And many more; see original README for additional cloud examples)

Refer to the original README for specific cloud installation requirements and how to run benchmarks on different platforms, how to configure with YAML files, and other useful flags.

## Licensing

PKB utilizes a variety of open-source benchmarks, each governed by its own license.  **You are responsible for understanding and adhering to the licenses of the benchmarks you execute.**  You must also accept the licenses of the benchmarks individually via the `--accept-licenses` flag when running PKB. See the original README for benchmark licensing details.

## Advanced Usage & Customization

*   **Running Selective Stages:**  Control the execution stages of a benchmark (provision, prepare, run, teardown).
*   **Configuration Overrides:** Use `--config_override` to modify benchmark parameters without altering the default configuration.
*   **Running on Static Machines:** Execute benchmarks on pre-existing machines (local workstations, etc.).
*   **Specifying Flags in Configuration Files:** Set default flag values within YAML configuration files.
*   **Data Publishing:** Publish results to Elasticsearch or InfluxDB for advanced analysis.

## Extending PerfKit Benchmarker

PKB is designed to be extensible. See [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) for details on how to contribute and add new benchmarks, cloud providers, and more.

## Integration Testing

Run integration tests using:
```bash
$ tox -e integration
```
However these tests require cloud credentials and can incur costs.

## Further Information

*   [Wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki): Detailed documentation and tutorials.
*   [Issues](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues):  Report issues or request features.
*   [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) : Contributing details and guidelines.