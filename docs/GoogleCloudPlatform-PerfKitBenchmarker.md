# PerfKit Benchmarker: The Open-Source Tool for Cloud Performance Benchmarking

**Measure and compare cloud offerings with confidence.** PerfKit Benchmarker is a powerful, open-source framework designed to standardize cloud performance evaluation.  Learn more and contribute at [PerfKitBenchmarker's GitHub Repository](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker).

## Key Features

*   **Automated Benchmarking:** Easily run a variety of benchmarks on multiple cloud providers, automating VM instantiation, benchmark installation, and workload execution.
*   **Vendor-Neutral:** Designed to operate consistently across cloud platforms, promoting fair comparisons.
*   **Extensible:**  Supports a wide range of benchmarks, with easy options to add new benchmarks, cloud providers and OS types.
*   **Configuration Flexibility:** Provides robust configuration options through YAML files, allowing for complex setups and overrides.
*   **Detailed Reporting:** Generates comprehensive performance results, including metrics that are easy to view and share.

## Getting Started

### Installation and Setup

1.  **Prerequisites:** Ensure you have a Python 3 environment (at least 3.11) with `venv` installed.
2.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```
3.  **Clone the Repository:**
    ```bash
    $ cd $HOME
    $ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    ```
4.  **Install Dependencies:**
    ```bash
    $ pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
    ```
    Additional provider-specific dependencies may be required.  See `requirements.txt` files in the `perfkitbenchmarker/providers/<provider>` directories (e.g., `perfkitbenchmarker/providers/aws/requirements.txt`).
5.  **Cloud Provider Accounts:** You will need active accounts on the cloud providers you plan to benchmark.

### Running Benchmarks

*   **Example Run (GCP):**
    ```bash
    $ ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
    ```
*   **Example Run (AWS):**
    ```bash
    $ cd PerfKitBenchmarker
    $ ./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro
    ```
*   **Example Run (Azure):**
    ```bash
    $ ./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf
    ```

    See the original README for additional example runs on different cloud providers, as well as flags for running Windows Benchmarks, Benchmarks with Juju, and running all benchmarks in a named set.

### Useful Global Flags

*   `--helpmatch=<keyword>`: Display help for a specific keyword (e.g., `pkb`, benchmark names).
*   `--benchmarks`: Specify a comma-separated list of benchmarks or benchmark sets.
*   `--cloud`: Select the cloud provider (GCP, AWS, Azure, etc.).
*   `--machine_type`: Define the machine type for provisioning.
*   `--zones`: Override the default zone.
*   `--data_disk_type`: Select the type of data disk.

## License and Benchmarks

PerfKit Benchmarker leverages several open-source benchmark tools. Before using this tool, you must accept the licenses of the included benchmarks.  The README includes a complete list of benchmarks, their licenses, and any additional requirements (such as the need to manually install `cpu2006` or download Java JRE).  Use the `--accept-licenses` flag when running to acknowledge these licenses.

## Advanced Usage

*   **Static VMs:**  Run benchmarks on existing machines using a configuration file.
*   **Configurations:** Override default configurations using YAML files and the `--config_override` flag.
*   **Preprovisioned Data:**  Upload and use pre-populated data for certain benchmarks. See the original README for detailed instructions on using the preprovisioned data feature.

## Extending PerfKit Benchmarker

Consult the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) file and explore the wiki for detailed information on extending PerfKit Benchmarker by adding new benchmarks, provider support, or OS support.

## Integration Testing

To run unit and integration tests, ensure you have `tox >= 2.0.0` installed. Integration tests require cloud provider SDKs and are run with the `PERFKIT_INTEGRATION` environment variable set: `tox -e integration`.