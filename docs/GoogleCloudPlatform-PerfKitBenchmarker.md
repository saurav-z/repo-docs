# PerfKit Benchmarker: The Open-Source Standard for Cloud Performance Benchmarking

**Tired of cloud performance guesswork?** PerfKit Benchmarker is an open-source tool that helps you rigorously measure and compare cloud offerings by automating the deployment and execution of a comprehensive set of benchmarks.  Access the original repo [here](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker).

## Key Features:

*   **Automated Benchmarking:** Instantly run a variety of pre-configured benchmarks across multiple cloud providers.
*   **Vendor-Neutral:** Designed to work with any cloud provider via command-line tools, promoting consistent and comparable results.
*   **Open and Extensible:** Add your own custom benchmarks and easily extend the tool to support new cloud providers.
*   **Comprehensive Benchmark Suite:** Includes a wide range of popular benchmarks for compute, storage, network, and more.
*   **Flexible Configuration:** Customize benchmark settings and infrastructure deployments through YAML-based configuration files.
*   **Advanced Reporting and Analysis:** Publish results to Elasticsearch and InfluxDB for powerful data visualization and analysis.
*   **Multi-Cloud Support:** Benchmark across Google Cloud Platform (GCP), Amazon Web Services (AWS), Microsoft Azure, and other providers.

## Getting Started

### Installation and Setup

1.  **Python 3 and Virtual Environment:**

    *   Ensure Python 3.11 or greater is installed.
    *   Create and activate a virtual environment:

        ```bash
        python3 -m venv $HOME/my_virtualenv
        source $HOME/my_virtualenv/bin/activate
        ```

2.  **Install PerfKit Benchmarker:**

    *   Clone the repository:

        ```bash
        cd $HOME
        git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
        ```

    *   Install Python dependencies:

        ```bash
        pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
        ```

    *   Install Cloud Provider Specific Dependencies (e.g. AWS)

        ```bash
        cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/aws
        pip3 install -r requirements.txt
        ```

3.  **Cloud Account Access:** Ensure you have the necessary credentials and command-line tools set up for your chosen cloud provider.

4.  **Running a Single Benchmark**

    *   **GCP:**
        ```bash
        ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
        ```

    *   **AWS:**
        ```bash
        ./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro
        ```

    *   **Azure:**
        ```bash
        ./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf
        ```

    *   **(and other providers)**  See README for more examples.

### Benchmark Licensing
Due to the level of automation you will not see prompts for software installed as part of a benchmark run.  You are responsible to accept the license of each of the
benchmarks individually, and take responsibility for using them before you use
the PerfKit Benchmarker.  Run PKB with the explicit flag
`--accept-licenses`.

### Configuration
*   **YAML Configuration:** Customize benchmark configurations via YAML files. Override default settings using `--benchmark_config_file` or `--config_override`.

### Preprovisioned Data

*   Upload necessary data to a cloud storage bucket (GCS for GCP, S3 for AWS).
*   Use flags like `--gcp_preprovisioned_data_bucket` or `--aws_preprovisioned_data_bucket` when running benchmarks to specify the bucket.

### Benchmarks
*   **Standard Set:** Run all standard benchmarks with `--benchmarks="standard_set"`
*   **Named Sets:** Run a set of benchmarks specific to a domain or project with `--benchmarks="<set_name>"`
*   **Individual Benchmarks:** Specify a single benchmark, such as `--benchmarks=iperf`.

## Advanced Usage

*   **Static Machine Execution:** Run PKB on local or pre-existing machines using static VM configurations.
*   **Configuration Files**: You can specify flags in configuration files by using the `flags` key. The expected value is a dictionary mapping flag names to their new default values.
*   **Elasticsearch Publisher:** Publish results to Elasticsearch for advanced data analysis.
*   **InfluxDB Publisher:** Publish results to InfluxDB for data visualization.
*   **Extend PerfKit Benchmarker:**  Add new benchmarks, OS types, or cloud providers.

## Contributions & Community

*   Consult the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) file to get started.
*   Join the community on [#PerfKitBenchmarker on freenode](https://web.libera.chat/)
*   Report issues or submit pull requests on GitHub.
*   Explore the wiki for detailed documentation:
    *   [FAQ](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/FAQ)
    *   [Tech Talks](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Tech-Talks)
    *   [Governing rules](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Governing-Rules)
    *   [Community meeting decks and notes](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Community-Meeting-Notes-Decks)
    *   [Design documents](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Design-Docs)

## Integration Testing

*   Ensure `tox >= 2.0.0` is installed.
*   Run integration tests with `tox -e integration`

## Planned Improvements
*  See GitHub issues for planned improvements.