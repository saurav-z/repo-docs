# PerfKit Benchmarker: The Open-Source Standard for Cloud Performance Measurement

**Are you looking to accurately benchmark and compare the performance of different cloud offerings?**  PerfKit Benchmarker is your go-to solution, providing a comprehensive suite of benchmarks and automated testing capabilities. [Check out the original repo here!](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker)

**Key Features:**

*   **Automated Benchmarking:**  Automates the entire process, from VM instantiation to benchmark execution and results reporting.
*   **Vendor Agnostic:**  Designed to work across all major cloud providers (GCP, AWS, Azure, DigitalOcean, and more!), ensuring fair and consistent comparisons.
*   **Extensive Benchmark Library:**  Supports a wide range of benchmarks, including network, storage, CPU, database, and more.
*   **Customizable Configurations:**  Offers flexible configuration options to fine-tune benchmarks for specific needs and cloud environments.
*   **Easy Integration:**  Simple installation and setup with clear tutorials and documentation to get you started quickly.
*   **Open Source:**  Benefit from community contributions and continuous improvement.

## Getting Started

1.  **Prerequisites:** Ensure you have accounts on the cloud providers you plan to benchmark and the necessary command-line tools.  Review the [providers README](perfkitbenchmarker/providers/README.md) for detailed information on provider-specific requirements.
2.  **Python Setup:** The project is built for Python 3.11 or newer. Install the latest virtual environment and activate it:

```bash
python3 -m venv $HOME/my_virtualenv
source $HOME/my_virtualenv/bin/activate
```

3.  **Installation:**

    *   Clone the repository:
        ```bash
        $ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
        ```
    *   Install dependencies:
        ```bash
        $ pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
        ```

    *   Install cloud provider-specific requirements (e.g., for AWS):
        ```bash
        $ cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/aws
        $ pip3 install -r requirements.txt
        ```

4.  **Running a Benchmark:**  Use the `pkb.py` script to run benchmarks.  Here are a few examples:

    *   **GCP:**
        ```bash
        $ ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
        ```
    *   **AWS:**
        ```bash
        $ cd PerfKitBenchmarker
        $ ./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro
        ```
    *   **Azure:**
        ```bash
        $ ./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf
        ```
        (Replace placeholders `<GCP project ID>`, `t2.micro`, `Standard_A0` as needed)

    *   Other clouds, including DigitalOcean, IBM Cloud, AliCloud, and OpenStack are also supported. For a complete list of supported providers and how to run benchmarks on them, see the [examples](#running-a-single-benchmark) section.
5.  **Advanced Usage:**  Explore the provided tutorials, documentation, and community resources for in-depth information on configuration, customizations, and contributions.

## Licensing

PerfKit Benchmarker provides wrappers and workload definitions around popular benchmark tools. Users are responsible for accepting the licenses of each benchmark individually. You will need to run PKB with the explicit flag `--accept-licenses`.

## Preprovisioned Data

Some benchmarks require preprovisioned data. Consult the documentation for specific benchmarks and how to upload data to different clouds. For example, to upload data to GCS, use `gsutil`. For AWS, use `aws s3 cp`.

## Configurations and Configuration Overrides

You can configure and customize benchmarks using YAML configuration files or the `--config_override` flag. For example:

```bash
--config_override=cluster_boot.vm_groups.default.vm_count=100
```

## Documentation and Resources

*   [FAQ](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/FAQ)
*   [Tech Talks](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Tech-Talks)
*   [Governing rules](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Governing-Rules)
*   [Community meeting decks and notes](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Community-Meeting-Notes-Decks)
*   [Design documents](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki/Design-Docs)
*   [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md)

## Contribute!

You are welcome to [open an issue](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues) or join us on #PerfKitBenchmarker on freenode to discuss anything related to PerfKitBenchmarker.

## Planned Improvements

Many... please add new requests via GitHub issues.