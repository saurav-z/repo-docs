# PerfKit Benchmarker: The Open Source Tool for Cloud Performance Evaluation

**Tired of vendor lock-in?  PerfKit Benchmarker (PKB) is a powerful, open-source tool that helps you objectively measure and compare cloud offerings by automating benchmark execution.**  Get started today by visiting the [original repository](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker)!

## Key Features

*   **Automated Benchmarking:**  PKB automates the entire benchmarking process, from provisioning VMs on your chosen cloud provider to installing benchmarks and running workloads.
*   **Cross-Cloud Compatibility:** Supports benchmarking across major cloud providers like GCP, AWS, Azure, and more.
*   **Extensive Benchmark Library:**  Includes a wide range of popular benchmarks, with configurations for consistency and comparability.
*   **Flexible Configuration:**  Customize benchmarks with various flags, YAML configuration files, and override settings to suit your specific needs.
*   **Easy to Extend:**  Open-source and well-documented, making it easy to add new benchmarks, providers, and features.
*   **Integration Testing:** Comprehensive testing suite ensures reliability and accuracy.

## Quick Start

1.  **Installation:**

    *   **Prerequisites:** Python 3 (at least 3.12) and a cloud provider account (GCP, AWS, Azure, etc.). Install Python using your distribution's package manager or by following the steps below:

    ```bash
    # install pyenv to install python on persistent home directory
    curl https://pyenv.run | bash

    # add to path
    echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

    # update bashrc
    source ~/.bashrc

    # install python 3.12 and make default
    pyenv install 3.12
    pyenv global 3.12
    ```

    *   **Clone the repository:**

        ```bash
        $ cd $HOME
        $ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
        ```

    *   **Install dependencies:**

        ```bash
        $ pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
        ```

        Install cloud provider-specific dependencies as needed (e.g., for AWS:  `cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/aws && pip3 install -r requirements.txt`)

2.  **Basic Usage:**

    *   **Run a benchmark (e.g., iperf on GCP):**

        ```bash
        $ ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
        ```

    *   **Example runs for other clouds (replace placeholders with your values):**

        *   **AWS:**  `./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro`
        *   **Azure:**  `./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf`
        *   **(And many more - see the original README for options!)**

3.  **Further Resources:**

    *   **Tutorials:**  Explore the [Beginner tutorial](./tutorials/beginner_walkthrough) or the [Docker tutorial](./tutorials/docker_walkthrough).
    *   **Wiki:**  For detailed information, consult the [wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki) including FAQs, Design Docs, and Community meeting notes.
    *   **Contribution:** [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) for guidance.
    *   **Issues/Questions:** [Open an issue](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues) or join us on #PerfKitBenchmarker on freenode.

## Licensing

PerfKit Benchmarker utilizes and wraps a variety of popular benchmark tools.  **By using PerfKit Benchmarker, you acknowledge that you are responsible for understanding and agreeing to the licenses of the individual benchmarks used, and you must explicitly agree to the licenses using the `--accept-licenses` flag when executing PKB.**  See the original README for the specific licenses of each benchmark.

## More information

*   **Advanced configurations**: See the [configurations and configuration overrides section](#configurations-and-configuration-overrides).
*   **Running selective stages**: See the [running selective stages of a benchmark](#running-selective-stages-of-a-benchmark) section.
*   **Running on local machines**: See the [Advanced: How To Run Benchmarks Without Cloud Provisioning (e.g., local workstation)](#advanced-how-to-run-benchmarks-without-cloud-provisioning-e.g.-local-workstation) section.
*   **How to extend PKB:** See the [How to Extend PerfKit Benchmarker](#how-to-extend-perfkit-benchmarker) section.

## Getting Help

*   **Issues:**  Report bugs and feature requests on the [GitHub Issues page](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues).
*   **Community:** Join the #PerfKitBenchmarker channel on freenode to connect with other users and developers.