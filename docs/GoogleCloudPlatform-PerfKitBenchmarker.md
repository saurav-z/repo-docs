# PerfKit Benchmarker: Benchmarking Cloud Offerings with Ease

**PerfKit Benchmarker (PKB) is an open-source tool designed to benchmark and compare cloud offerings using vendor-provided command-line tools, providing a consistent and automated approach to performance evaluation.** [Visit the original repository](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker)

## Key Features:

*   **Automated Benchmarking:** PKB automates the entire benchmarking process, from VM provisioning to benchmark execution, minimizing user interaction.
*   **Vendor-Neutral Design:** Utilizes vendor-provided tools for benchmarking, ensuring consistency across different cloud providers.
*   **Extensive Benchmark Suite:** Includes a wide range of popular benchmarks covering various aspects of cloud performance.
*   **Flexible Configuration:** Supports customizable configurations to tailor benchmarks to specific needs.
*   **Cloud Provider Support:**  Offers support for major cloud providers like GCP, AWS, Azure, and more.
*   **Detailed Documentation:** Comprehensive documentation, including FAQs, tutorials, and design documents.
*   **Easy to Extend:**  Supports adding new benchmarks, package types, OS support, and cloud providers.
*   **Data Publishing**: Integrates publishing results to Elasticsearch or InfluxDB.

## Getting Started

To begin benchmarking, first set up your environment, installing the correct Python version, activating a virtual environment, and installing the required dependencies.

**1. Python Setup**

The recommended way to install and run PKB is in a virtualenv with the latest version of Python 3 (at least Python 3.11). Most Linux distributions and recent Mac OS X versions already have Python 3 installed at `/usr/bin/python3`.

If Python is not installed, you can likely install it using your distribution's package manager, or see the
[Python Download page](https://www.python.org/downloads/).

```bash
python3 -m venv $HOME/my_virtualenv
source $HOME/my_virtualenv/bin/activate
```

**2. Install PerfKit Benchmarker**

Download the latest PerfKit Benchmarker release from
[GitHub](http://github.com/GoogleCloudPlatform/PerfKitBenchmarker/releases). You
can also clone the working version with:

```bash
$ cd $HOME
$ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
```

**3. Install Python Library Dependencies:**

```bash
$ pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
```

You may need to install additional dependencies depending on the cloud provider
you are using. For example, for
[AWS](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/perfkitbenchmarker/providers/aws/requirements.txt):

```bash
$ cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/aws
$ pip3 install -r requirements.txt
```

## Running a Single Benchmark

PKB can run benchmarks both on Cloud Providers (GCP, AWS, Azure,
DigitalOcean) as well as any "machine" you can SSH into.

*   **Example run on GCP:**
    ```bash
    $ ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
    ```
*   **Example run on AWS:**
    ```bash
    $ cd PerfKitBenchmarker
    $ ./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro
    ```
*   **Example run on Azure:**
    ```bash
    $ ./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf
    ```
*   **Example run on IBMCloud:**
    ```bash
    $ ./pkb.py --cloud=IBMCloud --machine_type=cx2-4x8 --benchmarks=iperf
    ```
*   **Example run on AliCloud:**
    ```bash
    $ ./pkb.py --cloud=AliCloud --machine_type=ecs.s2.large --benchmarks=iperf
    ```
*   **Example run on DigitalOcean:**
    ```bash
    $ ./pkb.py --cloud=DigitalOcean --machine_type=16gb --benchmarks=iperf
    ```
*   **Example run on OpenStack:**
    ```bash
    $ ./pkb.py --cloud=OpenStack --machine_type=m1.medium \
               --openstack_network=private --benchmarks=iperf
    ```
*   **Example run on Kubernetes:**
    ```bash
    $ ./pkb.py --vm_platform=Kubernetes --benchmarks=iperf \
               --kubeconfig=/path/to/kubeconfig --use_k8s_vm_node_selectors=False
    ```
*   **Example run on Mesos:**
    ```bash
    $ ./pkb.py --cloud=Mesos --benchmarks=iperf --marathon_address=localhost:8080
    ```
*   **Example run on CloudStack:**
    ```bash
    ./pkb.py --cloud=CloudStack --benchmarks=ping --cs_network_offering=DefaultNetworkOffering
    ```
*   **Example run on Rackspace:**
    ```bash
    $ ./pkb.py --cloud=Rackspace --machine_type=general1-2 --benchmarks=iperf
    ```
*   **Example run on ProfitBricks:**
    ```bash
    $ ./pkb.py --cloud=ProfitBricks --machine_type=Small --benchmarks=iperf
    ```
    **Note**: *This will take some time!*

## Available Benchmarks

PKB supports a wide array of benchmarks. For a full list of available benchmarks you can use the command:
```bash
./pkb.py --helpmatch=benchmarks
```

### Licensing

PerfKit Benchmarker provides wrappers and workload definitions around popular
benchmark tools. Due to the level of automation you will not see prompts for software installed
as part of a benchmark run. Therefore you must accept the license of each of the
benchmarks individually, and take responsibility for using them before you use
the PerfKit Benchmarker. Moving forward, you will need to run PKB with the explicit flag
--accept-licenses.

You must accept the license of each benchmark individually, and you take responsibility for using them before using the PerfKit Benchmarker.

For a list of the required licenses per benchmark, view the original `README`.

## Advanced Usage

*   **How to Run Windows Benchmarks** Install all dependencies as above and ensure that smbclient is installed on your system if you are running on a linux controller and run with `--os_type=windows`. The target VM OS is Windows Server 2012 R2.
*   **How to Run Benchmarks with Juju** Supported benchmarks will deploy a Juju-modeled service automatically, with no extra user configuration required, by specifying the `--os_type=juju` flag.
*   **How to Run All Standard Benchmarks** Run with `--benchmarks="standard_set"`
*   **How to Run All Benchmarks in a Named Set** Specify the set name in the benchmarks parameter (e.g., `--benchmarks="standard_set"`).
*   **Running selective stages of a benchmark** Allows the user to choose specific stages of the benchmark.
*   **Preprovisioned Data:** Describes how to configure with preprovisioned data.
*   **How to Run Benchmarks Without Cloud Provisioning:** Enables running benchmarks on local machines or external points.
*   **Configurations and Configuration Overrides:**  Explains how to use YAML configuration files to customize benchmark settings.
*   **Specifying Flags in Configuration Files:**  Allows you to define default flag values within configuration files.
*   **Using Elasticsearch Publisher:**  Enables publishing benchmark results to Elasticsearch.
*   **Using InfluxDB Publisher:** Publishes Perfkit data to an InfluxDB server.

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) file for guidance on how to contribute.

## Planned Improvements

Suggestions and feature requests are welcome; please submit issues on GitHub.