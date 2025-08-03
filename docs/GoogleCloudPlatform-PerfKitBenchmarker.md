# PerfKit Benchmarker: The Open Source Standard for Cloud Benchmarking

**PerfKit Benchmarker (PKB) empowers you to rigorously benchmark and compare cloud offerings using a comprehensive set of tests.**  [Visit the original repository](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker) for more information.

**Key Features:**

*   **Automated Cloud Resource Provisioning:** PKB simplifies benchmarking by automatically provisioning VMs on various cloud platforms.
*   **Comprehensive Benchmark Suite:** Runs a wide array of industry-standard benchmarks, ensuring a robust evaluation.
*   **Vendor-Agnostic Design:** Operates via vendor-provided command-line tools for consistent results across platforms.
*   **Customizable Configurations:** Easily configure benchmarks to suit your specific needs using YAML-based configuration files.
*   **Extensible and Open Source:**  Contribute and tailor PKB to meet your evolving benchmarking requirements.
*   **Support for Multiple Cloud Providers:**
    *   GCP
    *   AWS
    *   Azure
    *   DigitalOcean
    *   IBMCloud
    *   AliCloud
    *   OpenStack
    *   CloudStack
    *   Rackspace
    *   Kubernetes
    *   ProfitBricks
    *   Mesos

**Get Started Quickly:**

*   [Beginner Tutorial](./tutorials/beginner_walkthrough)
*   [Docker Tutorial](./tutorials/docker_walkthrough)

## Installation and Setup

1.  **Prerequisites:**  Ensure you have accounts on the cloud providers you intend to benchmark and have the necessary command-line tools and credentials configured.

2.  **Python 3 Setup:** Create a virtual environment (Python 3.11 or later recommended):

    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```

3.  **Install PerfKit Benchmarker:**

    *   Download the latest release from [GitHub](http://github.com/GoogleCloudPlatform/PerfKitBenchmarker/releases) or clone the repository:

        ```bash
        $ cd $HOME
        $ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
        ```

    *   Install Python library dependencies:

        ```bash
        $ pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
        ```

    *   Install cloud provider specific requirements

        ```bash
        $ cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/<cloud-provider>
        $ pip3 install -r requirements.txt
        ```

## Running a Single Benchmark

Below are example commands for running the iperf benchmark on various cloud platforms:

```bash
# GCP
./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro

# AWS
./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro

# Azure
./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf

# IBMCloud
./pkb.py --cloud=IBMCloud --machine_type=cx2-4x8 --benchmarks=iperf

# AliCloud
./pkb.py --cloud=AliCloud --machine_type=ecs.s2.large --benchmarks=iperf

# DigitalOcean
./pkb.py --cloud=DigitalOcean --machine_type=16gb --benchmarks=iperf

# OpenStack
./pkb.py --cloud=OpenStack --machine_type=m1.medium --openstack_network=private --benchmarks=iperf

# Kubernetes
./pkb.py --vm_platform=Kubernetes --benchmarks=iperf --kubeconfig=/path/to/kubeconfig --use_k8s_vm_node_selectors=False

# Mesos
./pkb.py --cloud=Mesos --benchmarks=iperf --marathon_address=localhost:8080

# CloudStack
./pkb.py --cloud=CloudStack --benchmarks=ping --cs_network_offering=DefaultNetworkOffering

# Rackspace
./pkb.py --cloud=Rackspace --machine_type=general1-2 --benchmarks=iperf

# ProfitBricks
./pkb.py --cloud=ProfitBricks --machine_type=Small --benchmarks=iperf
```

## Running Windows Benchmarks

To benchmark on Windows, use the `--os_type=windows` flag. Ensure `smbclient` is installed. Windows has a different set of benchmarks, which can be found under [`perfkitbenchmarker/windows_benchmarks/`](perfkitbenchmarker/windows_benchmarks).  The target VM OS is Windows Server 2012 R2.

## Running Benchmarks with Juju

Use the `--os_type=juju` flag for automated deployment using Juju. Example:
```bash
./pkb.py --cloud=AWS --os_type=juju --benchmarks=cassandra_stress
```

## Additional Information

*   **Useful Global Flags:** See the README for a table of helpful global flags.
*   **Configurations and Configuration Overrides:** Learn how to customize benchmark settings using YAML files or the `--config_override` flag.
*   **Advanced: Benchmarking Without Cloud Provisioning:**  Run benchmarks on local machines using the static VM configuration.
*   **Extending PerfKit Benchmarker:**  Contribute new features by following the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) guidelines.