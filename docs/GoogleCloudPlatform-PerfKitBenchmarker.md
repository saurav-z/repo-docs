# PerfKit Benchmarker: The Open-Source Cloud Benchmark Tool (Learn More)

**PerfKit Benchmarker (PKB) is an open-source tool designed to provide a standardized and automated way to measure and compare the performance of various cloud offerings.** This README provides a comprehensive overview to help you get started, including installation, usage, and advanced features. Explore the official repository: [https://github.com/GoogleCloudPlatform/PerfKitBenchmarker](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker)

**Key Features:**

*   **Automated Benchmarking:** PKB automates the entire benchmarking process, from VM instantiation to benchmark execution, reducing the need for manual intervention.
*   **Vendor-Agnostic:** Designed to work with various cloud providers (GCP, AWS, Azure, etc.), allowing for consistent comparisons across platforms.
*   **Standardized Benchmarks:** Utilizes a set of canonical benchmarks to ensure consistency and comparability of results.
*   **Flexible Configuration:** Allows users to customize benchmark settings and target specific hardware configurations.
*   **Detailed Reporting:** Generates comprehensive reports with performance metrics for easy analysis.
*   **Extendable:** Offers a robust framework for adding new benchmarks and supporting additional cloud providers.

## Table of Contents
*   [Getting Started](#getting-started)
    *   [Installation and Setup](#installation-and-setup)
    *   [Running a Single Benchmark](#running-a-single-benchmark)
*   [Advanced Usage](#advanced-usage)
    *   [How to Run Windows Benchmarks](#how-to-run-windows-benchmarks)
    *   [How to Run Benchmarks with Juju](#how-to-run-benchmarks-with-juju)
    *   [How to Run All Standard Benchmarks](#how-to-run-all-standard-benchmarks)
    *   [How to Run All Benchmarks in a Named Set](#how-to-run-all-benchmarks-in-a-named-set)
    *   [Running selective stages of a benchmark](#running-selective-stages-of-a-benchmark)
    *   [Useful Global Flags](#useful-global-flags)
    *   [Proxy configuration for VM guests.](#proxy-configuration-for-vm-guests)
    *   [Preprovisioned Data](#preprovisioned-data)
    *   [Configurations and Configuration Overrides](#configurations-and-configuration-overrides)
    *   [Advanced: How To Run Benchmarks Without Cloud Provisioning (e.g., local workstation)](#advanced-how-to-run-benchmarks-without-cloud-provisioning-eg-local-workstation)
    *   [Specifying Flags in Configuration Files](#specifying-flags-in-configuration-files)
*   [Publishing Results](#publishing-results)
    *   [Using Elasticsearch Publisher](#using-elasticsearch-publisher)
    *   [Using InfluxDB Publisher](#using-influxdb-publisher)
*   [Extending PerfKit Benchmarker](#how-to-extend-perfkit-benchmarker)
*   [Integration Testing](#integration-testing)
*   [Planned Improvements](#planned-improvements)
## Getting Started

### Installation and Setup

1.  **Prerequisites:** Ensure you have Python 3 (at least 3.11) installed and available.

    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```

2.  **Install PerfKit Benchmarker:** Clone the repository and install the required Python libraries.

    ```bash
    $ cd $HOME
    $ git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    $ pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
    ```

3.  **Cloud Provider Dependencies:** Install any additional dependencies specific to your chosen cloud provider(s).  Refer to the `requirements.txt` files within each provider's directory in `perfkitbenchmarker/providers/` (e.g., `perfkitbenchmarker/providers/aws/requirements.txt`).

### Running a Single Benchmark

Run benchmarks on various Cloud Providers, using the syntax below as examples.

```bash
$ ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
```
```bash
$ cd PerfKitBenchmarker
$ ./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro
```
```bash
$ ./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf
```
```bash
$ ./pkb.py --cloud=IBMCloud --machine_type=cx2-4x8 --benchmarks=iperf
```
```bash
$ ./pkb.py --cloud=AliCloud --machine_type=ecs.s2.large --benchmarks=iperf
```
```bash
$ ./pkb.py --cloud=DigitalOcean --machine_type=16gb --benchmarks=iperf
```
```bash
$ ./pkb.py --cloud=OpenStack --machine_type=m1.medium \
           --openstack_network=private --benchmarks=iperf
```
```bash
$ ./pkb.py --vm_platform=Kubernetes --benchmarks=iperf \
           --kubeconfig=/path/to/kubeconfig --use_k8s_vm_node_selectors=False
```
```bash
$ ./pkb.py --cloud=Mesos --benchmarks=iperf --marathon_address=localhost:8080
```
```bash
./pkb.py --cloud=CloudStack --benchmarks=ping --cs_network_offering=DefaultNetworkOffering
```
```bash
$ ./pkb.py --cloud=Rackspace --machine_type=general1-2 --benchmarks=iperf
```
```bash
$ ./pkb.py --cloud=ProfitBricks --machine_type=Small --benchmarks=iperf
```

## Advanced Usage

### How to Run Windows Benchmarks
Install all dependencies and run with `--os_type=windows`. Windows has a different set of benchmarks than Linux does. They can be found under [`perfkitbenchmarker/windows_benchmarks/`](perfkitbenchmarker/windows_benchmarks).
### How to Run Benchmarks with Juju
Juju is a service orchestration tool. Supported benchmarks will deploy a Juju-modeled service automatically, with no extra user configuration required, by specifying the `--os_type=juju` flag.

### How to Run All Standard Benchmarks

Run with `--benchmarks="standard_set"` to execute all benchmarks within the standard set.

### How to Run All Benchmarks in a Named Set

Specify the set name in the `--benchmarks` parameter (e.g., `--benchmarks="standard_set"`).

### Running selective stages of a benchmark
This procedure demonstrates how to run only selective stages of a benchmark.
This technique can be useful for examining a machine after it has been prepared,
but before the benchmark runs.

This example shows how to provision and prepare the `cluster_boot` benchmark
without actually running the benchmark.

1.  Change to your local version of PKB: `cd $HOME/PerfKitBenchmarker`

1.  Run provision, prepare, and run stages of `cluster_boot`.

    ```
    ./pkb.py --benchmarks=cluster_boot --machine_type=n1-standard-2 --zones=us-central1-f --run_stage=provision,prepare,run
    ```

1.  The output from the console will tell you the run URI for your benchmark.
    Try to ssh into the VM. The machine "Default-0" came from the VM group which
    is specified in the benchmark_config for cluster_boot.

    ```
    ssh -F /tmp/perfkitbenchmarker/runs/<run_uri>/ssh_config default-0
    ```

1.  Now that you have examined the machines, teardown the instances that were
    made and cleanup.

    ```
    ./pkb.py --benchmarks=cluster_boot --run_stage=teardown -run_uri=<run_uri>
    ```

### Useful Global Flags

| Flag               | Notes                                                 |
| ------------------ | ----------------------------------------------------- |
| `--helpmatch=pkb`  | see all global flags                                  |
| `--helpmatch=hpcc` | see all flags associated with the hpcc benchmark. You |
:                    : can substitute any benchmark name to see the          :
:                    : associated flags.                                     :
| `--benchmarks`     | A comma separated list of benchmarks or benchmark     |
:                    : sets to run such as `--benchmarks=iperf,ping` . To    :
:                    : see the full list, run `./pkb.py                      :
:                    : --helpmatch=benchmarks                                :
| `--cloud`          | Cloud where the benchmarks are run. See the table     |
:                    : below for choices.                                    :
| `--machine_type`   | Type of machine to provision if pre-provisioned       |
:                    : machines are not used. Most cloud providers accept    :
:                    : the names of pre-defined provider-specific machine    :
:                    : types (for example, GCP supports                      :
:                    : `--machine_type=n1-standard-8` for a GCE              :
:                    : n1-standard-8 VM). Some cloud providers support YAML  :
:                    : expressions that match the corresponding VM spec      :
:                    : machine_type property in the [YAML                    :
:                    : configs](#configurations-and-configuration-overrides) :
:                    : (for example, GCP supports `--machine_type="{cpus\:   :
:                    : 1, memory\: 4.5GiB}"` for a GCE custom VM with 1 vCPU :
:                    : and 4.5GiB memory). Note that the value provided by   :
:                    : this flag will affect all provisioned machines; users :
:                    : who wish to provision different machine types for     :
:                    : different roles within a single benchmark run should  :
:                    : use the [YAML                                         :
:                    : configs](#configurations-and-configuration-overrides) :
:                    : for finer control.                                    :
| `--zones`          | This flag allows you to override the default zone.    |
:                    : See the table below.                                  :
| `--data_disk_type` | Type of disk to use. Names are provider-specific, but |
:                    : see table below.                                      :

The default cloud is 'GCP', override with the `--cloud` flag. Each cloud has a
default zone which you can override with the `--zones` flag, the flag supports
the same values that the corresponding Cloud CLIs take:

| Cloud name   | Default zone  | Notes                                       |
| ------------ | ------------- | ------------------------------------------- |
| GCP          | us-central1-a |                                             |
| AWS          | us-east-1a    |                                             |
| Azure        | eastus2       |                                             |
| IBMCloud     | us-south-1    |                                             |
| AliCloud     | West US       |                                             |
| DigitalOcean | sfo1          | You must use a zone that supports the       |
:              :               : features 'metadata' (for cloud config) and  :
:              :               : 'private_networking'.                       :
| OpenStack    | nova          |                                             |
| CloudStack   | QC-1          |                                             |
| Rackspace    | IAD           | OnMetal machine-types are available only in |
:              :               : IAD zone                                    :
| Kubernetes   | k8s           |                                             |
| ProfitBricks | AUTO          | Additional zones: ZONE_1, ZONE_2, or ZONE_3 |

Example:

```bash
./pkb.py --cloud=GCP --zones=us-central1-a --benchmarks=iperf,ping
```

The disk type names vary by provider, but the following table summarizes some
useful ones. (Many cloud providers have more disk types beyond these options.)

Cloud name | Network-attached SSD | Network-attached HDD
---------- | -------------------- | --------------------
GCP        | pd-ssd               | pd-standard
AWS        | gp3                  | st1
Azure      | Premium_LRS          | Standard_LRS
Rackspace  | cbs-ssd              | cbs-sata

Also note that `--data_disk_type=local` tells PKB not to allocate a separate
disk, but to use whatever comes with the VM. This is useful with AWS instance
types that come with local SSDs, or with the GCP `--gce_num_local_ssds` flag.

If an instance type comes with more than one disk, PKB uses whichever does *not*
hold the root partition. Specifically, on Azure, PKB always uses `/dev/sdb` as
its scratch device.

### Proxy configuration for VM guests.

If the VM guests do not have direct Internet access in the cloud environment,
you can configure proxy settings through `pkb.py` flags.

To do that simple setup three flags (All urls are in notation ): The flag values
use the same `<protocol>://<server>:<port>` syntax as the corresponding
environment variables, for example `--http_proxy=http://proxy.example.com:8080`
.

| Flag            | Notes                                                   |
| --------------- | ------------------------------------------------------- |
| `--http_proxy`  | Needed for package manager on Guest OS and for some     |
:                 : Perfkit packages                                        :
| `--https_proxy` | Needed for package manager or Ubuntu guest and for from |
:                 : Github downloaded packages                              :
| `--ftp_proxy`   | Needed for some Perfkit packages                        |

### Preprovisioned Data

As mentioned above, some benchmarks require preprovisioned data. This section
describes how to preprovision this data.

#### Sample Preprovision Benchmark

This benchmark demonstrates the use of preprovisioned data. Create the following
file to upload using the command line:

```bash
echo "1234567890" > preprovisioned_data.txt
```

To upload, follow the instructions below with a filename of
`preprovisioned_data.txt` and a benchmark name of `sample`.

### Clouds with Preprovisioned Data

#### Google Cloud

To preprovision data on Google Cloud, you will need to upload each file to
Google Cloud Storage using gsutil. First, you will need to create a storage
bucket that is accessible from VMs created in Google Cloud by PKB. Then copy
each file to this bucket using the command

```bash
gsutil cp <filename> gs://<bucket>/<benchmark-name>/<filename>
```

To run a benchmark on Google Cloud that uses the preprovisioned data, use the
flag `--gcp_preprovisioned_data_bucket=<bucket>`.

#### AWS

To preprovision data on AWS, you will need to upload each file to S3 using the
AWS CLI. First, you will need to create a storage bucket that is accessible from
VMs created in AWS by PKB. Then copy each file to this bucket using the command

```bash
aws s3 cp <filename> s3://<bucket>/<benchmark-name>/<filename>
```

To run a benchmark on AWS that uses the preprovisioned data, use the flag
`--aws_preprovisioned_data_bucket=<bucket>`.

### Configurations and Configuration Overrides

Each benchmark now has an independent configuration which is written in YAML.
Users may override this default configuration by providing a configuration. This
allows for much more complex setups than previously possible, including running
benchmarks across clouds.

A benchmark configuration has a somewhat simple structure. It is essentially
just a series of nested dictionaries. At the top level, it contains VM groups.
VM groups are logical groups of homogenous machines. The VM groups hold both a
`vm_spec` and a `disk_spec` which contain the parameters needed to create
members of that group. Here is an example of an expanded configuration:

```yaml
hbase_ycsb:
  vm_groups:
    loaders:
      vm_count: 4
      vm_spec:
        GCP:
          machine_type: n1-standard-1
          image: ubuntu-16-04
          zone: us-central1-c
        AWS:
          machine_type: m3.medium
          image: ami-######
          zone: us-east-1a
        # Other clouds here...
      # This specifies the cloud to use for the group. This allows for
      # benchmark configurations that span clouds.
      cloud: AWS
      # No disk_spec here since these are loaders.
    master:
      vm_count: 1
      cloud: GCP
      vm_spec:
        GCP:
          machine_type:
            cpus: 2
            memory: 10.0GiB
          image: ubuntu-16-04
          zone: us-central1-c
        # Other clouds here...
      disk_count: 1
      disk_spec:
        GCP:
          disk_size: 100
          disk_type: standard
          mount_point: /scratch
        # Other clouds here...
    workers:
      vm_count: 4
      cloud: GCP
      vm_spec:
        GCP:
          machine_type: n1-standard-4
          image: ubuntu-16-04
          zone: us-central1-c
        # Other clouds here...
      disk_count: 1
      disk_spec:
        GCP:
          disk_size: 500
          disk_type: remote_ssd
          mount_point: /scratch
        # Other clouds here...
```

For a complete list of keys for `vm_spec`s and `disk_spec`s see
[`virtual_machine.BaseVmSpec`](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/perfkitbenchmarker/virtual_machine.py)
and
[`disk.BaseDiskSpec`](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/perfkitbenchmarker/disk.py)
and their derived classes.

User configs are applied on top of the existing default config and can be
specified in two ways. The first is by supplying a config file via the
`--benchmark_config_file` flag. The second is by specifying a single setting to
override via the `--config_override` flag.

A user config file only needs to specify the settings which it is intended to
override. For example if the only thing you want to do is change the number of
VMs for the `cluster_boot` benchmark, this config is sufficient:

```yaml
cluster_boot:
  vm_groups:
    default:
      vm_count: 100
```

You can achieve the same effect by specifying the `--config_override` flag. The
value of the flag should be a path within the YAML (with keys delimited by
periods), an equals sign, and finally the new value:

```bash
--config_override=cluster_boot.vm_groups.default.vm_count=100
```

See the section below for how to use static (i.e. pre-provisioned) machines in
your config.

### Advanced: How To Run Benchmarks Without Cloud Provisioning (e.g., local workstation)

It is possible to run PerfKit Benchmarker without running the Cloud provisioning
steps. This is useful if you want to run on a local machine, or have a benchmark
like iperf run from an external point to a Cloud VM.

In order to do this you need to make sure:

*   The static (i.e. not provisioned by PerfKit Benchmarker) machine is ssh'able
*   The user PerfKitBenchmarker will login as has 'sudo' access. (*** Note we
    hope to remove this restriction soon ***)

Next, you will want to create a YAML user config file describing how to connect
to the machine as follows:

```yaml
static_vms:
  - &vm1 # Using the & character creates an anchor that we can
         # reference later by using the same name and a * character.
    ip_address: 170.200.60.23
    user_name: voellm
    ssh_private_key: /home/voellm/perfkitkeys/my_key_file.pem
    zone: Siberia
    disk_specs:
      - mount_point: /data_dir
```

*   The `ip_address` is the address where you want benchmarks to run.
*   `ssh_private_key` is where to find the private ssh key.
*   `zone` can be anything you want. It is used when publishing results.
*   `disk_specs` is used by all benchmarks which use disk (i.e., `fio`,
    `bonnie++`, many others).

In the same file, configure any number of benchmarks (in this case just iperf),
and reference the static VM as follows:

```yaml
iperf:
  vm_groups:
    vm_1:
      static_vms:
        - *vm1
```

I called my file `iperf.yaml` and used it to run iperf from Siberia to a GCP VM
in us-central1-f as follows:

```bash
$ ./pkb.py --benchmarks=iperf --machine_type=f1-micro --benchmark_config_file=iperf.yaml --zones=us-central1-f --ip_addresses=EXTERNAL
```

*   `ip_addresses=EXTERNAL` tells PerfKit Benchmarker not to use 10.X (ie
    Internal) machine addresses that all Cloud VMs have. Just use the external
    IP address.

If a benchmark requires two machines like iperf, you can have two machines in
the same YAML file as shown below. This means you can indeed run between two
machines and never provision any VMs in the Cloud.

```yaml
static_vms:
  - &vm1
    ip_address: <ip1>
    user_name: connormccoy
    ssh_private_key: /home/connormccoy/.ssh/google_compute_engine
    internal_ip: 10.240.223.37
    install_packages: false
  - &vm2
    ip_address: <ip2>
    user_name: connormccoy
    ssh_private_key: /home/connormccoy/.ssh/google_compute_engine
    internal_ip: 10.240.234.189
    ssh_port: 2222

iperf:
  vm_groups:
    vm_1:
      static_vms:
        - *vm2
    vm_2:
      static_vms:
        - *vm1
```

### Specifying Flags in Configuration Files

You can now specify flags in configuration files by using the `flags` key at the
top level in a benchmark config. The expected value is a dictionary mapping flag
names to their new default values. The flags are only defaults; it's still
possible to override them via the command line. It's even possible to specify
conflicting values of the same flag in different benchmarks:

```yaml
iperf:
  flags:
    machine_type: n1-standard-2
    zone: us-central1-b
    iperf_sending_thread_count: 2

netperf:
  flags:
    machine_type: n1-standard-8
```

The new defaults will only apply to the benchmark in which they are specified.

## Publishing Results

### Using Elasticsearch Publisher

Requires the `elasticsearch` Python package to be installed.

```bash
$ pip install elasticsearch
```

Use the following flags:

| Flag         | Notes                                                     |
| ------------ | --------------------------------------------------------- |
| `--es_uri`   | The Elasticsearch server address and port (e.g.           |
:              : localhost\:9200)                                          :
| `--es_index` | The Elasticsearch index name to store documents (default: |
:              : perfkit)                                                  :
| `--es_type`  | The Elasticsearch document type (default: result)         |

### Using InfluxDB Publisher

No additional packages need to be installed in order to publish Perfkit data to
an InfluxDB server.

InfluxDB Publisher takes in the flags for the Influx uri and the Influx DB name.
The publisher will default to the pre-set defaults, identified below, if no uri
or DB name is set. However, the user is required to at the very least call the
`--influx_uri` flag to publish data to Influx.

| Flag               | Notes                               | Default        |
| ------------------ | ----------------------------------- | -------------- |
| `--influx_uri`     | The Influx DB address and port.     | localhost:8086 |
:                    : Expects the format hostname\:port   :                :
| `--influx_db_name` | The name of Influx DB database that | perfkit        |
:                    : you wish to publish to or create    :                :

## How to Extend PerfKit Benchmarker

Refer to the [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) file. You will find the documentation we have on the
[wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki).

## Integration Testing

Run unit or integration tests via `tox -e integration`.

## Planned Improvements

Many... please add new requests via GitHub issues.