# PerfKit Benchmarker: The Open Source Cloud Benchmarking Tool

**Measure and compare cloud performance across providers with PerfKit Benchmarker, the open-source benchmarking tool.**

[View the original repository](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker)

## Key Features

*   **Vendor-Neutral Benchmarking:** Provides a consistent and reliable way to measure and compare cloud offerings using vendor-provided command-line tools.
*   **Automated Setup:** Automates VM instantiation, benchmark installation, and workload execution on various cloud providers.
*   **Extensive Benchmark Suite:**  Supports a wide array of benchmarks, including Aerospike, Bonnie++, Cassandra, Coremark, FIO, iperf, and many more.  See the detailed benchmark list below.
*   **Flexible Configuration:** Allows customization through command-line flags and YAML configuration files, enabling tailored benchmarking for specific needs.
*   **Cross-Cloud Support:**  Supports benchmarking on multiple cloud platforms, including GCP, AWS, Azure, DigitalOcean, OpenStack, and more.
*   **Integration Testing:**  Includes integration tests to create actual cloud resources, verifying the functionality of the tool.
*   **Data Publishing:** Supports publishing results to Elasticsearch and InfluxDB for analysis and visualization.

## Getting Started

### Installation and Setup

1.  **Prerequisites:** Ensure you have accounts on your target cloud providers and have installed Python 3 (at least version 3.11) and pip.
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
5.  **Cloud Provider Specific Dependencies:** Install additional dependencies for each cloud provider. For example, AWS:
    ```bash
    $ cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/aws
    $ pip3 install -r requirements.txt
    ```

### Running a Single Benchmark

PerfKit Benchmarker can run benchmarks on Cloud Providers (GCP, AWS, Azure, DigitalOcean) and any "machine" you can SSH into.

*   **GCP Example:**
    ```bash
    ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
    ```
*   **AWS Example:**
    ```bash
    ./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro
    ```
*   **Azure Example:**
    ```bash
    ./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf
    ```
*   **Other Cloud Examples:** Example runs are available for IBMCloud, AliCloud, DigitalOcean, OpenStack, Kubernetes, Mesos, CloudStack, Rackspace, and ProfitBricks in the original documentation.

### Running All Standard Benchmarks

Run with `--benchmarks="standard_set"` to execute the standard suite of benchmarks serially (can take a couple of hours). If you don't specify `--cloud=...`, the benchmarks will run on the Google Cloud Platform.

### Running Benchmarks in a Named Set

Named sets are groupings of one or more benchmarks, maintained by the set owner (ex. GoogleSet is maintained by Google). Use the `--benchmarks="<set_name>"` command to run all benchmarks in a named set.

### Using Proxy configuration

If the VM guests do not have direct Internet access in the cloud environment,
you can configure proxy settings through `pkb.py` flags.

*   `--http_proxy`: Needed for package manager on Guest OS and for some Perfkit packages
*   `--https_proxy`: Needed for package manager or Ubuntu guest and for from Github downloaded packages
*   `--ftp_proxy`: Needed for some Perfkit packages

### Preprovisioned Data

Some benchmarks require preprovisioned data. This section describes how to preprovision this data.

#### Sample Preprovision Benchmark

This benchmark demonstrates the use of preprovisioned data. Create the following
file to upload using the command line:
```bash
echo "1234567890" > preprovisioned_data.txt
```
To upload, follow the instructions below with a filename of
`preprovisioned_data.txt` and a benchmark name of `sample`.

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

### Running Selective Stages of a Benchmark

Utilize this technique to run only certain stages of a benchmark (provision, prepare, run, teardown). This is useful for inspecting the machines after it has been prepared, but before the benchmark runs.

*   Run provision, prepare, and run stages of `cluster_boot`.
    ```
    ./pkb.py --benchmarks=cluster_boot --machine_type=n1-standard-2 --zones=us-central1-f --run_stage=provision,prepare,run
    ```
*   The output from the console will tell you the run URI for your benchmark. Try to ssh into the VM.
*   Now that you have examined the machines, teardown the instances that were made and cleanup.
    ```
    ./pkb.py --benchmarks=cluster_boot --run_stage=teardown -run_uri=<run_uri>
    ```

### Useful Global Flags

| Flag               | Notes                                                                                                                                                                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--helpmatch=pkb`  | Show all global flags.                                                                                                                                                                                                             |
| `--helpmatch=hpcc` | Show all flags associated with the hpcc benchmark.  You can substitute any benchmark name to see the associated flags.                                                                                                             |
| `--benchmarks`     | A comma-separated list of benchmarks or benchmark sets to run (e.g., `--benchmarks=iperf,ping`). Use `./pkb.py --helpmatch=benchmarks` to view the full list.                                                                      |
| `--cloud`          | The cloud provider where the benchmarks will be run (GCP, AWS, Azure, IBMCloud, AliCloud, DigitalOcean, OpenStack, CloudStack, Rackspace, Kubernetes, ProfitBricks).                                                               |
| `--machine_type`   | Specifies the machine type. Accepts provider-specific names or YAML expressions to match VM specs. This flag affects all machines; use YAML configurations for more precise control over machine types for different roles.     |
| `--zones`          | Overrides the default zone. Uses the same values as the cloud provider CLIs.                                                                                                                                                         |
| `--data_disk_type` | Specifies the type of data disk to use (e.g., pd-ssd, gp3, Premium_LRS).  `local` tells PKB to use the existing disk.                                                                                                               |

### Supported Benchmarks

The tool provides wrappers and workload definitions around various popular benchmark tools. Please ensure you agree to the licenses before using them.

*   `aerospike`
*   `bonnie++`
*   `cassandra_ycsb`
*   `cassandra_stress`
*   `cloudsuite3.0`
*   `cluster_boot`
*   `coremark`
*   `copy_throughput`
*   `fio`
*   `gpu_pcie_bandwidth`
*   `hadoop_terasort`
*   `hpcc`
*   `hpcg`
*   `iperf`
*   `memtier_benchmark`
*   `mesh_network`
*   `mongodb` **(Deprecated)**
*   `mongodb_ycsb`
*   `multichase`
*   `netperf`
*   `oldisim`
*   `object_storage_service`
*   `pgbench`
*   `ping`
*   `silo`
*   `scimark2`
*   `speccpu2006`
*   `SHOC`
*   `sysbench_oltp`
*   `TensorFlow`
*   `tomcat`
*   `unixbench`
*   `wrk`
*   `ycsb`

Some benchmarks require Java.

- `openjdk-7-jre`

[SPEC CPU2006](https://www.spec.org/cpu2006/) benchmark setup cannot be automated. Users must manually download and setup.

## Configuration and Configuration Overrides

*   **Benchmark Configuration:** Benchmarks have independent YAML configurations.
*   **User Configuration:** Override the default configurations using the `--benchmark_config_file` flag or the `--config_override` flag (e.g., `--config_override=cluster_boot.vm_groups.default.vm_count=100`).
*   **Static Machines**:  Configure to run benchmarks on pre-existing, non-provisioned machines using YAML config files.

## Advanced Features

### Running Benchmarks Without Cloud Provisioning

Run on a local machine or use a benchmark like iperf from an external point to a Cloud VM.

*   Ensure the static machine is SSH accessible with sudo access.
*   Create a YAML config file that describes how to connect to the machine.
*   Configure any number of benchmarks and reference the static VM.

### Specifying Flags in Configuration Files

Specify flag values within benchmark config files using the `flags` key. This allows setting default values for individual benchmarks.

### Data Publishing

*   **Elasticsearch Publisher:**  Publish results to an Elasticsearch server after installation.
    *   Install `elasticsearch` Python package (`pip install elasticsearch`).
    *   Use the `--es_uri`, `--es_index`, and `--es_type` flags.
*   **InfluxDB Publisher:**  Publish data to InfluxDB.
    *   Use the `--influx_uri` and `--influx_db_name` flags.

## Extending PerfKit Benchmarker

*   Refer to [CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md) for guidance.
*   Use comments in the code to add new benchmarks, package/OS type support, providers, etc.
*   Consult the [wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki) for documentation.

## Integration Testing

Run integration tests to create cloud resources by defining `PERFKIT_INTEGRATION` in the environment and running `tox -e integration`. This will fail if you have not installed and configured all of the relevant cloud provider SDKs.

## Planned Improvements

The project is continuously being improved.  Please submit feature requests via GitHub issues.