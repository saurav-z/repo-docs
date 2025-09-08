# PerfKit Benchmarker: The Open Source Standard for Cloud Performance Testing

**PerfKit Benchmarker (PKB) is your go-to open-source tool for consistently measuring and comparing cloud offerings.  Get started benchmarking your cloud environments today!** ([Original Repo](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker))

## Key Features

*   **Automated Benchmarking:** Easily deploy VMs on various cloud providers, automatically install benchmarks, and execute workloads without user interaction.
*   **Vendor-Agnostic:** Designed to work across all major cloud providers (GCP, AWS, Azure, DigitalOcean, OpenStack, etc.) using vendor-provided command-line tools.
*   **Standardized Benchmarks:** Provides a canonical set of benchmarks with default settings to ensure consistency across services.
*   **Flexible Configuration:** Supports YAML-based configurations for complex setups, including custom machine types, zones, and cloud-spanning benchmarks.
*   **Extensive Benchmark Library:** Includes a wide range of popular benchmarks, including iperf, fio, Hadoop Terasort, and many more.  See full list below.
*   **Data Visualization and Publishing:** Integrates with Elasticsearch and InfluxDB for robust result analysis and reporting.
*   **Extensible and Customizable:**  Easily extend PKB by adding new benchmarks, cloud providers, or operating system support.
*   **Integration Testing:** Includes integration tests to ensure consistent performance.

## Getting Started

### Installation and Setup

1.  **Prerequisites:**

    *   A valid account with your chosen cloud provider(s).
    *   Python 3 (at least version 3.11) and `pip`.

2.  **Virtual Environment (Recommended):**

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

    *   Install cloud provider-specific requirements as needed (e.g., for AWS: `cd $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/aws && pip3 install -r requirements.txt`).

### Running a Single Benchmark

Use the following example commands to run a simple iperf benchmark, replacing `<GCP project ID>` with your project ID and adjusting cloud provider and machine type as needed:

*Example GCP*
```bash
./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
```

*Example AWS*
```bash
$ ./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro
```
*Example Azure*
```bash
$ ./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf
```
### Example Run on Other Clouds (IBMCloud, AliCloud, DigitalOcean, OpenStack, Kubernetes, Mesos, CloudStack, Rackspace, ProfitBricks):

```bash
# Example Run on IBMCloud
$ ./pkb.py --cloud=IBMCloud --machine_type=cx2-4x8 --benchmarks=iperf
```

```bash
# Example Run on AliCloud
$ ./pkb.py --cloud=AliCloud --machine_type=ecs.s2.large --benchmarks=iperf
```

```bash
# Example Run on DigitalOcean
$ ./pkb.py --cloud=DigitalOcean --machine_type=16gb --benchmarks=iperf
```

```bash
# Example Run on OpenStack
$ ./pkb.py --cloud=OpenStack --machine_type=m1.medium \
           --openstack_network=private --benchmarks=iperf
```

```bash
# Example Run on Kubernetes
$ ./pkb.py --vm_platform=Kubernetes --benchmarks=iperf \
           --kubeconfig=/path/to/kubeconfig --use_k8s_vm_node_selectors=False
```

```bash
# Example Run on Mesos
$ ./pkb.py --cloud=Mesos --benchmarks=iperf --marathon_address=localhost:8080
```

```bash
# Example Run on CloudStack
./pkb.py --cloud=CloudStack --benchmarks=ping --cs_network_offering=DefaultNetworkOffering
```

```bash
# Example Run on Rackspace
$ ./pkb.py --cloud=Rackspace --machine_type=general1-2 --benchmarks=iperf
```

```bash
# Example Run on ProfitBricks
$ ./pkb.py --cloud=ProfitBricks --machine_type=Small --benchmarks=iperf
```
### Running Windows Benchmarks

*   Install dependencies
*   Run with `--os_type=windows`

### Running Benchmarks with Juju

*   Run with `--os_type=juju`.

### Running All Standard Benchmarks

*   Run with `--benchmarks="standard_set"`.

### Running Selective Stages of a Benchmark
*   Use the `--run_stage` flag to run specific stages of the benchmark process (provision, prepare, run, teardown).

### Running Benchmarks without Cloud Provisioning (Static VMs)

*   Create a YAML configuration file with your static VM's IP, credentials, and disk specifications.
*   Use the `--benchmark_config_file` flag to specify the configuration.

### How to Extend PerfKit Benchmarker

*   Refer to the `CONTRIBUTING.md` file for contribution guidelines.
*   Add new benchmarks, package/OS type support, or cloud providers.
*   Provide and request documentation via the [wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki) and [issues](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues).

### Integration Testing
*   To run unit or integration tests, ensure that you have `tox >= 2.0.0` installed.

## Benchmark Licensing

**Before running PerfKit Benchmarker, you must accept the licenses of the individual benchmarks.** Run PKB with the explicit flag `--accept-licenses`.

The following benchmarks are executed:

-   `aerospike`: [Apache v2 for the client](http://www.aerospike.com/aerospike-licensing/) and [GNU AGPL v3.0 for the server](https://github.com/aerospike/aerospike-server/blob/master/LICENSE)
-   `bonnie++`: [GPL v2](http://www.coker.com.au/bonnie++/readme.html)
-   `cassandra_ycsb`: [Apache v2](http://cassandra.apache.org/)
-   `cassandra_stress`: [Apache v2](http://cassandra.apache.org/)
-   `cloudsuite3.0`: [CloudSuite 3.0 license](http://cloudsuite.ch/pages/license/)
-   `cluster_boot`: MIT License
-   `coremark`: [EEMBC](https://www.eembc.org/)
-   `copy_throughput`: Apache v2
-   `fio`: [GPL v2](https://github.com/axboe/fio/blob/master/COPYING)
-   [`gpu_pcie_bandwidth`](https://developer.nvidia.com/cuda-downloads): [NVIDIA Software Licence Agreement](http://docs.nvidia.com/cuda/eula/index.html#nvidia-driver-license)
-   `hadoop_terasort`: [Apache v2](http://hadoop.apache.org/)
-   `hpcc`: [Original BSD license](http://icl.cs.utk.edu/hpcc/faq/#263)
-   [`hpcg`](https://github.com/hpcg-benchmark/hpcg/): [BSD 3-clause](https://github.com/hpcg-benchmark/hpcg/blob/master/LICENSE)
-   `iperf`: [UIUC License](https://sourceforge.net/p/iperf2/code/ci/master/tree/doc/ui_license.html)
-   `memtier_benchmark`: [GPL v2](https://github.com/RedisLabs/memtier_benchmark)
-   `mesh_network`: [HP license](http://www.calculate-linux.org/packages/licenses/netperf)
-   `mongodb`: **Deprecated**. [GNU AGPL v3.0](http://www.mongodb.org/about/licensing/)
-   `mongodb_ycsb`: [GNU AGPL v3.0](http://www.mongodb.org/about/licensing/)
-   [`multichase`](https://github.com/google/multichase): [Apache v2](https://github.com/google/multichase/blob/master/LICENSE)
-   `netperf`: [HP license](http://www.calculate-linux.org/packages/licenses/netperf)
-   [`oldisim`](https://github.com/GoogleCloudPlatform/oldisim): [Apache v2](https://github.com/GoogleCloudPlatform/oldisim/blob/master/LICENSE.txt)
-   `object_storage_service`: Apache v2
-   `pgbench`: [PostgreSQL Licence](https://www.postgresql.org/about/licence/)
-   `ping`: No license needed.
-   `silo`: MIT License
-   `scimark2`: [public domain](http://math.nist.gov/scimark2/credits.html)
-   `speccpu2006`: [SPEC CPU2006](http://www.spec.org/cpu2006/)
-   [`SHOC`](https://github.com/vetter/shoc): [BSD 3-clause](https://github.com/vetter/shoc/blob/master/LICENSE.md)
-   `sysbench_oltp`: [GPL v2](https://github.com/akopytov/sysbench)
-   [`TensorFlow`](https://github.com/tensorflow/tensorflow): [Apache v2](https://github.com/tensorflow/tensorflow/blob/master/LICENSE)
-   [`tomcat`](https://github.com/apache/tomcat): [Apache v2](https://github.com/apache/tomcat/blob/trunk/LICENSE)
-   [`unixbench`](https://github.com/kdlucas/byte-unixbench): [GPL v2](https://github.com/kdlucas/byte-unixbench/blob/master/LICENSE.txt)
-   [`wrk`](https://github.com/wg/wrk): [Modified Apache v2](https://github.com/wg/wrk/blob/master/LICENSE)
-   [`ycsb`](https://github.com/brianfrankcooper/YCSB) (used by `mongodb`, `hbase_ycsb`, and others): [Apache v2](https://github.com/brianfrankcooper/YCSB/blob/master/LICENSE.txt)

## Useful Global Flags

| Flag               | Description                                                                                                           |
| ------------------ | --------------------------------------------------------------------------------------------------------------------- |
| `--helpmatch=pkb`  | Display all global flags.                                                                                              |
| `--helpmatch=hpcc` | Display all flags associated with the hpcc benchmark.  Replace `hpcc` with any benchmark name to see its specific flags. |
| `--benchmarks`     | Comma-separated list of benchmarks or benchmark sets to run (e.g., `--benchmarks=iperf,ping`).                          |
| `--cloud`          | Cloud provider to run benchmarks on (e.g., GCP, AWS, Azure, IBMCloud, AliCloud, DigitalOcean, OpenStack, etc.).        |
| `--machine_type`   | Specifies the machine type for VMs.                                                                                    |
| `--zones`          | Specifies the zone for VMs.                                                                                            |
| `--data_disk_type` | Specifies the type of data disk to use.                                                                                |

## Preprovisioned Data

Some benchmarks require preprovisioned data.

### Clouds with Preprovisioned Data

1.  **Google Cloud:**
    *   Upload files to Google Cloud Storage using `gsutil`.
    *   Use the `--gcp_preprovisioned_data_bucket=<bucket>` flag.

2.  **AWS:**
    *   Upload files to S3 using the AWS CLI.
    *   Use the `--aws_preprovisioned_data_bucket=<bucket>` flag.

## Configurations and Configuration Overrides
*   Use YAML files to define benchmark configurations and override defaults using `--benchmark_config_file`.
*   Use the `--config_override` flag to specify single settings.

## Elasticsearch and InfluxDB Publishers

*   Install the `elasticsearch` Python package to use Elasticsearch.
*   Set `--es_uri`, `--es_index`, and `--es_type` for Elasticsearch publishing.
*   Set `--influx_uri` and `--influx_db_name` for InfluxDB publishing.

## Planned Improvements

*   Check the GitHub issues for the planned improvements.  Feel free to submit new requests!