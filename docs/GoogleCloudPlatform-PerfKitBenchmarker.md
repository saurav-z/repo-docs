# PerfKit Benchmarker: The Open Source Standard for Cloud Performance Measurement

**Are you comparing cloud offerings and need a reliable, automated way to benchmark them?** [PerfKit Benchmarker](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker) is an open-source tool designed to consistently measure and compare the performance of various cloud providers using their command-line tools.

## Key Features:

*   **Vendor-Agnostic Benchmarking:** Designed to work across different cloud providers (GCP, AWS, Azure, DigitalOcean, OpenStack, IBMCloud, AliCloud, CloudStack, Rackspace, Kubernetes, ProfitBricks, and Mesos) with a focus on consistent results.
*   **Automated Execution:** Automates the process of provisioning VMs, installing benchmarks, and running workloads, minimizing user interaction and ensuring reproducibility.
*   **Extensive Benchmark Library:** Includes wrappers and workload definitions for a wide range of popular benchmarks, including iperf, fio, Hadoop Terasort, and many more (see list below).
*   **Flexible Configuration:** Supports YAML-based configurations for complex setups, including multi-cloud benchmarking and customized VM specifications.
*   **Comprehensive Reporting:** Integrates with Elasticsearch and InfluxDB for advanced data analysis and visualization.
*   **Open Source & Community Driven:** Built with contributions from the community and welcomes new benchmarks and provider support.
*   **Extensible:** Provides detailed documentation, including the
    [wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki) and code comments, to help you add new benchmarks, OS types, and providers.
*   **Integration Testing:** Integrates with `tox >= 2.0.0` and `PERFKIT_INTEGRATION` to run unit and integration tests, which create actual cloud resources.

## Getting Started

1.  **Prerequisites:** Ensure you have Python 3 and the necessary cloud provider accounts (see [providers](perfkitbenchmarker/providers/README.md)).
2.  **Installation:**
    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    cd $HOME
    git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    cd $HOME/PerfKitBenchmarker
    pip3 install -r requirements.txt
    ```
3.  **Run a Benchmark:**
    ```bash
    ./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
    ```
    (Adjust flags for other cloud providers and benchmarks; see examples below).

## Example Runs

**GCP:**

```bash
./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
```

**AWS:**

```bash
./pkb.py --cloud=AWS --benchmarks=iperf --machine_type=t2.micro
```

**Azure:**

```bash
./pkb.py --cloud=Azure --machine_type=Standard_A0 --benchmarks=iperf
```

**(See full examples in the original README, including OpenStack, Kubernetes, etc.)**

## Available Benchmarks:

*   aerospike
*   bonnie++
*   cassandra_ycsb
*   cassandra_stress
*   cloudsuite3.0
*   cluster_boot
*   coremark
*   copy_throughput
*   fio
*   gpu_pcie_bandwidth
*   hadoop_terasort
*   hpcc
*   hpcg
*   iperf
*   memtier_benchmark
*   mesh_network
*   mongodb  (Deprecated)
*   mongodb_ycsb
*   multichase
*   netperf
*   oldisim
*   object_storage_service
*   pgbench
*   ping
*   silo
*   scimark2
*   speccpu2006
*   SHOC
*   sysbench_oltp
*   TensorFlow
*   tomcat
*   unixbench
*   wrk
*   ycsb

(Note:  Specific licenses apply to each benchmark; you are responsible for reviewing and accepting licenses before use.  See the original README for details)

## Additional Information

*   **YAML Configuration:** [Configurations and Configuration Overrides](#configurations-and-configuration-overrides)
*   **Preprovisioned Data:** [Preprovisioned Data](#preprovisioned-data)
*   **Advanced Usage:** [Advanced: How To Run Benchmarks Without Cloud Provisioning (e.g., local workstation)](#advanced-how-to-run-benchmarks-without-cloud-provisioning-e.g.-local-workstation)
*   **Extending PerfKit Benchmarker:** [How to Extend PerfKit Benchmarker](#how-to-extend-perfkit-benchmarker)
*   **Useful Flags:** [Useful Global Flags](#useful-global-flags)

**Find the complete documentation and contribute on the [GitHub repository](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker).**