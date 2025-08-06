# PerfKit Benchmarker: The Open-Source Standard for Cloud Performance Evaluation

**Measure and compare cloud offerings with ease using PerfKit Benchmarker, an open-source tool for defining and running standardized benchmarks.** [Explore the PerfKit Benchmarker repository](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker).

## Key Features:

*   **Automated Benchmarking:** Easily run benchmarks on various cloud providers (GCP, AWS, Azure, etc.) with automatic VM instantiation and benchmark installation.
*   **Standardized Benchmarks:**  Utilizes a canonical set of benchmarks for consistent and comparable performance measurements.
*   **Customizable Configurations:** Offers flexible configuration options, including YAML-based configurations and overrides, to tailor benchmarks to specific needs.
*   **Extensive Cloud Support:** Supports a wide range of cloud providers and platforms including Kubernetes, OpenStack and Mesos.
*   **Flexible Deployment:** Supports running benchmarks on cloud providers, local machines, and pre-provisioned infrastructure.
*   **Advanced Features:** Includes support for pre-provisioned data, proxy configurations, and data publishing to Elasticsearch and InfluxDB.
*   **Extensible:** Easily add new benchmarks, providers, and features through well-documented code and a supportive community.

## Getting Started

### Installation & Setup

1.  **Python 3 & Virtual Environment:**

    ```bash
    python3 -m venv $HOME/my_virtualenv
    source $HOME/my_virtualenv/bin/activate
    ```

2.  **Install PerfKit Benchmarker:**

    ```bash
    cd $HOME
    git clone https://github.com/GoogleCloudPlatform/PerfKitBenchmarker.git
    pip3 install -r $HOME/PerfKitBenchmarker/requirements.txt
    ```
    Install provider-specific dependencies as needed (e.g., AWS:
    `pip3 install -r $HOME/PerfKitBenchmarker/perfkitbenchmarker/providers/aws/requirements.txt`)

### Running a Single Benchmark

```bash
./pkb.py --project=<GCP project ID> --benchmarks=iperf --machine_type=f1-micro
```
(See README for more examples)

## License

*   PerfKit Benchmarker uses wrappers and workload definitions around popular benchmark tools.
*   You are responsible for accepting the licenses of each benchmark before using PerfKit Benchmarker.
*   Please run PKB with the explicit flag `--accept-licenses`

## Benchmarks Included
PerfKit Benchmarker provides wrappers and workload definitions around popular benchmark tools. 
-   Aerospike
-   Bonnie++
-   Cassandra YCSB
-   Cloudsuite 3.0
-   Cluster Boot
-   Coremark
-   Copy Throughput
-   Fio
-   GPU PCIE Bandwidth
-   Hadoop Terasort
-   HPCC
-   HPCG
-   Iperf
-   Memtier Benchmark
-   Mesh Network
-   Netperf
-   Oldisim
-   Object Storage Service
-   Pgbench
-   Ping
-   Silo
-   Scimark2
-   Speccpu2006
-   SHOC
-   Sysbench OLTP
-   TensorFlow
-   Tomcat
-   Unixbench
-   Wrk
-   YCSB

## Further Reading

*   **[Wiki](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/wiki)**: Comprehensive documentation, FAQs, and design documents.
*   **[Contributing Guidelines](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/blob/master/CONTRIBUTING.md)**: Learn how to contribute to the project.
*   **[Issues](https://github.com/GoogleCloudPlatform/PerfKitBenchmarker/issues)**:  Report issues, suggest improvements, and engage with the community.