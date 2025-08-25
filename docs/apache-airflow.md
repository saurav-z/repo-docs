# Apache Airflow: Orchestrate Your Workflows with Code

**Apache Airflow** is a powerful, open-source platform that allows you to programmatically author, schedule, and monitor complex workflows. Automate your data pipelines and tasks with ease, allowing for greater maintainability, versioning, and collaboration. [Explore the original repository](https://github.com/apache/airflow).

## Key Features

*   **Dynamic Workflows:** Define pipelines as code, enabling dynamic DAG generation and parameterization.
*   **Extensible:** Benefit from a wide range of built-in operators and customize Airflow to fit your specific needs.
*   **Flexible Templating:** Leverage the Jinja templating engine for rich customizations.
*   **Rich User Interface:** Monitor pipeline progress, visualize dependencies, and troubleshoot issues through an intuitive web UI.
*   **Scalable:** Airflow is designed to handle complex workflows at scale, with support for various executors.

## Getting Started

Get up and running with Apache Airflow quickly:

*   **Installation:** Detailed instructions can be found in the [INSTALLING.md](INSTALLING.md) file.
*   **Documentation:** Explore the latest stable documentation for [installation](https://airflow.apache.org/docs/apache-airflow/stable/installation/), [getting started](https://airflow.apache.org/docs/apache-airflow/stable/start.html), and [tutorials](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/).

## Installing from PyPI

Install Apache Airflow using pip:

```bash
pip install 'apache-airflow==3.0.5' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.5/constraints-3.10.txt"
```

Install with extras (i.e., postgres, google):

```bash
pip install 'apache-airflow[postgres,google]==3.0.5' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.5/constraints-3.10.txt"
```

## User Interface

Explore the intuitive Airflow UI:

*   **DAGs:** Overview of all DAGs
*   **Assets:** Overview of Assets with dependencies
*   **Grid:** Grid representation of a DAG that spans across time.
*   **Graph:** Visualization of a DAG's dependencies
*   **Home:** Summary statistics
*   **Backfill:** Backfilling a DAG for a specific date range
*   **Code:** Quick way to view source code of a DAG

## Versioning

Apache Airflow follows a [strict SemVer](https://semver.org/) approach as of version 2.0.0.

## Support

*   **Python Versions:** Supports multiple Python versions (3.9, 3.10, 3.11, 3.12, 3.13).
*   **Kubernetes:** Compatible with Kubernetes versions 1.30, 1.31, 1.32, 1.33
*   **Databases:** Supports various databases.

**See the full version and support details in the README.**

## Contribute

Contribute to Apache Airflow: Check out the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)

## Sponsors

*   <a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
*   <a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>