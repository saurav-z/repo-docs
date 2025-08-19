# Apache Airflow: Orchestrate, Schedule, and Monitor Workflows

**Effortlessly automate, monitor, and manage your data pipelines with Apache Airflow – the leading open-source workflow orchestration platform.**  [Explore the Apache Airflow GitHub repository](https://github.com/apache/airflow) for the latest updates.

[![License](https://img.shields.io/:license-Apache%202-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0.txt)
[![PyPI version](https://badge.fury.io/py/apache-airflow.svg)](https://badge.fury.io/py/apache-airflow)
[![Docker Pulls](https://img.shields.io/docker/pulls/apache/airflow.svg)](https://hub.docker.com/r/apache/airflow)
[![Contributors](https://img.shields.io/github/contributors/apache/airflow)](https://github.com/apache/airflow/graphs/contributors)

<picture width="500">
  <img
    src="https://github.com/apache/airflow/blob/19ebcac2395ef9a6b6ded3a2faa29dc960c1e635/docs/apache-airflow/img/logos/wordmark_1.png?raw=true"
    alt="Apache Airflow logo"
  />
</picture>

## Key Features

*   **Dynamic Workflow Definition:**  Define workflows as code for maintainability, versioning, testing, and collaboration.
*   **Extensible Architecture:** Leverage a wide range of built-in operators and easily extend Airflow to meet specific needs.
*   **Flexible Templating:** Customize workflows with the Jinja templating engine.
*   **Robust Scheduling and Execution:** Schedule tasks with dependencies, and execute them on a distributed worker pool.
*   **Comprehensive Monitoring & UI:**  Visualize pipeline progress, monitor performance, and troubleshoot issues through a rich user interface.

## Core Principles

*   **Dynamic:** Define pipelines using code, enabling dynamic DAG generation and parameterization.
*   **Extensible:** Extend the Airflow framework to fit your needs with a wide range of built-in operators.
*   **Flexible:** Customize workflows leveraging the Jinja templating engine.

## Requirements

*   **Python:** 3.9, 3.10, 3.11, 3.12. (3.13 supported on the main branch)
*   **Platform:** AMD64/ARM64 (\*Experimental)
*   **Kubernetes:** 1.30, 1.31, 1.32, 1.33. (1.30, 1.31, 1.32, 1.33 for stable)
*   **Database:** PostgreSQL, MySQL, SQLite (for testing)
*   **OS:** POSIX-compliant Operating Systems. For development, tested on Linux Distros and macOS. Windows support via WSL2

## Getting Started

Begin your Airflow journey with the official documentation for detailed [installation](https://airflow.apache.org/docs/apache-airflow/stable/installation/), [quickstart](https://airflow.apache.org/docs/apache-airflow/stable/start.html), and [tutorials](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/).

## Installing from PyPI

Install Airflow using `pip`, utilizing constraint files for reproducible installations. Example:

```bash
pip install 'apache-airflow==3.0.4' --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.4/constraints-3.10.txt"
```
Use extras to add additional dependencies
```bash
pip install 'apache-airflow[postgres,google]==3.0.4' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.4/constraints-3.10.txt"
```

## Installation

For detailed setup and installation instructions, consult the [INSTALLING.md](INSTALLING.md) file.

## Official Source Code

Apache Airflow is an Apache Software Foundation (ASF) project with official source code releases available from the [ASF Distribution Directory](https://downloads.apache.org/airflow).

## User Interface

The Airflow UI provides comprehensive tools for managing your workflows, including:

*   DAGs Overview
*   Assets Overview
*   Grid Representation
*   Graph Visualization
*   Home Dashboard
*   Backfill Functionality
*   Code Viewing

## Semantic Versioning

Airflow follows a strict [SemVer](https://semver.org/) approach starting with version 2.0.0.

## Version Life Cycle

See the table below for the version support life cycle.

| Version   | Current Patch/Minor   | State     | First Release   | Limited Maintenance   | EOL/Terminated   |
|-----------|-----------------------|-----------|-----------------|-----------------------|------------------|
| 3         | 3.0.4                 | Supported | Apr 22, 2025    | TBD                   | TBD              |
| 2         | 2.11.0                | Supported | Dec 17, 2020    | Oct 22, 2025          | Apr 22, 2026     |

## Contributing

Contribute to Apache Airflow by following the guidelines in the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst).

## Who uses Apache Airflow?

Airflow is used by over 500 organizations.  See the [INTHEWILD.md](https://github.com/apache/airflow/blob/main/INTHEWILD.md) for a list.

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

Apache Airflow CI infrastructure is sponsored by:

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>