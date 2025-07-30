# Apache Airflow: The Open-Source Platform for Workflow Orchestration

**Automate, schedule, and monitor your workflows with Apache Airflow, the leading platform for programmatic data pipelines.**

[![License](https://img.shields.io/:license-Apache%202-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0.txt)
[![PyPI version](https://badge.fury.io/py/apache-airflow.svg)](https://badge.fury.io/py/apache-airflow)
[![Docker Pulls](https://img.shields.io/docker/pulls/apache/airflow.svg)](https://hub.docker.com/r/apache/airflow)
[![GitHub Build main](https://github.com/apache/airflow/actions/workflows/ci-amd.yml/badge.svg)](https://github.com/apache/airflow/actions)
<picture width="500">
  <img
    src="https://github.com/apache/airflow/blob/19ebcac2395ef9a6b6ded3a2faa29dc960c1e635/docs/apache-airflow/img/logos/wordmark_1.png?raw=true"
    alt="Apache Airflow logo"
  />
</picture>

[Apache Airflow](https://airflow.apache.org/docs/apache-airflow/stable/) is an open-source platform designed to programmatically author, schedule, and monitor workflows.  It's a powerful tool for creating and managing complex data pipelines.  Visit the [original repo](https://github.com/apache/airflow) for more details.

**Key Features:**

*   **Dynamic Pipelines:** Define workflows as code, enabling dynamic DAG generation and parameterization for flexibility.
*   **Extensible:** Built-in operators and easy extension to integrate with various services and tools.
*   **Flexible:** Leverage Jinja templating for rich customizations and control over your workflows.
*   **Scalable:** Execute tasks on a distributed set of workers, allowing for efficient scaling.
*   **UI:** Rich user interface for easy pipeline visualization, monitoring, and troubleshooting.
*   **Idempotent Tasks:** Airflow emphasizes tasks that produce the same results on subsequent runs to eliminate duplicates.
*   **Not a Streaming Solution**: Airflow is not designed for stream processing, but for batch processing of real-time data streams.

**Learn More:**

*   **Official Documentation:** [https://airflow.apache.org/docs/apache-airflow/stable/](https://airflow.apache.org/docs/apache-airflow/stable/)
*   **Getting Started:** [https://airflow.apache.org/docs/apache-airflow/stable/start.html](https://airflow.apache.org/docs/apache-airflow/stable/start.html)
*   **Installation:** Refer to the [INSTALLING.md](INSTALLING.md) file for comprehensive instructions.

**Sections below are organized as in original README but heavily summarized:**

## Requirements

*   Lists supported Python and Database versions and platforms.

## Getting Started

*   Links to official Airflow website documentation for installation, getting started, and tutorials.
*   Links to documentation for main branch and improvement proposals.

## Installing from PyPI

*   Guidance on installing using `pip` with constraint files.
*   Explains extras for installing with providers.

## Installation

*   For detailed information on setting up your development environment and installing Apache Airflow, refer to the [INSTALLING.md](INSTALLING.md) file.

## Official source code

*   Airflow is an Apache Software Foundation (ASF) project.
*   Source code releases follow ASF release policies and are cryptographically signed.

## Convenience packages

*   Describes PyPI releases, Docker Images, and GitHub Tags as alternative methods.

## User Interface

*   Demonstrates the Airflow UI with screenshots of DAGs, Assets, Grid, Graph, Home, Backfill, and Code views.

## Semantic versioning

*   Explanation of SemVer approach for releases, core packages, providers, and API clients.

## Version Life Cycle

*   Table with versioning, states, and EOL dates.

## Support for Python and Kubernetes versions

*   Airflow policies on dropping support for Python and Kubernetes.

## Base OS support for reference Airflow images

*   Details the operating system and components included in the container images provided by the Airflow community.

## Approach to dependencies of Airflow

*   Airflow's approach to managing dependencies using constraints and upper-bounding strategies, including reasons for specific upper bounds.

## Contributing

*   Information on how to contribute.

## Voting Policy

*   Details on the voting policies for Apache Airflow.

## Who uses Apache Airflow?

*   Provides a link to a list of organizations using Airflow.

## Who maintains Apache Airflow?

*   Information about the Airflow community and maintainers.

## What goes into the next release?

*   Overview of how features and bug fixes are released.

## Can I use the Apache Airflow logo in my presentation?

*   Guidelines for using the Apache Airflow logo.

## Links

*   Links to Documentation, Chat, and Community information.

## Sponsors

*   Acknowledgements of sponsors and links to their websites.