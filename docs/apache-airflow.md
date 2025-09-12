# Apache Airflow: Orchestrate, Schedule, and Monitor Workflows at Scale

**Apache Airflow** is a powerful platform for programmatically authoring, scheduling, and monitoring workflows, enabling data engineers and data scientists to streamline their data pipelines.  [Explore Airflow on GitHub](https://github.com/apache/airflow).

**Key Features:**

*   **Workflow as Code:** Define your data pipelines using Python, making them more maintainable, versionable, testable, and collaborative.
*   **Dynamic DAG Generation:**  Create workflows (DAGs) with dynamic task generation and parameterization using the Jinja templating engine.
*   **Extensible Architecture:** Easily extend Airflow with a wide range of built-in operators and customize it to fit your specific needs.
*   **Robust Scheduling & Execution:** The Airflow scheduler executes tasks on a distributed worker pool while respecting specified dependencies.
*   **Rich User Interface:** Visualize pipelines, monitor progress, and troubleshoot issues with an intuitive web interface.

**Why Use Airflow?**

Airflow is the perfect tool for:

*   Data pipeline orchestration
*   ETL (Extract, Transform, Load) workflows
*   Machine learning model training and deployment
*   Automating data-related tasks

## Core Principles

Airflow is designed with the following principles in mind:

*   **Dynamic:** Pipelines are defined in code, enabling dynamic dag generation and parameterization.
*   **Extensible:** The Airflow framework includes a wide range of built-in operators and can be extended to fit your needs.
*   **Flexible**: Airflow leverages the [**Jinja**](https://jinja.palletsprojects.com) templating engine, allowing rich customizations.

## Requirements

See [Requirements](#requirements) to check your environment configuration.

## Getting Started

To begin using Airflow, consult the official documentation for:

*   [Installing Airflow](https://airflow.apache.org/docs/apache-airflow/stable/installation/)
*   [Getting Started](https://airflow.apache.org/docs/apache-airflow/stable/start.html)
*   [Tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/)

## Installing from PyPI

Follow the instructions in [Installing from PyPI](#installing-from-pypi) to install Airflow.

## Installation

For a detailed guide on setting up your local development environment, refer to the [INSTALLING.md](INSTALLING.md) file.

## Official Source Code

Apache Airflow is an [Apache Software Foundation](https://www.apache.org) (ASF) project.  Our official releases are governed by the [ASF Release Policy](https://www.apache.org/legal/release-policy.html) and can be downloaded from [the ASF Distribution Directory](https://downloads.apache.org/airflow).

## Convenience Packages

Learn about alternative installation methods:
* [PyPI releases](https://pypi.org/project/apache-airflow/)
* [Docker Images](https://hub.docker.com/r/apache/airflow)
* [Tags in GitHub](https://github.com/apache/airflow/tags)

## User Interface

*   **DAGs**: Overview of all DAGs in your environment.

  ![DAGs](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/dags.png)

*   **Assets**: Overview of Assets with dependencies.

  ![Asset Dependencies](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/assets_graph.png)

*   **Grid**: Grid representation of a DAG that spans across time.

  ![Grid](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/grid.png)

*   **Graph**: Visualization of a DAG's dependencies and their current status for a specific run.

  ![Graph](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/graph.png)

*   **Home**: Summary statistics of your Airflow environment.

  ![Home](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/home.png)

*   **Backfill**: Backfilling a DAG for a specific date range.

  ![Backfill](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/backfill.png)

*   **Code**: Quick way to view source code of a DAG.

  ![Code](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/code.png)

## Semantic Versioning

Airflow follows a strict [SemVer](https://semver.org/) approach.

## Version Life Cycle

Check the [Version Life Cycle](#version-life-cycle) to understand how the Airflow versions are managed.

## Support for Python and Kubernetes versions

See [Support for Python and Kubernetes versions](#support-for-python-and-kubernetes-versions) for the latest version policies.

## Base OS support for reference Airflow images

Airflow community provides ready to use container images that are published whenever we publish an Apache Airflow release. Those images contain:

* Base OS with necessary packages to install Airflow (stable Debian OS)
* Base Python installation in versions supported at the time of release for the MINOR version of
  Airflow released (so there could be different versions for 2.3 and 2.2 line for example)
* Libraries required to connect to supported Databases (again the set of databases supported depends
  on the MINOR version of Airflow)
* Predefined set of popular providers (for details see the [Dockerfile](https://raw.githubusercontent.com/apache/airflow/main/Dockerfile)).
* Possibility of building your own, custom image where the user can choose their own set of providers
  and libraries (see [Building the image](https://airflow.apache.org/docs/docker-stack/build.html))
* In the future Airflow might also support a "slim" version without providers nor database clients installed

## Approach to dependencies of Airflow

Airflow dependencies are managed in `pyproject.toml` and `provider.yaml`. [Approach for dependencies of Airflow](#approach-to-dependencies-of-airflow) explains in detail how it works.

## Contributing

Contribute to Apache Airflow! Read the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst) for details on how to help build Airflow.

## Voting Policy

See the [Voting Policy](#voting-policy)

## Who uses Apache Airflow?

Apache Airflow is used by thousands of organizations worldwide. [Who uses Apache Airflow?](#who-uses-apache-airflow) lists all the organizations known that use Apache Airflow.

## Who maintains Apache Airflow?

Apache Airflow is maintained by a dedicated [community](https://github.com/apache/airflow/graphs/contributors) of committers/maintainers.

## What goes into the next release?

Check the [What goes into the next release](#what-goes-into-the-next-release) to get more details.

## Can I use the Apache Airflow logo in my presentation?

Yes! Be sure to abide by the Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook). The most up-to-date logos are found in [this repo](https://github.com/apache/airflow/tree/main/airflow-core/docs/img/logos/) and on the Apache Software Foundation [website](https://www.apache.org/logos/about.html).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>