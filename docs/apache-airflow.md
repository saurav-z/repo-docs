# Apache Airflow: Orchestrate Your Workflows with Code

**Apache Airflow is a powerful platform for programmatically authoring, scheduling, and monitoring workflows, making complex data pipelines manageable and efficient.**  [Explore the Apache Airflow Repository](https://github.com/apache/airflow)

[![License](https://img.shields.io/:license-Apache%202-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0.txt)
[![PyPI version](https://badge.fury.io/py/apache-airflow.svg)](https://badge.fury.io/py/apache-airflow)
[![Docker Pulls](https://img.shields.io/docker/pulls/apache/airflow.svg)](https://hub.docker.com/r/apache/airflow)
[![Slack Status](https://img.shields.io/badge/slack-join_chat-white.svg?logo=slack&style=social)](https://s.apache.org/airflow-slack)

<picture width="500">
  <img
    src="https://github.com/apache/airflow/blob/19ebcac2395ef9a6b6ded3a2faa29dc960c1e635/docs/apache-airflow/img/logos/wordmark_1.png?raw=true"
    alt="Apache Airflow logo"
  />
</picture>

**Key Features:**

*   **Dynamic Workflows:** Define pipelines in code (Python), enabling dynamic DAG generation and parameterization.
*   **Extensible Architecture:** Built-in operators and easy customization to suit your needs.
*   **Flexible Templating:** Leverage Jinja templating for rich customizations and control.
*   **Scalable Scheduling:** Airflow's scheduler executes tasks on a distributed worker pool.
*   **Robust Monitoring:** A rich user interface provides visualization, monitoring, and troubleshooting capabilities.
*   **Idempotent Tasks:** Best practice to ensure that tasks can be safely rerun.
*   **Rich UI:** Easy to use UI for visualizing and monitoring pipelines.

## Getting Started

Explore the comprehensive [Airflow documentation](https://airflow.apache.org/docs/apache-airflow/stable/) for detailed installation guides, tutorials, and how-to guides.

## Installation

For detailed information about getting started with Apache Airflow and setting up your local development environment, please refer to the [INSTALLING.md](INSTALLING.md) file.

## Installing from PyPI

Install the core Airflow package with:

```bash
pip install 'apache-airflow==3.0.2' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.2/constraints-3.10.txt"
```

Install with extras (example: postgres, google):

```bash
pip install 'apache-airflow[postgres,google]==3.0.2' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.2/constraints-3.10.txt"
```

For installation information on provider distributions, check the [providers](http://airflow.apache.org/docs/apache-airflow-providers/index.html).

## Official Source Code

Airflow is an Apache Software Foundation (ASF) project with official releases from the [ASF Distribution Directory](https://downloads.apache.org/airflow).

## User Interface

*   **DAGs:** Overview of all DAGs in your environment.

    ![DAGs](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/dags.png)

*   **Assets**: Overview of Assets with dependencies.

    ![Asset Dependencies](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/assets_graph.png)

*   **Grid:** Grid representation of a DAG that spans across time.

    ![Grid](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/grid.png)

*   **Graph:** Visualization of a DAG's dependencies and their current status for a specific run.

    ![Graph](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/graph.png)

*   **Home:** Summary statistics of your Airflow environment.

    ![Home](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/home.png)

*   **Backfill:** Backfilling a DAG for a specific date range.

    ![Backfill](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/backfill.png)

*   **Code:** Quick way to view source code of a DAG.

    ![Code](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/code.png)

## Semantic Versioning

Airflow adheres to strict [SemVer](https://semver.org/) versioning.

## Version Life Cycle

Apache Airflow version life cycle:

<!-- This table is automatically updated by pre-commit scripts/ci/pre_commit/supported_versions.py -->
<!-- Beginning of auto-generated table -->

| Version   | Current Patch/Minor   | State     | First Release   | Limited Maintenance   | EOL/Terminated   |
|-----------|-----------------------|-----------|-----------------|-----------------------|------------------|
| 3         | 3.0.2                 | Supported | Apr 22, 2025    | TBD                   | TBD              |
| 2         | 2.11.0                | Supported | Dec 17, 2020    | Oct 22, 2025          | Apr 22, 2026     |
| 1.10      | 1.10.15               | EOL       | Aug 27, 2018    | Dec 17, 2020          | June 17, 2021    |
| 1.9       | 1.9.0                 | EOL       | Jan 03, 2018    | Aug 27, 2018          | Aug 27, 2018     |
| 1.8       | 1.8.2                 | EOL       | Mar 19, 2017    | Jan 03, 2018          | Jan 03, 2018     |
| 1.7       | 1.7.1.2               | EOL       | Mar 28, 2016    | Mar 19, 2017          | Mar 19, 2017     |

<!-- End of auto-generated table -->

## Support for Python and Kubernetes versions

Airflow follows a defined policy for supporting Python and Kubernetes versions, aligned with their official release schedules.

## Base OS support for reference Airflow images

The Airflow Community provides container images with a stable Debian OS, supporting multiple versions of Python,
and libraries for database connections.

## Approach to dependencies of Airflow

Airflow uses constraints to ensure repeatable installations.

## Contributing

Help improve Airflow! Read the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst).

## Voting Policy

* Commits need a +1 vote from a committer who is not the author
* When we do AIP voting, both PMC member's and committer's `+1s` are considered a binding vote.

## Who uses Apache Airflow?

Hundreds of organizations use Airflow. See the [INTHEWILD.md](https://github.com/apache/airflow/blob/main/INTHEWILD.md) list.

## Who maintains Apache Airflow?

The Airflow community maintains the project, with core committers/maintainers responsible for reviews and merging PRs.

## What goes into the next release?

Check the [dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md](dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md) document for more information.

## Can I use the Apache Airflow logo in my presentation?

Yes, follow the Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>