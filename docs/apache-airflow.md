# Apache Airflow: Orchestrate, Schedule, and Monitor Workflows with Code

**Apache Airflow** is a powerful platform that allows you to programmatically author, schedule, and monitor your workflows.  Define your workflows as code for improved maintainability, version control, and collaboration.  [Explore the official Apache Airflow repository](https://github.com/apache/airflow) for more details.

---

**Key Features:**

*   **Dynamic Pipelines:** Define workflows in Python, enabling dynamic DAG generation and parameterization.
*   **Extensible Framework:** Leverage a wide range of built-in operators and customize Airflow to fit your specific needs.
*   **Flexible Templating:** Utilize the **Jinja** templating engine for rich customization options.
*   **Rich User Interface**: Monitor progress and troubleshoot issues with the UI.

---

## Core Concepts and Usage

Apache Airflow excels at managing primarily static, slowly-changing workflows.  It's ideal for data processing, with a focus on idempotency and efficient task delegation.

## Key Principles

*   **Dynamic:** Workflows are written in code.
*   **Extensible:** Easily extendable with operators and integrations.
*   **Flexible:** Airflow leverages Jinja2 template engine

## Requirements

Airflow supports Python 3.9+ and offers broad compatibility with popular databases and Kubernetes versions. See the table below for specific compatibility:

|            | Main version (dev)     | Stable version (3.0.5) |
|------------|------------------------|------------------------|
| Python     | 3.10, 3.11, 3.12, 3.13 | 3.9, 3.10, 3.11, 3.12  |
| Platform   | AMD64/ARM64(\*)        | AMD64/ARM64(\*)        |
| Kubernetes | 1.30, 1.31, 1.32, 1.33 | 1.30, 1.31, 1.32, 1.33 |
| PostgreSQL | 13, 14, 15, 16, 17     | 13, 14, 15, 16, 17     |
| MySQL      | 8.0, 8.4, Innovation   | 8.0, 8.4, Innovation   |
| SQLite     | 3.15.0+                | 3.15.0+                |

\* Experimental

## Getting Started

Get up and running quickly with Airflow.  Refer to the [official documentation](https://airflow.apache.org/docs/apache-airflow/stable/) for installation, tutorials, and in-depth guides.

## Installation

Choose your preferred installation method:

### Installing from PyPI

Install Airflow and required extras via pip with constraint files for reliable dependency management.

```bash
pip install 'apache-airflow==3.0.5' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.5/constraints-3.10.txt"
```

Install with extras:

```bash
pip install 'apache-airflow[postgres,google]==3.0.5' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.5/constraints-3.10.txt"
```

### Other Installation Methods

-   Official documentation for more options [here](https://airflow.apache.org/docs/apache-airflow/stable/installation/).
-   Official Docker Images (See [Docker Images](https://hub.docker.com/r/apache/airflow) or [documentation](https://airflow.apache.org/docs/docker-stack/installing-airflow.html))

## User Interface

Airflow features an intuitive web interface for workflow management:

*   **DAGs**: Overview of all DAGs.
*   **Assets**: Overview of Assets with dependencies.
*   **Grid**: Grid representation of a DAG that spans across time.
*   **Graph**: Visualization of a DAG's dependencies and their current status for a specific run.
*   **Home**: Summary statistics of your Airflow environment.
*   **Backfill**: Backfilling a DAG for a specific date range.
*   **Code**: Quick way to view source code of a DAG.

## Official source code

Apache Airflow is an [Apache Software Foundation](https://www.apache.org) (ASF) project,
and our official source code releases:

- Follow the [ASF Release Policy](https://www.apache.org/legal/release-policy.html)
- Can be downloaded from [the ASF Distribution Directory](https://downloads.apache.org/airflow)
- Are cryptographically signed by the release manager
- Are officially voted on by the PMC members during the
  [Release Approval Process](https://www.apache.org/legal/release-policy.html#release-approval)

## Semantic versioning

Airflow follows strict SemVer principles for versioning.  See the [official documentation](https://airflow.apache.org/docs/apache-airflow/stable/release-process.html) for details.

## Version Life Cycle

Check the table below for the lifecycle of Airflow versions:

| Version   | Current Patch/Minor   | State     | First Release   | Limited Maintenance   | EOL/Terminated   |
|-----------|-----------------------|-----------|-----------------|-----------------------|------------------|
| 3         | 3.0.5                 | Supported | Apr 22, 2025    | TBD                   | TBD              |
| 2         | 2.11.0                | Supported | Dec 17, 2020    | Oct 22, 2025          | Apr 22, 2026     |
| 1.10      | 1.10.15               | EOL       | Aug 27, 2018    | Dec 17, 2020          | June 17, 2021    |
| 1.9       | 1.9.0                 | EOL       | Jan 03, 2018    | Aug 27, 2018          | Aug 27, 2018     |
| 1.8       | 1.8.2                 | EOL       | Mar 19, 2017    | Jan 03, 2018          | Jan 03, 2018     |
| 1.7       | 1.7.1.2               | EOL       | Mar 28, 2016    | Mar 19, 2017          | Mar 19, 2017     |

## Contributing

Join the Airflow community!  Review the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst) for contribution guidelines, coding standards, and pull request instructions.

## Who Uses Apache Airflow?

Airflow is used by hundreds of organizations, check out [INTHEWILD.md](https://github.com/apache/airflow/blob/main/INTHEWILD.md) to see some.

## Who Maintains Apache Airflow?

Airflow is maintained by a dedicated community. The [core committers/maintainers](https://people.apache.org/committers-by-project.html#airflow) review and merge contributions.

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

*   <a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
*   <a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>