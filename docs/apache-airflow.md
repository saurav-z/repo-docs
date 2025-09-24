# Apache Airflow: The Workflow Orchestration Platform

**Automate, schedule, and monitor your workflows with Apache Airflow, a powerful, open-source platform for orchestrating complex data pipelines. [Explore the Apache Airflow Repository](https://github.com/apache/airflow)**

<picture width="500">
  <img
    src="https://github.com/apache/airflow/blob/19ebcac2395ef9a6b6ded3a2faa29dc960c1e635/docs/apache-airflow/img/logos/wordmark_1.png?raw=true"
    alt="Apache Airflow logo"
  />
</picture>

Apache Airflow is the leading platform for programmatically authoring, scheduling, and monitoring workflows.  It allows you to define workflows as code, making them more manageable, versionable, and collaborative.

**Key Features:**

*   **Dynamic Workflow Definition:** Define pipelines using code for dynamic DAG generation and parameterization.
*   **Extensibility:**  Leverage a wide range of built-in operators and customize Airflow to meet specific needs.
*   **Flexibility:** Utilize Jinja templating for rich customizations and control over your workflows.
*   **Scalability:** Executes tasks on an array of workers, enabling you to orchestrate the most complex tasks.
*   **User-Friendly Interface:** Visualize, monitor, and troubleshoot your pipelines with an intuitive UI.

## Project Focus

Airflow excels with workflows that are largely static and have infrequent changes, such as data processing and ETL pipelines.

## Principles

*   **Dynamic:** Pipelines are defined in code, enabling dynamic dag generation and parameterization.
*   **Extensible:** The Airflow framework includes a wide range of built-in operators and can be extended to fit your needs.
*   **Flexible:** Airflow leverages the [**Jinja**](https://jinja.palletsprojects.com) templating engine, allowing rich customizations.

## Requirements

Airflow is tested with:

|            | Main version (dev)           | Stable version (3.0.6) |
|------------|------------------------------|------------------------|
| Python     | 3.10, 3.11, 3.12, 3.13       | 3.9, 3.10, 3.11, 3.12  |
| Platform   | AMD64/ARM64(\*)              | AMD64/ARM64(\*)        |
| Kubernetes | 1.30, 1.31, 1.32, 1.33, 1.34 | 1.30, 1.31, 1.32, 1.33 |
| PostgreSQL | 13, 14, 15, 16, 17           | 13, 14, 15, 16, 17     |
| MySQL      | 8.0, 8.4, Innovation         | 8.0, 8.4, Innovation   |
| SQLite     | 3.15.0+                      | 3.15.0+                |

\* Experimental

**Note**: MariaDB is not tested/recommended.

**Note**: SQLite is used in Airflow tests. Do not use it in production. We recommend
using the latest stable version of SQLite for local development.

**Note**: Airflow currently can be run on POSIX-compliant Operating Systems. For development, it is regularly
tested on fairly modern Linux Distros and recent versions of macOS.
On Windows you can run it via WSL2 (Windows Subsystem for Linux 2) or via Linux Containers.
The work to add Windows support is tracked via [#10388](https://github.com/apache/airflow/issues/10388), but
it is not a high priority. You should only use Linux-based distros as "Production" execution environment
as this is the only environment that is supported. The only distro that is used in our CI tests and that
is used in the [Community managed DockerHub image](https://hub.docker.com/p/apache/airflow) is
`Debian Bookworm`.

## Getting Started

Explore the [official Airflow website](https://airflow.apache.org/docs/apache-airflow/stable/) for installation, getting started guides, and tutorials.

## Installing from PyPI

Install Airflow using pip, but be aware of the need for constraint files. For example:

```bash
pip install 'apache-airflow==3.0.6' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
```

For extras:

```bash
pip install 'apache-airflow[postgres,google]==3.0.6' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
```

## Installation

Refer to [INSTALLING.md](INSTALLING.md) for a comprehensive setup guide.

## Official Source Code

Airflow is an Apache Software Foundation project. Official releases:

*   Follow the ASF Release Policy
*   Available for download from the ASF Distribution Directory
*   Cryptographically signed
*   Officially voted on by PMC members

## Convenience Packages

Install Airflow in different ways:

*   PyPI releases:  `pip install apache-airflow`
*   Docker Images: Use the `docker` tool. Learn about using, customizing, and extending the images in the [Latest docs](https://airflow.apache.org/docs/docker-stack/index.html).
*   Tags in GitHub: Retrieve the git project sources via git.

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

Airflow uses SemVer (MAJOR.MINOR.PATCH) for releases starting with 2.0.0.

## Version Life Cycle

Apache Airflow version life cycle:

<!-- This table is automatically updated by prek scripts/ci/prek/supported_versions.py -->
<!-- Beginning of auto-generated table -->

| Version   | Current Patch/Minor   | State     | First Release   | Limited Maintenance   | EOL/Terminated   |
|-----------|-----------------------|-----------|-----------------|-----------------------|------------------|
| 3         | 3.0.6                 | Supported | Apr 22, 2025    | TBD                   | TBD              |
| 2         | 2.11.0                | Supported | Dec 17, 2020    | Oct 22, 2025          | Apr 22, 2026     |
| 1.10      | 1.10.15               | EOL       | Aug 27, 2018    | Dec 17, 2020          | June 17, 2021    |
| 1.9       | 1.9.0                 | EOL       | Jan 03, 2018    | Aug 27, 2018          | Aug 27, 2018     |
| 1.8       | 1.8.2                 | EOL       | Mar 19, 2017    | Jan 03, 2018          | Jan 03, 2018     |
| 1.7       | 1.7.1.2               | EOL       | Mar 28, 2016    | Mar 19, 2017          | Mar 19, 2017     |

<!-- End of auto-generated table -->

## Support for Python and Kubernetes versions

Airflow's support for Python and Kubernetes versions aligns with their respective EOL schedules.

## Base OS support for reference Airflow images

Airflow provides convenient container images built using stable Debian OS versions, including:

* Base OS with necessary packages to install Airflow (stable Debian OS)
* Base Python installation
* Libraries required to connect to supported Databases
* Predefined set of popular providers
* Possibility of building your own, custom image
* In the future Airflow might also support a "slim" version without providers nor database clients installed

## Approach to dependencies of Airflow

Airflow uses constraints to manage dependencies effectively.

### Approach for dependencies for Airflow Core

*   `SQLAlchemy`: upper-bound
*   `Alembic`: upper-bound
*   `Flask`: upper-bound
*   `werkzeug`: upper-bound
*   `celery`: upper-bound
*   `kubernetes`: upper-bound

### Approach for dependencies in Airflow Providers and extras

By default, no dependencies are upper-bounded for providers, though maintainers can add limits.

## Contributing

Contribute to Airflow! Check out the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst).

## Voting Policy

*   Commits require a +1 vote from a non-author committer.
*   AIP voting: Binding votes from PMC members and committers.

## Who uses Apache Airflow?

Hundreds of organizations use Airflow - see [INTHEWILD.md](https://github.com/apache/airflow/blob/main/INTHEWILD.md)

## Who maintains Apache Airflow?

The Airflow community and core committers/maintainers are responsible for reviewing and merging PRs.

## What goes into the next release?

For info on releases, see the [dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md](dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md) document.

## Can I use the Apache Airflow logo in my presentation?

Yes, follow Apache Foundation trademark policies and the [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

*   <a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
*   <a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>