# Apache Airflow: Automate, Schedule, and Monitor Workflows

**Apache Airflow** is a powerful platform for programmatically authoring, scheduling, and monitoring workflows, allowing you to define pipelines as code for increased maintainability and collaboration. Explore the official [Apache Airflow repository](https://github.com/apache/airflow) for more information.

## Key Features

*   **Dynamic Workflows:** Define pipelines in code for dynamic generation and parameterization.
*   **Extensible:** Utilize a wide range of built-in operators and customize with extensions.
*   **Flexible:** Leverage the Jinja templating engine for rich customization.
*   **User-Friendly UI:** Visualize pipelines, monitor progress, and troubleshoot issues.
*   **Open Source:**  Benefit from a vibrant community and the Apache 2.0 license.

## Project Overview

Apache Airflow excels with primarily static and evolving workflows, offering clarity and continuity in data processing pipelines. It's designed for idempotent tasks and can process real-time data in batches. Consider Airflow for authoring workflows (DAGs) that orchestrate tasks. The Airflow scheduler executes your tasks on a collection of workers while observing task dependencies.

## Principles

-   **Dynamic**: Pipelines are defined in code, enabling dynamic dag generation and parameterization.
-   **Extensible**: The Airflow framework includes a wide range of built-in operators and can be extended to fit your needs.
-   **Flexible**: Airflow leverages the [**Jinja**](https://jinja.palletsprojects.com) templating engine, allowing rich customizations.

## Requirements

Airflow is tested and supports the following:

| Feature        | Main Version (dev)     | Stable Version (3.0.3) |
| -------------- | ----------------------- | ----------------------- |
| Python         | 3.10, 3.11, 3.12, 3.13 | 3.9, 3.10, 3.11, 3.12  |
| Platform       | AMD64/ARM64(\*)        | AMD64/ARM64(\*)        |
| Kubernetes     | 1.30, 1.31, 1.32, 1.33 | 1.30, 1.31, 1.32, 1.33 |
| PostgreSQL     | 13, 14, 15, 16, 17     | 13, 14, 15, 16, 17     |
| MySQL          | 8.0, 8.4, Innovation   | 8.0, 8.4, Innovation   |
| SQLite         | 3.15.0+                | 3.15.0+                |

\* Experimental

## Getting Started

Begin your journey with Airflow by exploring the official documentation for [installation](https://airflow.apache.org/docs/apache-airflow/stable/installation/), [getting started](https://airflow.apache.org/docs/apache-airflow/stable/start.html), and a comprehensive [tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/).

## Installing from PyPI

Install Airflow using pip, along with constraint files for reliable installation:

```bash
pip install 'apache-airflow==3.0.3' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.3/constraints-3.10.txt"
```

For extras (e.g., postgres, google):

```bash
pip install 'apache-airflow[postgres,google]==3.0.3' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.3/constraints-3.10.txt"
```

## Installation

Detailed installation instructions are available in the [INSTALLING.md](INSTALLING.md) file.

## Official Source Code

Apache Airflow is an [Apache Software Foundation](https://www.apache.org) project. Official source code releases:

*   Follow the [ASF Release Policy](https://www.apache.org/legal/release-policy.html).
*   Available from [the ASF Distribution Directory](https://downloads.apache.org/airflow).
*   Cryptographically signed.
*   Officially voted on by PMC members.

## Convenience Packages

Airflow offers convenient installation methods, including:

*   [PyPI releases](https://pypi.org/project/apache-airflow/) using pip.
*   [Docker Images](https://hub.docker.com/r/apache/airflow).
*   [Tags in GitHub](https://github.com/apache/airflow/tags).

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

Airflow 2.0+ follows SemVer for package releases.

## Version Life Cycle

The Airflow version life cycle is:

| Version   | Current Patch/Minor   | State     | First Release   | Limited Maintenance   | EOL/Terminated   |
|-----------|-----------------------|-----------|-----------------|-----------------------|------------------|
| 3         | 3.0.3                 | Supported | Apr 22, 2025    | TBD                   | TBD              |
| 2         | 2.11.0                | Supported | Dec 17, 2020    | Oct 22, 2025          | Apr 22, 2026     |
| 1.10      | 1.10.15               | EOL       | Aug 27, 2018    | Dec 17, 2020          | June 17, 2021    |
| 1.9       | 1.9.0                 | EOL       | Jan 03, 2018    | Aug 27, 2018          | Aug 27, 2018     |
| 1.8       | 1.8.2                 | EOL       | Mar 19, 2017    | Jan 03, 2018          | Jan 03, 2018     |
| 1.7       | 1.7.1.2               | EOL       | Mar 28, 2016    | Mar 19, 2017          | Mar 19, 2017     |

## Support for Python and Kubernetes versions

Airflow supports Python and Kubernetes versions based on their official release schedules, see the [Python Developer's Guide](https://devguide.python.org/#status-of-python-branches) and [Kubernetes version skew policy](https://kubernetes.io/docs/setup/release/version-skew-policy/).

## Base OS support for reference Airflow images

The Airflow Community provides packaged container images with:

*   Debian OS.
*   Supported Python versions.
*   Database connection libraries.
*   Predefined providers.
*   Custom image building.

## Approach to dependencies of Airflow

Airflow's dependencies are managed with constraints to ensure repeatable installations while allowing for dependency upgrades.

## Contributing

Contribute to Apache Airflow through our [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst).

## Voting Policy

*   Commits require a +1 vote from a committer.
*   AIP voting requires +1s from PMC members and committers.

## Who Uses Apache Airflow?

Discover organizations using Apache Airflow [in the wild](https://github.com/apache/airflow/blob/main/INTHEWILD.md).

## Who Maintains Apache Airflow?

The Apache Airflow project is maintained by the [community](https://github.com/apache/airflow/graphs/contributors), led by the [core committers/maintainers](https://people.apache.org/committers-by-project.html#airflow).

## What goes into the next release?

Find details in the [dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md](dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md) document in the repository.

## Can I use the Apache Airflow logo in my presentation?

Yes, while adhering to the Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

CI infrastructure sponsored by:

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>