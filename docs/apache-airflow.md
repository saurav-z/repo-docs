# Apache Airflow: Automate, Schedule, and Monitor Workflows

**Orchestrate your data pipelines with Apache Airflow, a leading platform for programmatically authoring, scheduling, and monitoring workflows.** ([Original Repository](https://github.com/apache/airflow))

<picture width="500">
  <img
    src="https://github.com/apache/airflow/blob/19ebcac2395ef9a6b6ded3a2faa29dc960c1e635/docs/apache-airflow/img/logos/wordmark_1.png?raw=true"
    alt="Apache Airflow logo"
  />
</picture>

Apache Airflow (or Airflow) empowers you to define workflows as code, making them manageable, versionable, testable, and collaborative. Build and manage your data pipelines with ease, ensuring efficiency and reliability.

**Key Features:**

*   **Dynamic Workflow Definition:** Define pipelines in code for flexibility and parameterization.
*   **Extensible Framework:** Leverage a wide array of built-in operators and easily extend Airflow to fit your specific needs.
*   **Rich UI and CLI:**  Visualize pipeline progress, monitor performance, and troubleshoot issues with a user-friendly interface and powerful command-line tools.
*   **Scalable Scheduling and Execution:** The Airflow scheduler executes tasks on a distributed worker pool, following your specified dependencies.
*   **Idempotent Task Execution:**  Designed with idempotent tasks in mind, promoting data integrity and preventing duplication.

## Quick Start

*   Install: [INSTALLING.md](INSTALLING.md)
*   Start: [Getting started](https://airflow.apache.org/docs/apache-airflow/stable/start.html)
*   Tutorial: [tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/)
*   Documentation: [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   Airflow Wiki: [Airflow Wiki](https://cwiki.apache.org/confluence/display/AIRFLOW/Airflow+Improvement+Proposals)

## Getting Started

Ready to dive in? Explore the official Airflow documentation for detailed installation instructions, a quick start guide, and a comprehensive tutorial to get you up and running. For more information on Airflow Improvement Proposals (AIPs), visit the Airflow Wiki, and explore documentation for dependent projects on the documentation index.

## Installation

Detailed setup and installation instructions are available in the [INSTALLING.md](INSTALLING.md) file.

## Installing from PyPI

Install Apache Airflow with pip using constraints for repeatable installations.

**Install Airflow:**

```bash
pip install 'apache-airflow==3.0.6' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
```

**Install with extras (e.g., postgres, google):**

```bash
pip install 'apache-airflow[postgres,google]==3.0.6' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
```
## Requirements

Airflow is tested with:

|            | Main version (dev)     | Stable version (3.0.6) |
|------------|------------------------|------------------------|
| Python     | 3.10, 3.11, 3.12, 3.13 | 3.9, 3.10, 3.11, 3.12  |
| Platform   | AMD64/ARM64(\*)        | AMD64/ARM64(\*)        |
| Kubernetes | 1.30, 1.31, 1.32, 1.33 | 1.30, 1.31, 1.32, 1.33 |
| PostgreSQL | 13, 14, 15, 16, 17     | 13, 14, 15, 16, 17     |
| MySQL      | 8.0, 8.4, Innovation   | 8.0, 8.4, Innovation   |
| SQLite     | 3.15.0+                | 3.15.0+                |

\* Experimental

## User Interface

Airflow provides a rich web UI for monitoring and managing your workflows:

*   **DAGs:** Overview of all DAGs in your environment.

  ![DAGs](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/dags.png)

*   **Assets:** Overview of Assets with dependencies.

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

## Official Source Code

Apache Airflow is an [Apache Software Foundation](https://www.apache.org) (ASF) project, and our official source code releases:

*   Follow the [ASF Release Policy](https://www.apache.org/legal/release-policy.html)
*   Can be downloaded from [the ASF Distribution Directory](https://downloads.apache.org/airflow)
*   Are cryptographically signed by the release manager
*   Are officially voted on by the PMC members during the
  [Release Approval Process](https://www.apache.org/legal/release-policy.html#release-approval)

## Convenience packages

There are other ways of installing and using Airflow. Those are "convenience" methods - they are
not "official releases" as stated by the `ASF Release Policy`, but they can be used by the users
who do not want to build the software themselves.

Those are - in the order of most common ways people install Airflow:

- [PyPI releases](https://pypi.org/project/apache-airflow/) to install Airflow using standard `pip` tool
- [Docker Images](https://hub.docker.com/r/apache/airflow) to install airflow via
  `docker` tool, use them in Kubernetes, Helm Charts, `docker-compose`, `docker swarm`, etc. You can
  read more about using, customizing, and extending the images in the
  [Latest docs](https://airflow.apache.org/docs/docker-stack/index.html), and
  learn details on the internals in the [images](https://airflow.apache.org/docs/docker-stack/index.html) document.
- [Tags in GitHub](https://github.com/apache/airflow/tags) to retrieve the git project sources that
  were used to generate official source packages via git

All those artifacts are not official releases, but they are prepared using officially released sources.
Some of those artifacts are "development" or "pre-release" ones, and they are clearly marked as such
following the ASF Policy.

## Semantic Versioning

Airflow uses [SemVer](https://semver.org/) (Semantic Versioning) for versioning from 2.0.0 onwards.

## Version Life Cycle

*   **3.0.6** | Supported | Apr 22, 2025 | TBD | TBD
*   **2.11.0** | Supported | Dec 17, 2020 | Oct 22, 2025 | Apr 22, 2026
*   1.10  | EOL | Aug 27, 2018 | Dec 17, 2020 | June 17, 2021
*   1.9 | EOL | Jan 03, 2018 | Aug 27, 2018 | Aug 27, 2018
*   1.8 | EOL | Mar 19, 2017 | Jan 03, 2018 | Jan 03, 2018
*   1.7 | EOL | Mar 28, 2016 | Mar 19, 2017 | Mar 19, 2017

## Contributing

Contribute to Apache Airflow! Check out the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst) for details on how to get involved.

## Voting Policy

* Commits need a +1 vote from a committer who is not the author
* When we do AIP voting, both PMC member's and committer's `+1s` are considered a binding vote.

## Who Uses Apache Airflow?

Apache Airflow is widely adopted. See the [INTHEWILD.md](https://github.com/apache/airflow/blob/main/INTHEWILD.md) for a list of organizations using Airflow.

## Who Maintains Apache Airflow?

Airflow is a community-driven project. Find the [core committers/maintainers](https://people.apache.org/committers-by-project.html#airflow) responsible for steering the project. Review the Apache Airflow [committer requirements](https://github.com/apache/airflow/blob/main/COMMITTERS.rst#guidelines-to-become-an-airflow-committer) if you'd like to contribute.

## What goes into the next release?

For details on the release process, check the
[What goes into the next release](dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md) document in the `dev` folder of the
repository.

## Can I use the Apache Airflow logo in my presentation?

Yes, abide by the Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook). The most up-to-date logos are found in [this repo](https://github.com/apache/airflow/tree/main/airflow-core/docs/img/logos/) and on the Apache Software Foundation [website](https://www.apache.org/logos/about.html).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>