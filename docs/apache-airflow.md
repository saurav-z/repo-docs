# Apache Airflow: Orchestrate Your Data Pipelines with Ease

**Apache Airflow is a powerful platform that allows you to programmatically author, schedule, and monitor your workflows.**

[![License](https://img.shields.io/:license-Apache%202-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0.txt)
[![PyPI version](https://badge.fury.io/py/apache-airflow.svg)](https://badge.fury.io/py/apache-airflow)
[![Docker Pulls](https://img.shields.io/docker/pulls/apache/airflow.svg)](https://hub.docker.com/r/apache/airflow)
[![GitHub Build main](https://github.com/apache/airflow/actions/workflows/ci-amd.yml/badge.svg)](https://github.com/apache/airflow/actions)
[![Slack Status](https://img.shields.io/badge/slack-join_chat-white.svg?logo=slack&style=social)](https://s.apache.org/airflow-slack)

<picture width="500">
  <img
    src="https://github.com/apache/airflow/blob/19ebcac2395ef9a6b6ded3a2faa29dc960c1e635/docs/apache-airflow/img/logos/wordmark_1.png?raw=true"
    alt="Apache Airflow logo"
  />
</picture>

Apache Airflow ([original repo](https://github.com/apache/airflow)) empowers data engineers and scientists to build, schedule, and monitor complex workflows as code, enabling maintainability, versioning, and collaboration.

**Key Features:**

*   **Dynamic Workflow Definition:** Define pipelines using Python code for flexibility and customization.
*   **Extensibility:** Leverage a rich set of built-in operators and easily extend Airflow to integrate with various systems.
*   **Scheduling and Monitoring:** Schedule tasks and monitor their execution through a user-friendly interface.
*   **Idempotent Tasks:** Designed for idempotent tasks, ensuring data consistency.
*   **Scalability:** Execute tasks on a distributed array of workers for handling large workloads.

## Project Focus

Airflow excels in managing mostly static, slowly changing workflows, often used for data processing.  It is designed to be flexible and extensible, ideal for data pipelines.

## Principles

-   **Dynamic:** Pipelines are defined in code, enabling dynamic dag generation and parameterization.
-   **Extensible:** The Airflow framework includes a wide range of built-in operators and can be extended to fit your needs.
-   **Flexible:** Airflow leverages the [**Jinja**](https://jinja.palletsprojects.com) templating engine, allowing rich customizations.

## Requirements

See Requirements section in the original README.

## Getting Started

Visit the official Airflow website documentation (latest **stable** release) for help with
[installing Airflow](https://airflow.apache.org/docs/apache-airflow/stable/installation/),
[getting started](https://airflow.apache.org/docs/apache-airflow/stable/start.html), or walking
through a more complete [tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/).

> Note: If you're looking for documentation for the main branch (latest development branch): you can find it on [s.apache.org/airflow-docs](https://s.apache.org/airflow-docs/).

For more information on Airflow Improvement Proposals (AIPs), visit
the [Airflow Wiki](https://cwiki.apache.org/confluence/display/AIRFLOW/Airflow+Improvement+Proposals).

Documentation for dependent projects like provider distributions, Docker image, Helm Chart, you'll find it in [the documentation index](https://airflow.apache.org/docs/).

## Installing from PyPI

See Installing from PyPI section in the original README.

## Installation

For comprehensive instructions on setting up your local development environment and installing Apache Airflow, please refer to the [INSTALLING.md](INSTALLING.md) file.

## Official Source Code

Apache Airflow is an [Apache Software Foundation](https://www.apache.org) (ASF) project,
and our official source code releases:

- Follow the [ASF Release Policy](https://www.apache.org/legal/release-policy.html)
- Can be downloaded from [the ASF Distribution Directory](https://downloads.apache.org/airflow)
- Are cryptographically signed by the release manager
- Are officially voted on by the PMC members during the
  [Release Approval Process](https://www.apache.org/legal/release-policy.html#release-approval)

Following the ASF rules, the source packages released must be sufficient for a user to build and test the
release provided they have access to the appropriate platform and tools.

## Convenience Packages

See Convenience Packages section in the original README.

## User Interface

-   **DAGs:** Overview of all DAGs in your environment.

    ![DAGs](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/dags.png)

-   **Assets:** Overview of Assets with dependencies.

    ![Asset Dependencies](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/assets_graph.png)

-   **Grid:** Grid representation of a DAG that spans across time.

    ![Grid](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/grid.png)

-   **Graph:** Visualization of a DAG's dependencies and their current status for a specific run.

    ![Graph](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/graph.png)

-   **Home:** Summary statistics of your Airflow environment.

    ![Home](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/home.png)

-   **Backfill:** Backfilling a DAG for a specific date range.

    ![Backfill](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/backfill.png)

-   **Code:** Quick way to view source code of a DAG.

    ![Code](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/code.png)

## Semantic Versioning

As of Airflow 2.0.0, we support a strict [SemVer](https://semver.org/) approach for all packages released.

There are few specific rules that we agreed to that define details of versioning of the different
packages:

* **Airflow**: SemVer rules apply to core airflow only (excludes any changes to providers).
  Changing limits for versions of Airflow dependencies is not a breaking change on its own.
* **Airflow Providers**: SemVer rules apply to changes in the particular provider's code only.
  SemVer MAJOR and MINOR versions for the packages are independent of the Airflow version.
  For example, `google 4.1.0` and `amazon 3.0.3` providers can happily be installed
  with `Airflow 2.1.2`. If there are limits of cross-dependencies between providers and Airflow packages,
  they are present in providers as `install_requires` limitations. We aim to keep backwards
  compatibility of providers with all previously released Airflow 2 versions but
  there will sometimes be breaking changes that might make some, or all
  providers, have minimum Airflow version specified.
* **Airflow Helm Chart**: SemVer rules apply to changes in the chart only. SemVer MAJOR and MINOR
  versions for the chart are independent of the Airflow version. We aim to keep backwards
  compatibility of the Helm Chart with all released Airflow 2 versions, but some new features might
  only work starting from specific Airflow releases. We might however limit the Helm
  Chart to depend on minimal Airflow version.
* **Airflow API clients**: Their versioning is independent from Airflow versions. They follow their own
  SemVer rules for breaking changes and new features - which for example allows to change the way we generate
  the clients.

## Version Life Cycle

Apache Airflow version life cycle:

<!-- This table is automatically updated by pre-commit scripts/ci/pre_commit/supported_versions.py -->
<!-- Beginning of auto-generated table -->

| Version   | Current Patch/Minor   | State     | First Release   | Limited Maintenance   | EOL/Terminated   |
|-----------|-----------------------|-----------|-----------------|-----------------------|------------------|
| 3         | 3.0.3                 | Supported | Apr 22, 2025    | TBD                   | TBD              |
| 2         | 2.11.0                | Supported | Dec 17, 2020    | Oct 22, 2025          | Apr 22, 2026     |
| 1.10      | 1.10.15               | EOL       | Aug 27, 2018    | Dec 17, 2020          | June 17, 2021    |
| 1.9       | 1.9.0                 | EOL       | Jan 03, 2018    | Aug 27, 2018          | Aug 27, 2018     |
| 1.8       | 1.8.2                 | EOL       | Mar 19, 2017    | Jan 03, 2018          | Jan 03, 2018     |
| 1.7       | 1.7.1.2               | EOL       | Mar 28, 2016    | Mar 19, 2017          | Mar 19, 2017     |

<!-- End of auto-generated table -->

Limited support versions will be supported with security and critical bug fix only.
EOL versions will not get any fixes nor support.
We always recommend that all users run the latest available minor release for whatever major version is in use.
We **highly** recommend upgrading to the latest Airflow major release at the earliest convenient time and before the EOL date.

## Support for Python and Kubernetes Versions

See Support for Python and Kubernetes Versions section in the original README.

## Base OS Support for Reference Airflow Images

See Base OS support for reference Airflow images section in the original README.

## Approach to Dependencies of Airflow

See Approach to dependencies of Airflow section in the original README.

## Contributing

Help improve Apache Airflow by following the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst) and [contribution quickstart](https://github.com/apache/airflow/blob/main/contributing-docs/03_contributors_quick_start.rst).

## Voting Policy

See Voting Policy section in the original README.

## Who Uses Apache Airflow?

See Who uses Apache Airflow? section in the original README.

## Who Maintains Apache Airflow?

See Who maintains Apache Airflow? section in the original README.

## What Goes into the Next Release?

See What goes into the next release? section in the original README.

## Can I Use the Apache Airflow Logo in My Presentation?

Yes, see Can I use the Apache Airflow logo in my presentation? section in the original README.

## Links

-   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
-   [Chat](https://s.apache.org/airflow-slack)
-   [Community Information](https://airflow.apache.org/community/)

## Sponsors

<!-- Ordered by most recently "funded" -->

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>