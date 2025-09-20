<!--
 Licensed to the Apache Software Foundation (ASF) under one
 or more contributor license agreements.  See the NOTICE file
 distributed with this work for additional information
 regarding copyright ownership.  The ASF licenses this file
 to you under the Apache License, Version 2.0 (the
 "License"); you may not use this file except in compliance
 with the License.  You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing,
 software distributed under the License is distributed on an
 "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 KIND, either express or implied.  See the License for the
 specific language governing permissions and limitations
 under the License.
-->

<!-- START Apache Airflow, please keep comment here to allow auto update of PyPI readme.md -->
# Apache Airflow: Orchestrate Your Workflows with Ease

**Apache Airflow is the leading platform for programmatically authoring, scheduling, and monitoring complex workflows.**

[![License](https://img.shields.io/:license-Apache%202-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0.txt)
[![PyPI version](https://badge.fury.io/py/apache-airflow.svg)](https://badge.fury.io/py/apache-airflow)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/apache-airflow.svg)](https://pypi.org/project/apache-airflow/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/apache-airflow)](https://pypi.org/project/apache-airflow/)
[![Docker Pulls](https://img.shields.io/docker/pulls/apache/airflow.svg)](https://hub.docker.com/r/apache/airflow)
[![Docker Stars](https://img.shields.io/docker/stars/apache/airflow.svg)](https://hub.docker.com/r/apache/airflow)
[![Artifact HUB](https://img.shields.io/endpoint?url=https://artifacthub.io/badge/repository/apache-airflow)](https://artifacthub.io/packages/search?repo=apache-airflow)
[![Contributors](https://img.shields.io/github/contributors/apache/airflow)](https://github.com/apache/airflow/graphs/contributors)
[![Slack Status](https://img.shields.io/badge/slack-join_chat-white.svg?logo=slack&style=social)](https://s.apache.org/airflow-slack)
![Commit Activity](https://img.shields.io/github/commit-activity/m/apache/airflow)
[![OSSRank](https://shields.io/endpoint?url=https://ossrank.com/shield/6)](https://ossrank.com/p/6)

[![GitHub Build main](https://github.com/apache/airflow/actions/workflows/ci-amd.yml/badge.svg)](https://github.com/apache/airflow/actions)
[![GitHub Build main](https://github.com/apache/airflow/actions/workflows/ci-arm.yml/badge.svg)](https://github.com/apache/airflow/actions)
[![GitHub Build 3.1](https://github.com/apache/airflow/actions/workflows/ci-amd.yml/badge.svg?branch=v3-1-test)](https://github.com/apache/airflow/actions)
[![GitHub Build 3.1](https://github.com/apache/airflow/actions/workflows/ci-arm.yml/badge.svg?branch=v3-1-test)](https://github.com/apache/airflow/actions)
[![GitHub Build 2.11](https://github.com/apache/airflow/actions/workflows/ci.yml/badge.svg?branch=v2-11-test)](https://github.com/apache/airflow/actions)

<picture width="500">
  <img
    src="https://github.com/apache/airflow/blob/19ebcac2395ef9a6b6ded3a2faa29dc960c1e635/docs/apache-airflow/img/logos/wordmark_1.png?raw=true"
    alt="Apache Airflow logo"
  />
</picture>

[Apache Airflow](https://airflow.apache.org/docs/apache-airflow/stable/) (or simply Airflow) is a platform to programmatically author, schedule, and monitor workflows.

When workflows are defined as code, they become more maintainable, versionable, testable, and collaborative.

Use Airflow to author workflows (Dags) that orchestrate tasks. The Airflow scheduler executes your tasks on an array of workers while following the specified dependencies. Rich command line utilities make performing complex surgeries on DAGs a snap. The rich user interface makes it easy to visualize pipelines running in production, monitor progress, and troubleshoot issues when needed.

<!-- END Apache Airflow, please keep comment here to allow auto update of PyPI readme.md -->
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of contents**

- [About Apache Airflow](#apache-airflow-orchestration)
    - [Key Features](#key-features)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Installation](#installation)
    - [Installing from PyPI](#installing-from-pypi)
- [Official Source Code](#official-source-code)
- [Convenience Packages](#convenience-packages)
- [User Interface](#user-interface)
- [Semantic Versioning](#semantic-versioning)
- [Version Life Cycle](#version-life-cycle)
- [Support for Python and Kubernetes Versions](#support-for-python-and-kubernetes-versions)
- [Base OS Support for Reference Airflow Images](#base-os-support-for-reference-airflow-images)
- [Approach to Dependencies of Airflow](#approach-to-dependencies-of-airflow)
- [Contributing](#contributing)
- [Voting Policy](#voting-policy)
- [Who Uses Apache Airflow?](#who-uses-apache-airflow)
- [Who Maintains Apache Airflow?](#who-maintains-apache-airflow)
- [What Goes Into the Next Release?](#what-goes-into-the-next-release)
- [Can I Use the Apache Airflow Logo in My Presentation?](#can-i-use-the-apache-airflow-logo-in-my-presentation)
- [Links](#links)
- [Sponsors](#sponsors)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## About Apache Airflow: Workflow Orchestration

Apache Airflow is a leading open-source platform for orchestrating complex data pipelines and workflows, offering unparalleled flexibility and control. Designed for developers and data engineers, Airflow empowers you to define, schedule, and monitor your workflows as code, making them more maintainable, versionable, and collaborative.

### Key Features:

*   **Dynamic Pipelines:** Define workflows using Python code, enabling dynamic DAG generation and parameterization.
*   **Extensible Architecture:** Benefit from a wide range of built-in operators and easy extensibility to meet your specific needs.
*   **Flexible Templating:** Leverage the Jinja templating engine for rich customizations and workflow control.
*   **Scalable Scheduling and Execution:** Airflow's scheduler handles task execution across distributed workers, ensuring efficient and reliable workflow runs.
*   **Rich User Interface:** Monitor your pipelines in production, track progress, and troubleshoot issues with an intuitive web interface.

## Requirements

Apache Airflow is tested with:

|            | Main version (dev)     | Stable version (3.0.6) |
|------------|------------------------|------------------------|
| Python     | 3.10, 3.11, 3.12, 3.13 | 3.9, 3.10, 3.11, 3.12  |
| Platform   | AMD64/ARM64(\*)        | AMD64/ARM64(\*)        |
| Kubernetes | 1.30, 1.31, 1.32, 1.33 | 1.30, 1.31, 1.32, 1.33 |
| PostgreSQL | 13, 14, 15, 16, 17     | 13, 14, 15, 16, 17     |
| MySQL      | 8.0, 8.4, Innovation   | 8.0, 8.4, Innovation   |
| SQLite     | 3.15.0+                | 3.15.0+                |

\* Experimental

**Note:** MariaDB is not tested/recommended.

**Note:** SQLite is used in Airflow tests. Do not use it in production. We recommend
using the latest stable version of SQLite for local development.

**Note:** Airflow currently can be run on POSIX-compliant Operating Systems. For development, it is regularly
tested on fairly modern Linux Distros and recent versions of macOS.
On Windows you can run it via WSL2 (Windows Subsystem for Linux 2) or via Linux Containers.
The work to add Windows support is tracked via [#10388](https://github.com/apache/airflow/issues/10388), but
it is not a high priority. You should only use Linux-based distros as "Production" execution environment
as this is the only environment that is supported. The only distro that is used in our CI tests and that
is used in the [Community managed DockerHub image](https://hub.docker.com/p/apache/airflow) is
`Debian Bookworm`.

## Getting Started

Get up and running with Apache Airflow quickly!  Explore the [official Airflow documentation](https://airflow.apache.org/docs/apache-airflow/stable/) for comprehensive installation instructions, getting started guides, and tutorials.

## Installation

For detailed instructions on setting up your local development environment and installing Apache Airflow, refer to the [INSTALLING.md](INSTALLING.md) file.

### Installing from PyPI

Install Apache Airflow easily using pip with specific constraints for a repeatable installation.

1.  Install just Airflow:

    ```bash
    pip install 'apache-airflow==3.0.6' \
     --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
    ```

2.  Install with extras (e.g., postgres, google):

    ```bash
    pip install 'apache-airflow[postgres,google]==3.0.6' \
     --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
    ```

## Official Source Code

Apache Airflow is an [Apache Software Foundation](https://www.apache.org) (ASF) project, and its official source code releases:

*   Follow the [ASF Release Policy](https://www.apache.org/legal/release-policy.html)
*   Can be downloaded from [the ASF Distribution Directory](https://downloads.apache.org/airflow)
*   Are cryptographically signed by the release manager
*   Are officially voted on by the PMC members during the
    [Release Approval Process](https://www.apache.org/legal/release-policy.html#release-approval)

## Convenience Packages

Besides the official releases, various convenience methods are available:

*   [PyPI releases](https://pypi.org/project/apache-airflow/) via `pip`.
*   [Docker Images](https://hub.docker.com/r/apache/airflow) via `docker`.
*   [Tags in GitHub](https://github.com/apache/airflow/tags).

## User Interface

Airflow offers a rich user interface for managing and monitoring your workflows:

*   **DAGs**: Overview of all DAGs.

    ![DAGs](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/dags.png)

*   **Assets**: Overview of Assets with dependencies.

    ![Asset Dependencies](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/assets_graph.png)

*   **Grid**: Grid representation of a DAG that spans across time.

    ![Grid](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/grid.png)

*   **Graph**: Visualization of a DAG's dependencies and their status.

    ![Graph](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/graph.png)

*   **Home**: Summary statistics of your Airflow environment.

    ![Home](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/home.png)

*   **Backfill**: Backfilling a DAG for a specific date range.

    ![Backfill](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/backfill.png)

*   **Code**: Quick view of DAG source code.

    ![Code](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/code.png)

## Semantic Versioning

Apache Airflow follows a strict [SemVer](https://semver.org/) approach.

## Version Life Cycle

Apache Airflow's version life cycle:

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

## Support for Python and Kubernetes Versions

See the detailed policy for Python and Kubernetes support in [the documentation](https://airflow.apache.org/docs/apache-airflow/stable/index.html).

## Base OS Support for Reference Airflow Images

Learn about Debian-based container images provided by the Airflow Community in [the documentation](https://airflow.apache.org/docs/docker-stack/index.html).

## Approach to Dependencies of Airflow

Airflow's dependency management is handled through constraints, as detailed in [Approach to Dependencies of Airflow](https://github.com/apache/airflow#approach-to-dependencies-of-airflow).

## Contributing

Join the Apache Airflow community!  Find out how to contribute via the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst).

## Voting Policy

* Commits need a +1 vote from a committer who is not the author
* When we do AIP voting, both PMC member's and committer's `+1s` are considered a binding vote.

## Who Uses Apache Airflow?

Apache Airflow is used by many organizations, see the list [in the wild](https://github.com/apache/airflow/blob/main/INTHEWILD.md).

## Who Maintains Apache Airflow?

Airflow is maintained by a dedicated [community](https://github.com/apache/airflow/graphs/contributors) of committers/maintainers.

## What Goes Into the Next Release?

Find out more about release processes and what goes into the next release by visiting the [What goes into the next release](dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md) document in the `dev` folder of the repository.

## Can I Use the Apache Airflow Logo in My Presentation?

Yes, see the [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook) for usage guidelines.

## Links

*   [Official Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Community Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)
*   [GitHub Repository](https://github.com/apache/airflow)

## Sponsors

Thank you to the organizations sponsoring the CI infrastructure for Apache Airflow:

<!-- Ordered by most recently "funded" -->

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>