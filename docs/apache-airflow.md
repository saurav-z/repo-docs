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
# Apache Airflow: Orchestrate Your Workflows with Code

**Apache Airflow** is an open-source platform that helps you programmatically author, schedule, and monitor your workflows, making data pipelines more manageable and efficient. ([Back to Original Repo](https://github.com/apache/airflow))

[![License](https://img.shields.io/:license-Apache%202-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0.txt)
[![PyPI version](https://badge.fury.io/py/apache-airflow.svg)](https://badge.fury.io/py/apache-airflow) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/apache-airflow.svg)](https://pypi.org/project/apache-airflow/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/apache-airflow)](https://pypi.org/project/apache-airflow/)
[![Docker Pulls](https://img.shields.io/docker/pulls/apache/airflow.svg)](https://hub.docker.com/r/apache/airflow) [![Docker Stars](https://img.shields.io/docker/stars/apache/airflow.svg)](https://hub.docker.com/r/apache/airflow) [![Artifact HUB](https://img.shields.io/endpoint?url=https://artifacthub.io/badge/repository/apache-airflow)](https://artifacthub.io/packages/search?repo=apache-airflow)
[![Contributors](https://img.shields.io/github/contributors/apache/airflow)](https://github.com/apache/airflow/graphs/contributors) [![Slack Status](https://img.shields.io/badge/slack-join_chat-white.svg?logo=slack&style=social)](https://s.apache.org/airflow-slack) ![Commit Activity](https://img.shields.io/github/commit-activity/m/apache/airflow) [![OSSRank](https://shields.io/endpoint?url=https://ossrank.com/shield/6)](https://ossrank.com/p/6)

[![GitHub Build main](https://github.com/apache/airflow/actions/workflows/ci-amd.yml/badge.svg)](https://github.com/apache/airflow/actions) [![GitHub Build main](https://github.com/apache/airflow/actions/workflows/ci-arm.yml/badge.svg)](https://github.com/apache/airflow/actions)
[![GitHub Build 3.1](https://github.com/apache/airflow/actions/workflows/ci-amd.yml/badge.svg?branch=v3-1-test)](https://github.com/apache/airflow/actions) [![GitHub Build 3.1](https://github.com/apache/airflow/actions/workflows/ci-arm.yml/badge.svg?branch=v3-1-test)](https://github.com/apache/airflow/actions)
[![GitHub Build 2.11](https://github.com/apache/airflow/actions/workflows/ci.yml/badge.svg?branch=v2-11-test)](https://github.com/apache/airflow/actions)

<picture width="500">
  <img
    src="https://github.com/apache/airflow/blob/19ebcac2395ef9a6b6ded3a2faa29dc960c1e635/docs/apache-airflow/img/logos/wordmark_1.png?raw=true"
    alt="Apache Airflow logo"
  />
</picture>

## Key Features

*   **Dynamic Pipelines:** Define workflows in code (DAGs) for dynamic generation and parameterization, promoting version control and maintainability.
*   **Extensible Architecture:** Leverage a wide array of built-in operators and easily extend Airflow to fit your specific needs.
*   **Flexible Templating:** Customize workflows with the Jinja templating engine.
*   **Scalable Scheduling & Execution**: Schedule and execute tasks across a distributed array of workers.
*   **Rich UI**: Visualize pipelines, monitor progress, and troubleshoot issues effectively through the user-friendly interface.
*   **Idempotent Tasks**: Airflow encourages tasks to be idempotent to prevent unintended data duplication, ensuring reliable task execution.
*   **XCom for Metadata**: Airflow's XCom feature allows tasks to pass metadata.

## Getting Started

*   **Install & Configure:** Refer to the [INSTALLING.md](INSTALLING.md) for local environment setup and installation guides.
*   **Official Documentation:** Explore the [official Airflow website](https://airflow.apache.org/docs/apache-airflow/stable/) for comprehensive installation, getting started guides, and tutorials.
*   **AIPs:** Dive into Airflow Improvement Proposals (AIPs) on the [Airflow Wiki](https://cwiki.apache.org/confluence/display/AIRFLOW/Airflow+Improvement+Proposals).
*   **Dependent Project Documentation:** Access documentation for provider distributions, Docker images, and Helm Charts through the [documentation index](https://airflow.apache.org/docs/).

## Installation

For detailed instructions on setting up your local development environment and installing Apache Airflow, see the [INSTALLING.md](INSTALLING.md) file.

## Official Source Code

Apache Airflow is a project of the [Apache Software Foundation](https://www.apache.org), with official source code releases adhering to the [ASF Release Policy](https://www.apache.org/legal/release-policy.html).
The source code can be downloaded from [the ASF Distribution Directory](https://downloads.apache.org/airflow) and are cryptographically signed by the release manager.

## Additional Resources
*   **Convenience Packages**: Various convenient ways of installing and using Airflow, including:
    *   [PyPI releases](https://pypi.org/project/apache-airflow/) to install Airflow using standard `pip` tool
    *   [Docker Images](https://hub.docker.com/r/apache/airflow) to install airflow via `docker` tool, use them in Kubernetes, Helm Charts, `docker-compose`, `docker swarm`, etc.
    *   [Tags in GitHub](https://github.com/apache/airflow/tags) to retrieve the git project sources that were used to generate official source packages via git

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


## Requirements

Airflow is tested on the following environments:

|            | Main version (dev)     | Stable version (3.0.6) |
|------------|------------------------|------------------------|
| Python     | 3.10, 3.11, 3.12, 3.13 | 3.9, 3.10, 3.11, 3.12  |
| Platform   | AMD64/ARM64(\*)        | AMD64/ARM64(\*)        |
| Kubernetes | 1.30, 1.31, 1.32, 1.33 | 1.30, 1.31, 1.32, 1.33 |
| PostgreSQL | 13, 14, 15, 16, 17     | 13, 14, 15, 16, 17     |
| MySQL      | 8.0, 8.4, Innovation   | 8.0, 8.4, Innovation   |
| SQLite     | 3.15.0+                | 3.15.0+                |

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

## Installing from PyPI

Install Apache Airflow using `pip`, and ensure the correct dependencies with the constraints files.

1.  **Install Airflow**:

```bash
pip install 'apache-airflow==3.0.6' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
```

2.  **Install with Extras**:

```bash
pip install 'apache-airflow[postgres,google]==3.0.6' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
```

## Semantic Versioning

Airflow adheres to [SemVer](https://semver.org/) principles.

## Version Life Cycle

| Version   | Current Patch/Minor   | State     | First Release   | Limited Maintenance   | EOL/Terminated   |
|-----------|-----------------------|-----------|-----------------|-----------------------|------------------|
| 3         | 3.0.6                 | Supported | Apr 22, 2025    | TBD                   | TBD              |
| 2         | 2.11.0                | Supported | Dec 17, 2020    | Oct 22, 2025          | Apr 22, 2026     |
| 1.10      | 1.10.15               | EOL       | Aug 27, 2018    | Dec 17, 2020          | June 17, 2021    |
| 1.9       | 1.9.0                 | EOL       | Jan 03, 2018    | Aug 27, 2018          | Aug 27, 2018     |
| 1.8       | 1.8.2                 | EOL       | Mar 19, 2017    | Jan 03, 2018          | Jan 03, 2018     |
| 1.7       | 1.7.1.2               | EOL       | Mar 28, 2016    | Mar 19, 2017          | Mar 19, 2017     |

## Support for Python and Kubernetes versions
As of Airflow 2.0, we agreed to certain rules we follow for Python and Kubernetes support, based on the official release schedule and version skew policies.

## Base OS support for reference Airflow images
Airflow provides container images that include Debian OS, Python, databases, pre-defined providers.

## Approach to dependencies of Airflow
Airflow uses constraint files to ensure repeatable installations while allowing users to upgrade dependencies. Dependencies are not upper-bound unless there are good reasons to do so.

## Contributing

Contribute to Apache Airflow!  Review the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst) for guidance.

## Voting Policy

* Commits require a +1 vote from a committer who is not the author.
* AIP voting: PMC and committer +1s are binding.

## Who Uses Apache Airflow?

[Many organizations](https://github.com/apache/airflow/blob/main/INTHEWILD.md) use Apache Airflow for workflow management. Add your organization to the list with a PR.

## Who Maintains Apache Airflow?

The Airflow community drives the project, with [core committers/maintainers](https://people.apache.org/committers-by-project.html#airflow) reviewing PRs.  See the Apache Airflow [committer requirements](https://github.com/apache/airflow/blob/main/COMMITTERS.rst#guidelines-to-become-an-airflow-committer) if you are interested in becoming a maintainer.

## What goes into the next release?
The answer to which release the merged PR(s) will be released in or which release the fixed issues will be in, depends on various scenarios. The answer is different for PRs and Issues.
More context and **FAQ** about the patchlevel release can be found in the
[What goes into the next release](dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md) document in the `dev` folder of the
repository.

## Can I use the Apache Airflow logo in my presentation?

Yes, follow the Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook) to use the logo.

## Links

-   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
-   [Chat](https://s.apache.org/airflow-slack)
-   [Community Information](https://airflow.apache.org/community/)

## Sponsors

*   <a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
*   <a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>
<!-- END Apache Airflow, please keep comment here to allow auto update of PyPI readme.md -->