# Apache Airflow: Orchestrate Your Workflows with Code

**Apache Airflow is a powerful, open-source platform that allows you to programmatically author, schedule, and monitor your data pipelines, making them more maintainable, testable, and collaborative.** Learn more about [Apache Airflow](https://github.com/apache/airflow).

[![License](https://img.shields.io/:license-Apache%202-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0.txt)
[![PyPI version](https://badge.fury.io/py/apache-airflow.svg)](https://badge.fury.io/py/apache-airflow)
[![Docker Pulls](https://img.shields.io/docker/pulls/apache/airflow.svg)](https://hub.docker.com/r/apache/airflow)
[![Contributors](https://img.shields.io/github/contributors/apache/airflow)](https://github.com/apache/airflow/graphs/contributors)
[![Slack Status](https://img.shields.io/badge/slack-join_chat-white.svg?logo=slack&style=social)](https://s.apache.org/airflow-slack)
[![GitHub Build main](https://github.com/apache/airflow/actions/workflows/ci-amd.yml/badge.svg)](https://github.com/apache/airflow/actions)
<picture width="500">
  <img
    src="https://github.com/apache/airflow/blob/19ebcac2395ef9a6b6ded3a2faa29dc960c1e635/docs/apache-airflow/img/logos/wordmark_1.png?raw=true"
    alt="Apache Airflow logo"
  />
</picture>

**Key Features:**

*   **Workflow as Code:** Define workflows (DAGs) in Python for maintainability, version control, and collaboration.
*   **Scheduling and Execution:** The Airflow scheduler executes tasks on a distributed array of workers, adhering to dependencies.
*   **Rich UI:** A user-friendly interface to visualize pipelines, monitor progress, and troubleshoot issues.
*   **Extensible:** Extend Airflow with custom operators and plugins to integrate with various systems.
*   **Dynamic:** Leverages Jinja templating engine for rich customizations.

### **Project Focus**

Airflow is ideal for workflows with relatively static structures and slowly changing DAGs. It excels at orchestrating tasks, especially those that are idempotent and delegate high-volume, data-intensive processes to specialized external services.

### **Principles**

*   **Dynamic:** Pipelines are defined in code, enabling dynamic dag generation and parameterization.
*   **Extensible:** The Airflow framework includes a wide range of built-in operators and can be extended to fit your needs.
*   **Flexible:** Airflow leverages the [**Jinja**](https://jinja.palletsprojects.com) templating engine, allowing rich customizations.

### **Requirements**

Apache Airflow is tested with:

|            | Main version (dev)     | Stable version (3.0.4) |
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

### **Getting Started**

Get started with Airflow by visiting the official documentation for [installing Airflow](https://airflow.apache.org/docs/apache-airflow/stable/installation/), [getting started](https://airflow.apache.org/docs/apache-airflow/stable/start.html), and walking through the [tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/).

### **Installing from PyPI**

Install Apache Airflow using `pip`, ensuring to follow the constraint files for repeatable installations:

```bash
pip install 'apache-airflow==3.0.4' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.4/constraints-3.10.txt"
```

Install with extras (i.e., postgres, google):

```bash
pip install 'apache-airflow[postgres,google]==3.0.4' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.4/constraints-3.10.txt"
```

### **Installation**

For comprehensive instructions on setting up your local development environment and installing Apache Airflow, please refer to the [INSTALLING.md](INSTALLING.md) file.

### **Official Source Code**

Airflow is an Apache Software Foundation (ASF) project, and official source code releases adhere to the ASF Release Policy. Download from the [ASF Distribution Directory](https://downloads.apache.org/airflow).

### **Convenience packages**

There are other ways of installing and using Airflow, including PyPI releases, Docker Images and git Tags.
The different artifacts are not "official releases" as stated by the `ASF Release Policy`.

### **User Interface**

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

### **Semantic Versioning**

Airflow 2.0.0 and later versions use [SemVer](https://semver.org/).

### **Version Life Cycle**

Apache Airflow version life cycle:

<!-- This table is automatically updated by pre-commit scripts/ci/pre_commit/supported_versions.py -->
<!-- Beginning of auto-generated table -->

| Version   | Current Patch/Minor   | State     | First Release   | Limited Maintenance   | EOL/Terminated   |
|-----------|-----------------------|-----------|-----------------|-----------------------|------------------|
| 3         | 3.0.4                 | Supported | Apr 22, 2025    | TBD                   | TBD              |
| 2         | 2.11.0                | Supported | Dec 17, 2020    | Oct 22, 2025          | Apr 22, 2026     |
| 1.10      | 1.10.15               | EOL       | Aug 27, 2018    | Dec 17, 2020          | June 17, 2021    |
| 1.9       | 1.9.0                 | EOL       | Jan 03, 2018    | Aug 27, 2018          | Aug 27, 2018     |
| 1.8       | 1.8.2                 | EOL       | Mar 19, 2017    | Jan 03, 2018          | Jan 03, 2018     |
| 1.7       | 1.7.1.2               | EOL       | Mar 28, 2016    | Mar 19, 2017          | Mar 19, 2017     |

<!-- End of auto-generated table -->

### **Support for Python and Kubernetes versions**

Support for Python and Kubernetes versions is based on their official release schedules.

### **Base OS support for reference Airflow images**

The Airflow Community provides conveniently packaged container images that are published whenever
we publish an Apache Airflow release. Those images contain stable Debian OS and libraries to connect
to supported Databases and pre-defined set of popular providers.

### **Approach to dependencies of Airflow**

Airflow uses `constraints` for repeatable installations, while allowing users to upgrade most dependencies.

### **Contributing**

Contribute to Apache Airflow by following the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst).

### **Voting Policy**

*   Commits require a +1 from a non-author committer.
*   AIP voting: +1s from PMC members and committers are binding.

### **Who uses Apache Airflow?**

Around 500 organizations use Apache Airflow, see [INTHEWILD.md](https://github.com/apache/airflow/blob/main/INTHEWILD.md).

### **Who maintains Apache Airflow?**

Airflow is maintained by a [community](https://github.com/apache/airflow/graphs/contributors) of contributors, with core committers/maintainers responsible for reviewing PRs and steering new feature requests.  See [committer requirements](https://github.com/apache/airflow/blob/main/COMMITTERS.rst#guidelines-to-become-an-airflow-committer) to become a maintainer.

### **What goes into the next release?**

The PRs merged to main will find their way to the next `MINOR` release. The next `MINOR` release is when the branch for the next `MINOR` branch is created. The versioning scheme can be found in [What goes into the next release](dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md) document in the `dev` folder of the repository.

### **Can I use the Apache Airflow logo in my presentation?**

Yes, abide by the Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook).

### **Links**

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

### **Sponsors**

<!-- Ordered by most recently "funded" -->

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>