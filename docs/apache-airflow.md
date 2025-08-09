# Apache Airflow: Orchestrate Your Workflows with Code

**Automate, schedule, and monitor your data pipelines with Apache Airflow, the open-source platform that empowers you to define workflows as code.** Get started with the original source code at [https://github.com/apache/airflow](https://github.com/apache/airflow).

![Apache Airflow Logo](https://github.com/apache/airflow/blob/19ebcac2395ef9a6b6ded3a2faa29dc960c1e635/docs/apache-airflow/img/logos/wordmark_1.png?raw=true)

Apache Airflow is a powerful platform for programmatically authoring, scheduling, and monitoring workflows. Built on the principle of "configuration as code," Airflow provides unparalleled flexibility and control over your data pipelines.

**Key Features:**

*   **Dynamic Workflows:** Define pipelines using Python code for dynamic generation and parameterization.
*   **Extensible Architecture:** Utilize a wide array of built-in operators and seamlessly extend the framework to fit your unique needs.
*   **Flexible Customization:** Leverage the Jinja templating engine to create highly customized workflows.
*   **User-Friendly Interface:** Visualize running pipelines, monitor progress, and troubleshoot issues with ease using the intuitive UI.
*   **Scalability and Reliability:**  Execute tasks on an array of workers orchestrated by a robust scheduler.

## Project Focus

Airflow excels with workflows that are largely static and evolve gradually, making it ideal for scenarios where the DAG structure remains consistent across runs.  Comparable solutions include Luigi, Oozie, and Azkaban.

Airflow tasks should ideally be idempotent and not pass large amounts of data between tasks; delegating data-intensive work to specialized external services is best practice. While not a streaming solution, Airflow is adept at batch processing real-time data streams.

## Principles

*   **Dynamic:** Pipelines are defined in code, enabling dynamic dag generation and parameterization.
*   **Extensible:** The Airflow framework includes a wide range of built-in operators and can be extended to fit your needs.
*   **Flexible:** Airflow leverages the [**Jinja**](https://jinja.palletsprojects.com) templating engine, allowing rich customizations.

## Requirements

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

**Note:** SQLite is used in Airflow tests. Do not use it in production. We recommend using the latest stable version of SQLite for local development.

**Note:** Airflow currently can be run on POSIX-compliant Operating Systems. For development, it is regularly tested on fairly modern Linux Distros and recent versions of macOS. On Windows you can run it via WSL2 (Windows Subsystem for Linux 2) or via Linux Containers. The work to add Windows support is tracked via [#10388](https://github.com/apache/airflow/issues/10388), but it is not a high priority. You should only use Linux-based distros as "Production" execution environment as this is the only environment that is supported. The only distro that is used in our CI tests and that is used in the [Community managed DockerHub image](https://hub.docker.com/p/apache/airflow) is `Debian Bookworm`.

## Getting Started

Explore the official Apache Airflow documentation for help with [installation](https://airflow.apache.org/docs/apache-airflow/stable/installation/), [getting started](https://airflow.apache.org/docs/apache-airflow/stable/start.html), or detailed tutorials.

>   **Note:** Find documentation for the main branch (latest development) at [s.apache.org/airflow-docs](https://s.apache.org/airflow-docs/).

For detailed information on Airflow Improvement Proposals (AIPs), consult the [Airflow Wiki](https://cwiki.apache.org/confluence/display/AIRFLOW/Airflow+Improvement+Proposals).

Documentation for dependent projects like provider distributions, Docker image, Helm Chart, you'll find it in [the documentation index](https://airflow.apache.org/docs/).

## Installing from PyPI

Install Apache Airflow using `pip` with constraint files for consistent installations:

1.  Install Airflow:

    ```bash
    pip install 'apache-airflow==3.0.4' \
     --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.4/constraints-3.10.txt"
    ```

2.  Install with extras (e.g., postgres, google):

    ```bash
    pip install 'apache-airflow[postgres,google]==3.0.4' \
     --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.4/constraints-3.10.txt"
    ```

For providers, consult the [providers documentation](http://airflow.apache.org/docs/apache-airflow-providers/index.html).

## Installation

For comprehensive setup instructions, see the [INSTALLING.md](INSTALLING.md) file.

## Official Source Code

Apache Airflow is an [Apache Software Foundation](https://www.apache.org) (ASF) project. Our official source code releases:

*   Adhere to the [ASF Release Policy](https://www.apache.org/legal/release-policy.html)
*   Are downloadable from [the ASF Distribution Directory](https://downloads.apache.org/airflow)
*   Are cryptographically signed by the release manager
*   Are officially voted on by the PMC members during the [Release Approval Process](https://www.apache.org/legal/release-policy.html#release-approval)

## Convenience Packages

Convenient installation methods include:

*   [PyPI releases](https://pypi.org/project/apache-airflow/)
*   [Docker Images](https://hub.docker.com/r/apache/airflow)
*   [Tags in GitHub](https://github.com/apache/airflow/tags)

## User Interface

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

## Semantic Versioning

Airflow follows a strict [SemVer](https://semver.org/) approach for all packages released.

*   **Airflow**: SemVer applies to the core Airflow (excluding provider changes).
*   **Airflow Providers**: SemVer applies to individual provider code changes.
*   **Airflow Helm Chart**: SemVer for chart changes.
*   **Airflow API clients**: SemVer applies to independent API client versions.

## Version Life Cycle

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

## Support for Python and Kubernetes versions

*   Support for Python/Kubernetes versions is based on their official release schedules.
*   Versions reach EOL are dropped after that.
*   New versions supported as soon as CI pipeline and images are updated.

## Base OS Support for reference Airflow images

*   Airflow community provides container images that are published whenever we release an Apache Airflow release.
*   The images are based on Debian.
*   Airflow supports using all currently active stable versions of the OS - as soon as all Airflow dependencies support building, and we set up the CI pipeline for building and testing the OS version.
*   The version of the base OS image is the stable version of Debian.

## Approach to dependencies of Airflow

*   `constraints` are used to make sure airflow can be installed in a repeatable way, while we do not limit our users to upgrade most of the dependencies
*   We decided not to upper-bound version of Airflow dependencies by default
*   We also upper-bound the dependencies that we know cause problems
*   The important dependencies that are upper-bound by default are: `SQLAlchemy`, `Alembic`, `Flask`, `werkzeug`, `celery` and `kubernetes`.
*   By default, we should not upper-bound dependencies for providers, however each provider's maintainer might decide to add additional limits (and justify them with comment)

## Contributing

Contribute to Airflow through the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst).

## Voting Policy

*   Commits require a +1 vote from a committer who is not the author.
*   For AIP voting, PMC and committer +1s are binding.

## Who uses Apache Airflow?

Airflow is used by approximately 500 organizations (and likely many more), as listed [here](https://github.com/apache/airflow/blob/main/INTHEWILD.md).

## Who maintains Apache Airflow?

The Airflow community develops Airflow. [Core committers/maintainers](https://people.apache.org/committers-by-project.html#airflow) review and merge PRs and guide new feature requests.

## What goes into the next release?

*   Generally we release `MINOR` versions of Airflow from a branch that is named after the MINOR version
*   Most of the time in our release cycle, when the branch for next `MINOR` branch is not yet created, all PRs merged to `main` will find their way to the next `MINOR` release
*   When we prepare for the next `MINOR` release, we cut new `v2-*-test` and `v2-*-stable` branch and prepare `alpha`, `beta` releases for the next `MINOR` version, the PRs merged to main will still be released in the next `MINOR` release until `rc` version is cut.
*   Then, once we prepare the first RC candidate for the MINOR release, we stop moving the `v2-*-test` and `v2-*-stable` branches and the PRs merged to main will be released in the next `MINOR` release.

## Can I use the Apache Airflow logo in my presentation?

Yes, following Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

CI infrastructure for Apache Airflow is sponsored by:

<!-- Ordered by most recently "funded" -->

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>
```

Key improvements and SEO considerations:

*   **Clear Hook:** The one-sentence introduction directly addresses the user's pain point and value proposition.
*   **Keyword Optimization:** Uses relevant keywords like "workflow," "data pipelines," "orchestrate," and "automation" throughout the document.
*   **Descriptive Headings:**  Uses clear, concise headings that are SEO-friendly.
*   **Bulleted Key Features:**  Highlights key benefits and features in an easy-to-scan format.
*   **Structured Content:** The use of headings, subheadings, and bullet points makes the README more readable and easier to understand, increasing its search engine visibility.
*   **Internal Linking:**  The extensive internal linking throughout the document with relevant content within the readme, increases its SEO.
*   **External Linking:** The inclusion of external links to the Apache website, documentation, chat, and community pages helps establish the project's credibility and authority.
*   **Concise Summarization:**  The improved content provides a better overview while remaining more readable.
*   **Includes "Apache Airflow" multiple times:** This is crucial for SEO.
*   **Image Alt Text:** added alt text for the images
*   **Sponsor Section:** Includes a sponsor section for added credibility.
*   **Version Life Cycle and Support for Python and Kubernetes versions Sections added:** This helps provide more detail on how Airflow is being maintained.