# Apache Airflow: The Leading Platform for Data Orchestration

**Apache Airflow** is the leading open-source platform that empowers you to programmatically author, schedule, and monitor data workflows, making complex data pipelines manageable and efficient. [Explore the Airflow Repository](https://github.com/apache/airflow) for more information.

**Key Features:**

*   **Dynamic Workflows:** Define pipelines as code for version control, testing, and collaboration.
*   **Extensible Architecture:** Leverage a rich set of built-in operators and easily customize Airflow to fit your specific needs.
*   **Flexible Scheduling & Monitoring:** Schedule tasks, monitor progress, and troubleshoot issues through a user-friendly UI.
*   **Scalable and Reliable:** Built to handle complex workflows and data-intensive tasks, with a robust scheduler and worker architecture.
*   **Idempotent Task Execution:** Airflow promotes idempotent tasks for reliable data processing.

**Benefits:**

*   **Increased Productivity:** Automate data pipeline execution and reduce manual intervention.
*   **Improved Reliability:** Ensure data integrity and consistency through robust scheduling and monitoring.
*   **Enhanced Collaboration:** Facilitate team collaboration and streamline workflow management.
*   **Scalability:** Handles workflows of any size.

**Key Features in a Nutshell:**

*   Define Workflows as Code: Use Python to describe your pipelines.
*   Schedule and Orchestrate: Airflow takes care of running your tasks at the right time.
*   Monitor and Troubleshoot: A user-friendly interface lets you see what's going on.
*   Extensible and Flexible: Custom operators, integrations, and templating available.

**Getting Started:**

*   **Documentation:** Explore the [official Apache Airflow documentation](https://airflow.apache.org/docs/apache-airflow/stable/) for installation, tutorials, and in-depth information.

**Installation:**

*   Refer to the [INSTALLING.md](INSTALLING.md) file for detailed instructions on setting up your local development environment and installing Apache Airflow.
*   Install using `pip`:

```bash
pip install 'apache-airflow==3.0.6' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
```
*  Install with extras (postgres, google):
```bash
pip install 'apache-airflow[postgres,google]==3.0.6' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
```
*   **Official Source Code:**  Get the official source code from the [Apache Software Foundation](https://www.apache.org) [ASF Distribution Directory](https://downloads.apache.org/airflow).

**User Interface:**

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

**Requirements:**

*   **Python:** 3.10, 3.11, 3.12, 3.13
*   **Platform:** AMD64/ARM64
*   **Kubernetes:** 1.30, 1.31, 1.32, 1.33, 1.34
*   **Databases:** PostgreSQL, MySQL, SQLite

**Contributing:**

*   Contribute to the project via our [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst).

**More Information:**

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

**Sponsors:**

*   <a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
*   <a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>