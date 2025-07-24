# Apache Airflow: Orchestrate Your Workflows with Code

**Apache Airflow** is a powerful platform for programmatically authoring, scheduling, and monitoring workflows, allowing you to define, manage, and track complex data pipelines with ease. Visit the [original repo](https://github.com/apache/airflow) for more information.

Key Features:

*   **Dynamic Workflows as Code:** Define your pipelines using Python code for maintainability, version control, and testability.
*   **Extensible Architecture:** Leverage a rich set of built-in operators and customize Airflow to fit your unique needs.
*   **Flexible Scheduling:** Schedule tasks with precision and monitor their execution through a comprehensive UI.
*   **Scalable and Reliable:** Designed to handle complex workflows and scale to meet your growing data processing demands.

*   **Rich User Interface**: Visualize pipelines running in production, monitor progress, and troubleshoot issues when needed.

## Why Choose Apache Airflow?

Apache Airflow empowers you to build robust and scalable data pipelines. Defining workflows as code makes them versionable and testable, and makes collaboration easier.

## Core Principles

-   **Dynamic**: Pipelines are defined in code, enabling dynamic dag generation and parameterization.
-   **Extensible**: The Airflow framework includes a wide range of built-in operators and can be extended to fit your needs.
-   **Flexible**: Airflow leverages the [**Jinja**](https://jinja.palletsprojects.com) templating engine, allowing rich customizations.

## Key Features

*   **Workflow Authoring:** Create workflows (DAGs) that orchestrate tasks.
*   **Scheduling and Execution:**  The Airflow scheduler executes tasks on a worker pool, adhering to defined dependencies.
*   **Monitoring and Management:**  Use the rich user interface to visualize pipelines, monitor progress, and troubleshoot issues.
*   **Idempotent Tasks:** Encourage idempotent tasks for data integrity.
*   **XCom Feature:** Tasks can pass metadata using Airflow's [XCom feature](https://airflow.apache.org/docs/apache-airflow/stable/concepts/xcoms.html).

## Installation and Setup

For comprehensive instructions on setting up your local development environment and installing Apache Airflow, please refer to the [INSTALLING.md](INSTALLING.md) file.

## Installing from PyPI

Install Apache Airflow using pip, including optional extras such as providers, such as `postgres` and `google`.

```bash
pip install 'apache-airflow[postgres,google]==3.0.3' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.3/constraints-3.10.txt"
```

## Community and Resources

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Contributing Guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst)

## Sponsors

The CI infrastructure for Apache Airflow has been sponsored by:

<!-- Ordered by most recently "funded" -->

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>