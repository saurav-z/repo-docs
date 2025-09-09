<div align="center">
<a href ="https://hatchet.run?utm_source=github&utm_campaign=readme">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./frontend/docs/public/hatchet_logo.png">
  <img width="200" alt="Hatchet Logo" src="./frontend/docs/public/hatchet_logo_light.png">
</picture>
</a>

</div>

## Hatchet: Run Background Tasks at Scale 🚀

**Hatchet is a robust background task management platform built on Postgres, simplifying the execution of complex workflows and enhancing application reliability.**  [Learn more at the Hatchet GitHub repository.](https://github.com/hatchet-dev/hatchet)

[![Docs](https://img.shields.io/badge/docs-docs.hatchet.run-3F16E4)](https://docs.hatchet.run)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
[![Go Reference](https://pkg.go.dev/badge/github.com/hatchet-dev/hatchet.svg)](https://pkg.go.dev/github.com/hatchet-dev/hatchet)
[![NPM Downloads](https://img.shields.io/npm/dm/%40hatchet-dev%2Ftypescript-sdk)](https://www.npmjs.com/package/@hatchet-dev/typescript-sdk)
[![Discord](https://img.shields.io/discord/1088927970518909068?style=social&logo=discord)](https://hatchet.run/discord)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/hatchet-dev.svg?style=social&label=Follow%20%40hatchet-dev)](https://twitter.com/hatchet_dev)
[![GitHub Repo stars](https://img.shields.io/github/stars/hatchet-dev/hatchet?style=social)](https://github.com/hatchet-dev/hatchet)

  <p align="center">
    <a href="https://cloud.onhatchet.run">Hatchet Cloud</a>
    ·
    <a href="https://docs.hatchet.run">Documentation</a>
    ·
    <a href="https://hatchet.run">Website</a>
    ·
    <a href="https://github.com/hatchet-dev/hatchet/issues">Issues</a>
  </p>

### Key Features

*   **📥 Queues:** Reliable task queues ensuring tasks are completed, even through application crashes. Provides:
    *   Ensures no user requests are dropped.
    *   Handles traffic spikes effectively.
    *   Breaks down complex logic into reusable tasks.

    *Code samples are available in Python, Typescript, and Go.*

*   **🎻 Task Orchestration:** Build complex workflows with multiple tasks using:
    *   **DAGs (Directed Acyclic Graphs):** Pre-define work structure.
    *   **Durable tasks:** Stores full histories and caches intermediate results.

    *Code samples are available in Python, Typescript, and Go.*

*   **🚦 Flow Control:** Manage and control task execution for increased system stability. Features:
    *   **Concurrency limits:** Set limits based on a dynamic key (e.g., user ID).
    *   **Rate limiting:** Implement global and dynamic rate limits.

    *Code samples are available in Python, Typescript, and Go.*

*   **📅 Scheduling:** Schedule tasks for various needs. Offers:
    *   **Cron schedules:** Runs tasks on a schedule.
    *   **One-time tasks:** Schedule tasks for a specific time.
    *   **Durable sleep:** Pause tasks for specific durations.

    *Code samples are available in Python, Typescript, and Go.*

*   **🚏 Task Routing:** Utilize flexible task routing mechanisms:
    *   **Sticky assignment:** tasks that prefer to run on the same worker.
    *   **Worker affinity:** Assigns tasks to best-suited workers.

    *Code samples are available in Python, Typescript, and Go.*

*   **⚡️ Event Triggers and Listeners:** Implement event-driven architectures with features like:
    *   **Event listening:** Pause tasks until a specific event occurs.
    *   **Event triggering:** Trigger workflows based on events.

    *Code samples are available in Python, Typescript, and Go.*

*   **🖥️ Real-time Web UI:** Monitor tasks, workflows, and queues with:
    *   Real-time dashboards and metrics
    *   Logging capabilities
    *   Alerting via Slack and email.

### Quick Start

Choose your deployment:

*   [Hatchet Cloud Quickstart](https://docs.hatchet.run/home/hatchet-cloud-quickstart)
*   [Hatchet Self-Hosted](https://docs.hatchet.run/self-hosting)

### Documentation

Refer to the [official documentation](https://docs.hatchet.run) for comprehensive information.

### Community & Support

*   [Discord](https://discord.gg/ZMeUafwH89) - Join the community and connect with maintainers.
*   [GitHub Issues](https://github.com/hatchet-dev/hatchet/issues) - Report bugs.
*   [GitHub Discussions](https://github.com/hatchet-dev/hatchet/discussions) - Engage in technical discussions.
*   [Email](mailto:contact@hatchet.run) - Cloud support, billing, and data deletion inquiries.

### Hatchet vs... (Comparisons)

*   <details>
    <summary>Hatchet vs Temporal</summary>
    Hatchet offers wider use-cases like queueing and DAG orchestration compared to Temporal's durable execution focus.
    </details>

*   <details>
    <summary>Hatchet vs Task Queues (BullMQ, Celery)</summary>
    Hatchet provides durable task history and monitoring capabilities not standard in BullMQ/Celery.
    </details>

*   <details>
    <summary>Hatchet vs DAG-based platforms (Airflow, Prefect, Dagster)</summary>
    Hatchet is optimized for high-throughput applications, making it a good choice if you need to write custom integrations.
    </details>

*   <details>
    <summary>Hatchet vs AI Frameworks</summary>
    Hatchet offers control, high availability, and durability for AI functions, unlike many AI frameworks.
    </details>

### Issues

Submit any bugs on [GitHub Issues](https://github.com/hatchet-dev/hatchet/issues).

### Contribute

Engage in the [#contributing](https://discord.gg/ZMeUafwH89) channel on Discord to shape project direction.
```
Key improvements and SEO optimizations:

*   **Clear Headline and Hook:** Starts with a concise one-sentence introduction, summarizing what Hatchet does and its core value proposition.
*   **SEO Keywords:** Includes relevant keywords such as "background tasks," "task orchestration," "task queue," and "workflow management" to improve search visibility.
*   **Structured Content:** Uses clear headings, bullet points, and concise descriptions to make the content easy to scan and understand.
*   **Emphasis on Benefits:** Highlights the benefits of using Hatchet.
*   **Call to Action:** Encourages users to learn more.
*   **Clean Formatting:** Utilizes Markdown for better readability and rendering on various platforms.
*   **Concise Language:** Streamlines the language for clarity and readability.
*   **Link to Original Repo:**  Included at the beginning for easy access.
*   **Summarized Sections**: Instead of copying large code blocks, it provides summary and mentions the availability of the sample in languages like Python, Typescript, and Go.