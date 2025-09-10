<div align="center">
<a href ="https://hatchet.run?utm_source=github&utm_campaign=readme">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./frontend/docs/public/hatchet_logo.png">
  <img width="200" alt="Hatchet Logo" src="./frontend/docs/public/hatchet_logo_light.png">
</picture>
</a>
</div>

## Hatchet: Run Background Tasks at Scale

**Simplify background task management and build robust workflows with Hatchet, a powerful and scalable platform built on Postgres.**  [Go to the Hatchet Repository](https://github.com/hatchet-dev/hatchet)

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

### Key Features of Hatchet

*   **📥 Queues:** Reliable task queues ensure your work gets done, even with application crashes, managing tasks efficiently and handling traffic spikes.
    *   Supports Python, Typescript, and Go. (with example code snippets)
    *   Ensures that you never drop a user request.
    *   Helps flatten large spikes in your application.
    *   Breaks down complex logic into smaller, reusable tasks.
    *   [Learn more](https://docs.hatchet.run/home/your-first-task)

*   **🎻 Task Orchestration:** Build complex workflows using DAGs and durable tasks to chain multiple tasks.
    *   Supports Python, Typescript, and Go. (with example code snippets)
    *   **DAGs (directed acyclic graphs):** Pre-define the shape of your work.
    *   **Durable tasks:** These tasks are responsible for orchestrating other tasks.
    *   [Learn more](https://docs.hatchet.run/home/dags) and [Durable Execution](https://docs.hatchet.run/home/durable-execution)

*   **🚦 Flow Control:** Control task execution with concurrency and rate limiting.
    *   Supports Python, Typescript, and Go. (with example code snippets)
    *   **Concurrency:** Limit tasks based on dynamic keys (e.g., user IDs).
    *   **Rate limiting:** Apply global and dynamic rate limits.
    *   [Learn more](https://docs.hatchet.run/home/concurrency) and [Rate Limits](https://docs.hatchet.run/home/rate-limits)

*   **📅 Scheduling:** Schedule tasks using cron, one-time scheduling, or duration-based pausing.
    *   Supports Python, Typescript, and Go. (with example code snippets)
    *   **Cron schedules:** Run tasks on a scheduled basis.
    *   **One-time tasks:** Schedule workflows for a specific future time.
    *   **Durable sleep:** Pause task execution for a specific duration.
    *   [Learn more](https://docs.hatchet.run/home/cron-runs), [Scheduled Runs](https://docs.hatchet.run/home/scheduled-runs) and [Durable Execution](https://docs.hatchet.run/home/durable-execution)

*   **🚏 Task Routing:** Define task placement with sticky assignment or worker affinity.
    *   Supports Python, Typescript, and Go. (with example code snippets)
    *   **Sticky assignment:** Prefer or require tasks to run on the same worker.
    *   **Worker affinity:** Prioritize workers for optimal task handling.
    *   [Learn more](https://docs.hatchet.run/home/sticky-assignment) and [Worker Affinity](https://docs.hatchet.run/home/worker-affinity)

*   **⚡️ Event Triggers and Listeners:** Create event-driven architectures to start tasks and workflows based on triggers.
    *   Supports Python, Typescript, and Go. (with example code snippets)
    *   **Event listening:** Tasks can pause, waiting for events.
    *   **Event triggering:** Events launch workflows or workflow steps.
    *   [Learn more](https://docs.hatchet.run/home/durable-execution) and [Run on Event](https://docs.hatchet.run/home/run-on-event)

*   **🖥️ Real-time Web UI:**
    *   Monitor tasks, workflows, and queues with real-time dashboards and metrics.
    *   Built-in alerting for quick issue detection.
    *   Logging from tasks to correlate failures with system logs.

### Quick Start

Get started with Hatchet quickly through the cloud or self-hosting.

*   [Hatchet Cloud Quickstart](https://docs.hatchet.run/home/hatchet-cloud-quickstart)
*   [Hatchet Self-Hosted](https://docs.hatchet.run/self-hosting)

### Documentation

Comprehensive documentation is available at: https://docs.hatchet.run.

### Community & Support

*   [Discord](https://discord.gg/ZMeUafwH89) - Best for interacting with maintainers and community.
*   [Github Issues](https://github.com/hatchet-dev/hatchet/issues) - For bug reports.
*   [Github Discussions](https://github.com/hatchet-dev/hatchet/discussions) - For in-depth technical discussions.
*   [Email](mailto:contact@hatchet.run) - For Hatchet Cloud support, billing inquiries, and data deletion requests.

### Hatchet vs. Other Technologies

Detailed comparisons to help you choose the right tool for your needs:

*   **Hatchet vs. Temporal:** Hatchet covers a wider array of use-cases, Temporal is narrowly focused on durable execution.
*   **Hatchet vs. Task Queues (BullMQ, Celery):** Hatchet is a durable task queue.
*   **Hatchet vs. DAG-based platforms (Airflow, Prefect, Dagster):** Hatchet is designed to run as part of a high-volume application.
*   **Hatchet vs. AI Frameworks:** Hatchet offers greater function control, high availability and durability.

### Issues

Report bugs via [GitHub issues](https://github.com/hatchet-dev/hatchet/issues).

### Contribute

Join the Hatchet community! Let us know what you're interested in contributing to in the `#contributing` channel on [Discord](https://discord.gg/ZMeUafwH89).