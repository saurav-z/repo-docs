<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](https://github.com/ray-project/ray/graphs/contributors)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

# Ray: The Unified Framework for Scaling AI and Python Applications

**Ray is a powerful, open-source framework that simplifies scaling your Python and AI applications, from your laptop to a large cluster.**

[![Documentation Status](https://readthedocs.org/projects/ray/badge/?version=master)](http://docs.ray.io/en/master/?badge=master)
[![Slack](https://img.shields.io/badge/Ray-Join%20Slack-blue)](https://www.ray.io/join-slack)
[![Discuss](https://img.shields.io/badge/Discuss-Ask%20Questions-blue)](https://discuss.ray.io/)
[![Twitter](https://img.shields.io/twitter/follow/raydistributed.svg?style=social&logo=twitter)](https://x.com/raydistributed)
[![Get Started for Free](https://img.shields.io/badge/Get_started_for_free-3C8AE9?logo=data%3Aimage%2Fpng%3B64%2CiVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8%2F9hAAAAAXNSR0IArs4c6QAAAERlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMAAAABAAEAAKACAAQAAAABAAAAEKADAAQAAAABAAAAEAAAAAA0VXHyAAABKElEQVQ4Ea2TvWoCQRRGnWCVWChIIlikC9hpJdikSbGgaONbpAoY8gKBdAGfwkfwKQypLQ1sEGyMYhN1Pd%2B6A8PqwBZeOHt%2FvsvMnd3ZXBRFPQjBZ9K6OY8ZxF%2B0IYw9PW3qz8aY6lk92bZ%2BVqSI3oC9T7%2FyCVnrF1ngj93us%2B540sf5BrCDfw9b6jJ5lx%2FyjtGKBBXc3cnqx0INN4ImbI%2Bl%2BPnI8zWfFEr4chLLrWHCp9OO9j19Kbc91HX0zzzBO8EbLK2Iv4ZvNO3is3h6jb%2BCwO0iL8AaWqB7ILPTxq3kDypqvBuYuwswqo6wgYJbT8XxBPZ8KS1TepkFdC79TAHHce%2F7LbVioi3wEfTpmeKtPRGEeoldSP%2FOeoEftpP4BRbgXrYZefsAI%2BP9JU7ImyEAAAAASUVORK5CYII%3D)](https://www.anyscale.com/ray-on-anyscale?utm_source=github&utm_medium=ray_readme&utm_campaign=get_started_badge)
[![Ray Logo](https://github.com/ray-project/ray/raw/master/doc/source/images/ray_header_logo.png)]

## Key Features

Ray offers a comprehensive set of tools and libraries for building and scaling AI and Python applications.

*   **Scalable AI Libraries:** Accelerate your machine learning workflows with built-in libraries for:
    *   Data processing and loading (`Ray Data`).
    *   Distributed training (`Ray Train`).
    *   Hyperparameter tuning (`Ray Tune`).
    *   Reinforcement learning (`Ray RLlib`).
    *   Model serving (`Ray Serve`).
*   **Ray Core - Distributed Runtime:** Provides the foundational building blocks for distributed computing:
    *   **Tasks:** Execute stateless functions across your cluster.
    *   **Actors:** Create stateful worker processes for managing resources.
    *   **Objects:** Efficiently share immutable data across the cluster.
*   **Monitoring and Debugging:** Easily monitor and debug your Ray applications:
    *   **Ray Dashboard:**  Monitor Ray apps and clusters for insights.
    *   **Ray Distributed Debugger:** Debug Ray applications in a distributed environment.
*   **Versatile Deployment:**  Run Ray seamlessly on various infrastructures:
    *   Any machine, cluster, or cloud provider.
    *   Kubernetes.
*   **Extensive Ecosystem:** Benefit from a growing ecosystem of community integrations.

## Why Choose Ray?

As AI workloads become increasingly demanding, single-node development environments fall short. Ray offers a unified solution to scale Python and AI applications from your laptop to a cluster, enabling you to:

*   **Seamless Scaling:** Easily scale the same Python code from your local machine to a cluster without significant code changes.
*   **General-Purpose:** Ray's flexibility allows you to run a wide variety of workloads efficiently.
*   **No Infrastructure Required:** Scale your Python applications without managing complex infrastructure.

## Installation

Install Ray with: `pip install ray`.
For nightly builds, refer to the  [Installation page](https://docs.ray.io/en/latest/ray-overview/installation.html).

## Learn More

*   [Documentation](http://docs.ray.io/en/latest/index.html)
*   [Ray Architecture Whitepaper](https://docs.google.com/document/d/1tBw9A4j62ruI5omIJbMxly-la5w4q_TjyJgJL_jN2fI/preview)

## Getting Involved

Ray has an active community and provides multiple avenues for engagement:

| Platform           | Purpose                                          | Estimated Response Time | Support Level |
| ------------------ | ------------------------------------------------ | ----------------------- | ------------- |
| [Discourse Forum](https://discuss.ray.io/)        | Discussions and usage questions.           | < 1 day               | Community     |
| [GitHub Issues](https://github.com/ray-project/ray/issues)         | Bug reports and feature requests. | < 2 days              | Ray OSS Team  |
| [Slack](https://www.ray.io/join-slack?utm_source=github&utm_medium=ray_readme&utm_campaign=getting_involved)                | Collaboration with other Ray users.      | < 2 days              | Community     |
| [StackOverflow](https://stackoverflow.com/questions/tagged/ray)    | Asking usage questions.                 | 3-5 days              | Community     |
| [Meetup Group](https://www.meetup.com/Bay-Area-Ray-Meetup/)         | Learning about Ray projects.              | Monthly               | Ray DevRel    |
| [Twitter](https://x.com/raydistributed)           | Stay up-to-date on new features.         | Daily                 | Ray DevRel    |

[Back to the top](https://github.com/ray-project/ray)