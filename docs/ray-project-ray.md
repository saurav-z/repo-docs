<div align="center">
  <a href="https://github.com/ray-project/ray">
    <img src="https://github.com/ray-project/ray/raw/master/doc/source/images/ray_header_logo.png" alt="Ray Logo" width="400"/>
  </a>
  <br/>
  <a href="https://www.anyscale.com/ray-on-anyscale?utm_source=github&utm_medium=ray_readme&utm_campaign=get_started_badge">
    <img src="https://img.shields.io/badge/Get_started_for_free-3C8AE9?logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8%2F9hAAAAAXNSR0IArs4c6QAAAERlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMAAAABAAEAAKACAAQAAAABAAAAEKADAAQAAAABAAAAEAAAAAA0VXHyAAABKElEQVQ4Ea2TvWoCQRRGnWCVWChIIlikC9hpJdikSbGgaONbpAoY8gKBdAGfwkfwKQypLQ1sEGyMYhN1Pd%2B6A8PqwBZeOHt%2FvsvMnd3ZXBRFPQjBZ9K6OY8ZxF%2B0IYw9PW3qz8aY6lk92bZ%2BVqSI3oC9T7%2FyCVnrF1ngj93us%2B540sf5BrCDfw9b6jJ5lx%2FyjtGKBBXc3cnqx0INN4ImbI%2Bl%2BPnI8zWfFEr4chLLrWHCp9OO9j19Kbc91HX0zzzBO8EbLK2Iv4ZvNO3is3h6jb%2BCwO0iL8AaWqB7ILPTxq3kDypqvBuYuwswqo6wgYJbT8XxBPZ8KS1TepkFdC79TAHHce%2F7LbVioi3wEfTpmeKtPRGEeoldSP%2FOeoEftpP4BRbgXrYZefsAI%2BP9JU7ImyEAAAAASUVORK5CYII%3D" alt="Get Started for Free"/>
  </a>
</div>

# Ray: Scale Your AI and Python Applications

**Ray is a unified framework for scaling AI and Python applications, empowering you to effortlessly transition from your laptop to a cluster.**

[![Documentation Status](https://readthedocs.org/projects/ray/badge/?version=master)](http://docs.ray.io/en/master/?badge=master)
[![Join Slack](https://img.shields.io/badge/Ray-Join%20Slack-blue)](https://www.ray.io/join-slack)
[![Discuss](https://img.shields.io/badge/Discuss-Ask%20Questions-blue)](https://discuss.ray.io/)
[![Twitter](https://img.shields.io/twitter/follow/raydistributed.svg?style=social&logo=twitter)](https://x.com/raydistributed)

## Key Features

Ray offers a comprehensive suite of tools for building and scaling AI and Python applications:

*   **Unified Framework:** A single framework for diverse workloads, simplifying development and deployment.
*   **Scalable AI Libraries:**  Pre-built libraries for key AI tasks, including:
    *   Scalable Datasets for ML
    *   Distributed Training
    *   Hyperparameter Tuning
    *   Scalable Reinforcement Learning (RLlib)
    *   Scalable and Programmable Serving
*   **Ray Core Abstractions:**  Fundamental building blocks for distributed computing:
    *   Tasks: Stateless functions executed in the cluster.
    *   Actors: Stateful worker processes created in the cluster.
    *   Objects: Immutable values accessible across the cluster.
*   **Monitoring and Debugging:**
    *   Ray Dashboard for monitoring apps and clusters.
    *   Ray Distributed Debugger for debugging applications.
*   **Versatile Deployment:** Runs on any machine, cluster, cloud provider, and Kubernetes.
*   **Extensive Ecosystem:** Growing ecosystem of community integrations.

## Why Ray?

Single-node environments limit the potential of today's compute-intensive ML workloads. Ray provides a seamless transition from local development to scalable cluster execution, allowing you to run the same Python code on your laptop or a large-scale distributed system. Ray's general-purpose design ensures optimal performance for any Python workload, eliminating the need for specialized infrastructure.

## Installation

Install Ray using pip:

```bash
pip install ray
```

For nightly builds, consult the [Installation page](https://docs.ray.io/en/latest/ray-overview/installation.html).

## Getting Involved

Connect with the Ray community and get support through various channels:

| Platform            | Purpose                                      | Estimated Response Time | Support Level |
| ------------------- | -------------------------------------------- | ----------------------- | ------------- |
| [Discourse Forum](https://discuss.ray.io/) | Discussions and usage questions | < 1 day                 | Community     |
| [GitHub Issues](https://github.com/ray-project/ray/issues) | Bug reports and feature requests     | < 2 days                | Ray OSS Team  |
| [Slack](https://www.ray.io/join-slack?utm_source=github&utm_medium=ray_readme&utm_campaign=getting_involved)      | Collaboration with other users          | < 2 days                | Community     |
| [StackOverflow](https://stackoverflow.com/questions/tagged/ray)   | Usage questions                       | 3-5 days                | Community     |
| [Meetup Group](https://www.meetup.com/Bay-Area-Ray-Meetup/)       | Learn about Ray projects and best practices | Monthly                 | Ray DevRel    |
| [Twitter](https://x.com/raydistributed)           | Stay up-to-date                         | Daily                   | Ray DevRel    |

## More Information

*   [Documentation](http://docs.ray.io/en/latest/index.html)
*   [Ray Architecture whitepaper](https://docs.google.com/document/d/1tBw9A4j62ruI5omIJbMxly-la5w4q_TjyJgJL_jN2fI/preview)
*   [Exoshuffle: large-scale data shuffle in Ray](https://arxiv.org/abs/2203.05072)
*   [Ownership: a distributed futures system for fine-grained tasks](https://www.usenix.org/system/files/nsdi21-wang.pdf)
*   [RLlib paper](https://arxiv.org/abs/1712.09381)
*   [Tune paper](https://arxiv.org/abs/1807.05118)

**[Visit the original repository on GitHub](https://github.com/ray-project/ray)**