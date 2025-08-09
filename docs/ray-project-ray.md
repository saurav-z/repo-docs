<div align="center">
  <img src="https://github.com/ray-project/ray/raw/master/doc/source/images/ray_header_logo.png" alt="Ray Logo" width="400"/>
</div>

# Ray: Scale Your Python and AI Applications

**Ray is a powerful, unified framework designed to effortlessly scale your Python and AI applications from a single laptop to a large cluster.**  Check out the [original repo](https://github.com/ray-project/ray) for more details.

[![Documentation Status](http://docs.ray.io/en/master/?badge=master)](http://docs.ray.io/en/master/?badge=master)
[![Join Slack](https://img.shields.io/badge/Ray-Join%20Slack-blue)](https://www.ray.io/join-slack)
[![Discuss](https://img.shields.io/badge/Discuss-Ask%20Questions-blue)](https://discuss.ray.io/)
[![Twitter](https://img.shields.io/twitter/follow/raydistributed.svg?style=social&logo=twitter)](https://x.com/raydistributed)
[![Get Started for Free](https://img.shields.io/badge/Get_started_for_free-3C8AE9?logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8%2F9hAAAAAXNSR0IArs4c6QAAAERlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMAAAABAAEAAKACAAQAAAABAAAAEKADAAQAAAABAAAAEAAAAAA0VXHyAAABKElEQVQ4Ea2TvWoCQRRGnWCVWChIIlikC9hpJdikSbGgaONbpAoY8gKBdAGfwkfwKQypLQ1sEGyMYhN1Pd%2B6A8PqwBZeOHt%2FvsvMnd3ZXBRFPQjBZ9K6OY8ZxF%2B0IYw9PW3qz8aY6lk92bZ%2BVqSI3oC9T7%2FyCVnrF1ngj93us%2B540sf5BrCDfw9b6jJ5lx%2FyjtGKBBXc3cnqx0INN4ImbI%2Bl%2BPnI8zWfFEr4chLLrWHCp9OO9j19Kbc91HX0zzzBO8EbLK2Iv4ZvNO3is3h6jb%2BCwO0iL8AaWqB7ILPTxq3kDypqvBuYuwswqo6wgYJbT8XxBPZ8KS1TepkFdC79TAHHce%2F7LbVioi3wEfTpmeKtPRGEeoldSP%2FOeoEftpP4BRbgXrYZefsAI%2BP9JU7ImyEAAAAASUVORK5CYII%3D)](https://www.anyscale.com/ray-on-anyscale?utm_source=github&utm_medium=ray_readme&utm_campaign=get_started_badge)

## What is Ray?

<div align="center">
  <img src="https://github.com/ray-project/ray/raw/master/doc/source/images/what-is-ray-padded.svg" alt="Ray Architecture" width="700"/>
</div>

Ray is a unified framework for scaling AI and Python applications. It provides a core distributed runtime and a set of powerful AI libraries, simplifying the development and deployment of machine learning and other compute-intensive workloads.

## Key Features of Ray

*   **Ray AI Libraries:**
    *   Scalable Datasets for ML (`Data <https://docs.ray.io/en/latest/data/dataset.html>`)
    *   Distributed Training (`Train <https://docs.ray.io/en/latest/train/train.html>`)
    *   Scalable Hyperparameter Tuning (`Tune <https://docs.ray.io/en/latest/tune/index.html>`)
    *   Scalable Reinforcement Learning (`RLlib <https://docs.ray.io/en/latest/rllib/index.html>`)
    *   Scalable and Programmable Serving (`Serve <https://docs.ray.io/en/latest/serve/index.html>`)

*   **Ray Core Abstractions:**
    *   `Tasks <https://docs.ray.io/en/latest/ray-core/tasks.html>`: Stateless functions executed in the cluster.
    *   `Actors <https://docs.ray.io/en/latest/ray-core/actors.html>`: Stateful worker processes created in the cluster.
    *   `Objects <https://docs.ray.io/en/latest/ray-core/objects.html>`: Immutable values accessible across the cluster.

*   **Monitoring and Debugging:**
    *   Monitor Ray apps and clusters with the `Ray Dashboard <https://docs.ray.io/en/latest/ray-core/ray-dashboard.html>`.
    *   Debug Ray apps with the `Ray Distributed Debugger <https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html>`.

*   **Cross-Platform Compatibility:**
    *   Runs on any machine, cluster, cloud provider, and Kubernetes.
    *   Offers a growing ecosystem of community integrations.

## Getting Started

Install Ray with: `pip install ray`

For nightly builds, check out the `Installation page <https://docs.ray.io/en/latest/ray-overview/installation.html>`.

## Why Use Ray?

Ray provides a unified solution for scaling your Python and AI applications from your local machine to a distributed cluster.  This enables you to:

*   **Seamlessly scale code:** Run the same code on your laptop or a cluster.
*   **General-purpose:** Adaptable to any kind of workload.
*   **Python-friendly:** Scale your Python applications without extra infrastructure.

## Additional Resources

*   `Documentation <http://docs.ray.io/en/latest/index.html>`
*   `Ray Architecture whitepaper <https://docs.google.com/document/d/1tBw9A4j62ruI5omIJbMxly-la5w4q_TjyJgJL_jN2fI/preview>`
*   `Exoshuffle: large-scale data shuffle in Ray <https://arxiv.org/abs/2203.05072>`
*   `Ownership: a distributed futures system for fine-grained tasks <https://www.usenix.org/system/files/nsdi21-wang.pdf>`
*   `RLlib paper <https://arxiv.org/abs/1712.09381>`
*   `Tune paper <https://arxiv.org/abs/1807.05118>`

*Older documents:*

*   `Ray paper <https://arxiv.org/abs/1712.05889>`
*   `Ray HotOS paper <https://arxiv.org/abs/1703.03924>`
*   `Ray Architecture v1 whitepaper <https://docs.google.com/document/d/1lAy0Owi-vPz2jEqBSaHNQcy2IBSDEHyXNOQZlGuj93c/preview>`

## Getting Involved

Join the Ray community and get support:

| Platform           | Purpose                                           | Estimated Response Time | Support Level |
| ------------------ | ------------------------------------------------- | ----------------------- | ------------- |
| `Discourse Forum <https://discuss.ray.io/>`          | Discussions about development and usage questions. | < 1 day               | Community     |
| `GitHub Issues <https://github.com/ray-project/ray/issues>`   | Reporting bugs and feature requests.          | < 2 days               | Ray OSS Team  |
| `Slack <https://www.ray.io/join-slack?utm_source=github&utm_medium=ray_readme&utm_campaign=getting_involved>`            | Collaborating with other Ray users.               | < 2 days               | Community     |
| `StackOverflow <https://stackoverflow.com/questions/tagged/ray>`   | Asking questions about how to use Ray.          | 3-5 days              | Community     |
| `Meetup Group <https://www.meetup.com/Bay-Area-Ray-Meetup/>`        | Learning about Ray projects and best practices.   | Monthly               | Ray DevRel    |
| `Twitter <https://x.com/raydistributed>`             | Staying up-to-date on new features.               | Daily                 | Ray DevRel    |