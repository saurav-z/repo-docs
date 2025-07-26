[![Ray Logo](https://github.com/ray-project/ray/raw/master/doc/source/images/ray_header_logo.png)](https://github.com/ray-project/ray)

[![Documentation Status](https://readthedocs.org/projects/ray/badge/?version=master)](http://docs.ray.io/en/master/?badge=master)
[![Join Slack](https://img.shields.io/badge/Ray-Join%20Slack-blue)](https://www.ray.io/join-slack)
[![Discuss on Discourse](https://img.shields.io/badge/Discuss-Ask%20Questions-blue)](https://discuss.ray.io/)
[![Follow on Twitter](https://img.shields.io/twitter/follow/raydistributed.svg?style=social&logo=twitter)](https://x.com/raydistributed)
[![Get Started for Free](https://img.shields.io/badge/Get_started_for_free-3C8AE9?logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8%2F9hAAAAAXNSR0IArs4c6QAAAERlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMAAAABAAEAAKACAAQAAAABAAAAEKADAAQAAAABAAAAEAAAAAA0VXHyAAABKElEQVQ4Ea2TvWoCQRRGnWCVWChIIlikC9hpJdikSbGgaONbpAoY8gKBdAGfwkfwKQypLQ1sEGyMYhN1Pd%2B6A8PqwBZeOHt%2FvsvMnd3ZXBRFPQjBZ9K6OY8ZxF%2B0IYw9PW3qz8aY6lk92bZ%2BVqSI3oC9T7%2FyCVnrF1ngj93us%2B540sf5BrCDfw9b6jJ5lx%2FyjtGKBBXc3cnqx0INN4ImbI%2Bl%2BPnI8zWfFEr4chLLrWHCp9OO9j19Kbc91HX0zzzBO8EbLK2Iv4ZvNO3is3h6jb%2BCwO0iL8AaWqB7ILPTxq3kDypqvBuYuwswqo6wgYJbT8XxBPZ8KS1TepkFdC79TAHHce%2F7LbVioi3wEfTpmeKtPRGEeoldSP%2FOeoEftpP4BRbgXrYZefsAI%2BP9JU7ImyEAAAAASUVORK5CYII%3D)](https://www.anyscale.com/ray-on-anyscale?utm_source=github&utm_medium=ray_readme&utm_campaign=get_started_badge)

## Ray: Scale Your Python and AI Applications with Ease

Ray is a powerful, unified framework designed for scaling AI and Python applications from your laptop to the cloud.  

[![What is Ray?](https://github.com/ray-project/ray/raw/master/doc/source/images/what-is-ray-padded.svg)](https://github.com/ray-project/ray)

### Key Features

*   **Unified Framework:** Easily scale Python and AI applications without rewriting your code.
*   **Distributed Runtime:** A core runtime engine optimized for distributed computing.
*   **AI Libraries:** Simplified ML compute with a suite of libraries:
    *   Scalable Datasets (`Data <https://docs.ray.io/en/latest/data/dataset.html>`__)
    *   Distributed Training (`Train <https://docs.ray.io/en/latest/train/train.html>`__)
    *   Scalable Hyperparameter Tuning (`Tune <https://docs.ray.io/en/latest/tune/index.html>`__)
    *   Scalable Reinforcement Learning (`RLlib <https://docs.ray.io/en/latest/rllib/index.html>`__)
    *   Scalable and Programmable Serving (`Serve <https://docs.ray.io/en/latest/serve/index.html>`__)
*   **Ray Core Abstractions:** Fundamental building blocks for distributed applications:
    *   Tasks: Stateless functions executed in the cluster.
    *   Actors: Stateful worker processes created in the cluster.
    *   Objects: Immutable values accessible across the cluster.
*   **Monitoring and Debugging:**  Monitor and debug your Ray applications with the Ray Dashboard and Distributed Debugger.
*   **Flexible Deployment:** Runs on any machine, cluster, cloud provider, and Kubernetes.
*   **Extensive Ecosystem:** Integrates with a growing `ecosystem of community integrations <https://docs.ray.io/en/latest/ray-overview/ray-libraries.html>`__.

### Why Choose Ray?

As ML workloads grow, single-node environments become a bottleneck. Ray provides a unified solution to scale your Python and AI applications seamlessly.  Ray's general-purpose design allows you to scale *any* Python workload efficiently, eliminating the need for complex infrastructure.

### Getting Started

Install Ray with: `pip install ray`

For nightly builds, consult the `Installation page <https://docs.ray.io/en/latest/ray-overview/installation.html>`__.

### Learn More

*   `Documentation <http://docs.ray.io/en/latest/index.html>`_
*   `Ray Architecture whitepaper <https://docs.google.com/document/d/1tBw9A4j62ruI5omIJbMxly-la5w4q_TjyJgJL_jN2fI/preview>`_
*   `Exoshuffle: large-scale data shuffle in Ray <https://arxiv.org/abs/2203.05072>`_
*   `Ownership: a distributed futures system for fine-grained tasks <https://www.usenix.org/system/files/nsdi21-wang.pdf>`_
*   `RLlib paper <https://arxiv.org/abs/1712.09381>`_
*   `Tune paper <https://arxiv.org/abs/1807.05118>`_
*   *Older documents:*
    *   `Ray paper <https://arxiv.org/abs/1712.05889>`_
    *   `Ray HotOS paper <https://arxiv.org/abs/1703.03924>`_
    *   `Ray Architecture v1 whitepaper <https://docs.google.com/document/d/1lAy0Owi-vPz2jEqBSaHNQcy2IBSDEHyXNOQZlGuj93c/preview>`_

### Getting Involved

Connect with the Ray community and get help:

| Platform           | Purpose                                                | Estimated Response Time | Support Level |
| ------------------ | ------------------------------------------------------ | ----------------------- | ------------- |
| `Discourse Forum <https://discuss.ray.io/>`_ | Discussions and usage questions.               | < 1 day                 | Community     |
| `GitHub Issues <https://github.com/ray-project/ray/issues>`_   | Bug reports and feature requests.        | < 2 days                | Ray OSS Team  |
| `Slack <https://www.ray.io/join-slack?utm_source=github&utm_medium=ray_readme&utm_campaign=getting_involved>`_   | Collaborate with other Ray users.                 | < 2 days                | Community     |
| `StackOverflow <https://stackoverflow.com/questions/tagged/ray>`_   | Ask questions about using Ray.           | 3-5 days                | Community     |
| `Meetup Group <https://www.meetup.com/Bay-Area-Ray-Meetup/>`_    | Learn about Ray projects and best practices. | Monthly                 | Ray DevRel    |
| `Twitter <https://x.com/raydistributed>`_      | Stay up-to-date on new features.              | Daily                   | Ray DevRel    |

[Back to top](https://github.com/ray-project/ray)