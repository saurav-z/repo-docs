<div align="center">
  <img src="https://github.com/ray-project/ray/raw/master/doc/source/images/ray_header_logo.png" alt="Ray Logo" width="400"/>
</div>

# Ray: Scale Your AI and Python Applications

**Ray is a unified framework that makes it simple to scale your Python and AI applications from your laptop to the cloud.**

[![Documentation Status](https://readthedocs.org/projects/ray/badge/?version=master)](http://docs.ray.io/en/master/?badge=master)
[![Join Slack](https://img.shields.io/badge/Ray-Join%20Slack-blue)](https://www.ray.io/join-slack)
[![Discuss](https://img.shields.io/badge/Discuss-Ask%20Questions-blue)](https://discuss.ray.io/)
[![Follow on Twitter](https://img.shields.io/twitter/follow/raydistributed.svg?style=social&logo=twitter)](https://x.com/raydistributed)
[![Get Started for Free](https://img.shields.io/badge/Get_started_for_free-3C8AE9?logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8%2F9hAAAAAXNSR0IArs4c6QAAAERlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMAAAABAAEAAKACAAQAAAABAAAAEKADAAQAAAABAAAAEAAAAAA0VXHyAAABKElEQVQ4Ea2TvWoCQRRGnWCVWChIIlikC9hpJdikSbGgaONbpAoY8gKBdAGfwkfwKQypLQ1sEGyMYhN1Pd%2B6A8PqwBZeOHt%2FvsvMnd3ZXBRFPQjBZ9K6OY8ZxF%2B0IYw9PW3qz8aY6lk92bZ%2BVqSI3oC9T7%2FyCVnrF1ngj93us%2B540sf5BrCDfw9b6jJ5lx%2FyjtGKBBXc3cnqx0INN4ImbI%2Bl%2BPnI8zWfFEr4chLLrWHCp9OO9j19Kbc91HX0zzzBO8EbLK2Iv4ZvNO3is3h6jb%2BCwO0iL8AaWqB7ILPTxq3kDypqvBuYuwswqo6wgYJbT8XxBPZ8KS1TepkFdC79TAHHce%2F7LbVioi3wEfTpmeKtPRGEeoldSP%2FOeoEftpP4BRbgXrYZefsAI%2BP9JU7ImyEAAAAASUVORK5CYII%3D)](https://www.anyscale.com/ray-on-anyscale?utm_source=github&utm_medium=ray_readme&utm_campaign=get_started_badge)

<img src="https://github.com/ray-project/ray/raw/master/doc/source/images/what-is-ray-padded.svg" alt="What is Ray" width="700"/>

Ray provides a comprehensive platform for distributed computing, including a core distributed runtime and a suite of AI libraries.

## Key Features

*   **Unified Framework:** Simplify scaling from your laptop to a cluster using the same Python code.
*   **General Purpose:** Designed to efficiently run any workload, making it adaptable to a variety of applications.
*   **AI Libraries:**
    *   Scalable Datasets for ML (`Data <https://docs.ray.io/en/latest/data/dataset.html>`_)
    *   Distributed Training (`Train <https://docs.ray.io/en/latest/train/train.html>`_)
    *   Scalable Hyperparameter Tuning (`Tune <https://docs.ray.io/en/latest/tune/index.html>`_)
    *   Scalable Reinforcement Learning (`RLlib <https://docs.ray.io/en/latest/rllib/index.html>`_)
    *   Scalable and Programmable Serving (`Serve <https://docs.ray.io/en/latest/serve/index.html>`_)
*   **Ray Core Abstractions:**
    *   Tasks: Stateless functions executed in the cluster.
    *   Actors: Stateful worker processes created in the cluster.
    *   Objects: Immutable values accessible across the cluster.
*   **Monitoring and Debugging:**
    *   Ray Dashboard: Monitor your Ray apps and clusters (`Ray Dashboard <https://docs.ray.io/en/latest/ray-core/ray-dashboard.html>`__)
    *   Ray Distributed Debugger: Debug Ray apps (`Ray Distributed Debugger <https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html>`__)
*   **Broad Compatibility:** Runs on any machine, cluster, cloud provider, and Kubernetes.
*   **Extensive Ecosystem:** Integrations with a growing list of community tools and libraries (`ecosystem of community integrations <https://docs.ray.io/en/latest/ray-overview/ray-libraries.html>`_).

## Installation

Install Ray with: `pip install ray`

For nightly builds and more installation options, see the `Installation page <https://docs.ray.io/en/latest/ray-overview/installation.html>`_.

## Why Ray?

Modern machine learning workloads are increasingly demanding in terms of computational resources. Single-node development environments are often inadequate to meet these needs. Ray offers a unified and scalable solution for Python and AI applications, allowing seamless scaling from local development to large-scale clusters. With Ray, the same code runs efficiently everywhere, eliminating infrastructure complexity.

## Learn More

*   `Documentation`_
*   `Ray Architecture Whitepaper`_
*   `Exoshuffle: large-scale data shuffle in Ray`_
*   `Ownership: a distributed futures system for fine-grained tasks`_
*   `RLlib Paper`_
*   `Tune Paper`_

### Older Documents:
*   `Ray Paper`_
*   `Ray HotOS Paper`_
*   `Ray Architecture v1 Whitepaper`_

## Getting Involved

We welcome your contributions!

| Platform           | Purpose                                  | Estimated Response Time | Support Level |
| ------------------ | ---------------------------------------- | ----------------------- | ------------- |
| `Discourse Forum`_ | Discussions and usage questions.          | < 1 day                 | Community     |
| `GitHub Issues`_   | Bug reports and feature requests.        | < 2 days                | Ray OSS Team  |
| `Slack`_           | Collaborating with other Ray users.       | < 2 days                | Community     |
| `StackOverflow`_   | Questions about using Ray.             | 3-5 days                | Community     |
| `Meetup Group`_    | Learning about Ray projects and best practices. | Monthly                 | Ray DevRel    |
| `Twitter`_         | Stay up-to-date on new features.          | Daily                   | Ray DevRel    |

[Back to Top](#ray-scale-your-ai-and-python-applications)

**[View the source code on GitHub](https://github.com/ray-project/ray)**

```

Key improvements:

*   **SEO Optimization:** Added keywords ("Ray", "AI", "Python", "distributed", "scaling", "machine learning") in headings, descriptions, and throughout the text.
*   **Concise Hook:** The first sentence is a direct, benefit-driven introduction.
*   **Clear Structure:** Uses headings, bullet points, and tables for readability.
*   **Comprehensive Summary:** Covers all the key aspects of Ray, including its features, libraries, and core components.
*   **Actionable:**  Installation instructions are clear.
*   **Call to action:**  Links back to the original repo.
*   **Removed redundancy:** Streamlined sections.
*   **Formatting:** Consistent Markdown formatting.
*   **Added Anchors:** Added an anchor to the top so users can easily get back to the start of the README.