[![Ray Logo](https://github.com/ray-project/ray/raw/master/doc/source/images/ray_header_logo.png)](https://github.com/ray-project/ray)

[![Documentation Status](https://readthedocs.org/projects/ray/badge/?version=master)](http://docs.ray.io/en/master/?badge=master)
[![Join Slack](https://img.shields.io/badge/Ray-Join%20Slack-blue)](https://www.ray.io/join-slack)
[![Discuss](https://img.shields.io/badge/Discuss-Ask%20Questions-blue)](https://discuss.ray.io/)
[![Twitter](https://img.shields.io/twitter/follow/raydistributed.svg?style=social&logo=twitter)](https://x.com/raydistributed)
[![Get Started Free](https://img.shields.io/badge/Get_started_for_free-3C8AE9?logo=data%3Aimage%2Fpng%3B64%2C%20iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8%2F9hAAAAAXNSR0IArs4c6QAAAERlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMAAAABAAEAAKACAAQAAAABAAAAEKADAAQAAAABAAAAEAAAAAA0VXHyAAABKElEQVQ4Ea2TvWoCQRRGnWCVWChIIlikC9hpJdikSbGgaONbpAoY8gKBdAGfwkfwKQypLQ1sEGyMYhN1Pd%2B6A8PqwBZeOHt%2FvsvMnd3ZXBRFPQjBZ9K6OY8ZxF%2B0IYw9PW3qz8aY6lk92bZ%2BVqSI3oC9T7%2FyCVnrF1ngj93us%2B540sf5BrCDfw9b6jJ5lx%2FyjtGKBBXc3cnqx0INN4ImbI%2Bl%2BPnI8zWfFEr4chLLrWHCp9OO9j19Kbc91HX0zzzBO8EbLK2Iv4ZvNO3is3h6jb%2BCwO0iL8AaWqB7ILPTxq3kDypqvBuYuwswqo6wgYJbT8XxBPZ8KS1TepkFdC79TAHHce%2F7LbVioi3wEfTpmeKtPRGEeoldSP%2FOeoEftpP4BRbgXrYZefsAI%2BP9JU7ImyEAAAAASUVORK5CYII%3D)](https://www.anyscale.com/ray-on-anyscale?utm_source=github&utm_medium=ray_readme&utm_campaign=get_started_badge)

## Ray: Scale Your AI and Python Applications

Ray is a powerful open-source framework designed to effortlessly scale your Python and AI applications from a laptop to a cluster.

[![What is Ray?](https://github.com/ray-project/ray/raw/master/doc/source/images/what-is-ray-padded.svg)](https://github.com/ray-project/ray)

Ray provides a unified platform for building and scaling distributed applications. Here's what makes Ray stand out:

**Key Features:**

*   **Unified Framework:** Scale any Python application without rewriting your code.
*   **AI Libraries:** Simplify ML compute with dedicated libraries for:
    *   Data: Scalable Datasets for ML
    *   Train: Distributed Training
    *   Tune: Scalable Hyperparameter Tuning
    *   RLlib: Scalable Reinforcement Learning
    *   Serve: Scalable and Programmable Serving
*   **Ray Core Abstractions:**
    *   Tasks: Stateless functions executed in the cluster.
    *   Actors: Stateful worker processes created in the cluster.
    *   Objects: Immutable values accessible across the cluster.
*   **Monitoring and Debugging:** Monitor and debug your Ray applications with the Ray Dashboard and Ray Distributed Debugger.
*   **Versatile Deployment:** Runs on any machine, cluster, cloud provider, and Kubernetes.
*   **Extensive Ecosystem:** Integrates with a growing number of community tools and libraries.

**Get Started:**

Install Ray easily using pip:

```bash
pip install ray
```

For nightly builds, refer to the [Installation page](https://docs.ray.io/en/latest/ray-overview/installation.html).

**Why Ray?**

Modern ML workloads demand significant computational resources.  Ray empowers you to scale your Python and AI applications seamlessly, eliminating the need for complex infrastructure.  It's designed to be general-purpose, allowing you to scale any Python application efficiently.

**More Information:**

*   [Documentation](http://docs.ray.io/en/latest/index.html)
*   [Ray Architecture Whitepaper](https://docs.google.com/document/d/1tBw9A4j62ruI5omIJbMxly-la5w4q_TjyJgJL_jN2fI/preview)
*   [Exoshuffle: large-scale data shuffle in Ray](https://arxiv.org/abs/2203.05072)
*   [Ownership: a distributed futures system for fine-grained tasks](https://www.usenix.org/system/files/nsdi21-wang.pdf)
*   [RLlib paper](https://arxiv.org/abs/1712.09381)
*   [Tune paper](https://arxiv.org/abs/1807.05118)

**Getting Involved:**

| Platform          | Purpose                                                                  | Estimated Response Time | Support Level |
| :---------------- | :----------------------------------------------------------------------- | :---------------------- | :------------ |
| [Discourse Forum](https://discuss.ray.io/)    | Discussions and usage questions.                                   | < 1 day              | Community     |
| [GitHub Issues](https://github.com/ray-project/ray/issues)   | Bug reports and feature requests.                              | < 2 days             | Ray OSS Team  |
| [Slack](https://www.ray.io/join-slack?utm_source=github&utm_medium=ray_readme&utm_campaign=getting_involved)            | Collaborating with other Ray users.                                     | < 2 days             | Community     |
| [StackOverflow](https://stackoverflow.com/questions/tagged/ray) | Asking usage questions.                                     | 3-5 days              | Community     |
| [Meetup Group](https://www.meetup.com/Bay-Area-Ray-Meetup/)    | Learning about Ray projects and best practices.                | Monthly               | Ray DevRel    |
| [Twitter](https://x.com/raydistributed)            | Staying up-to-date on new features.                                | Daily                 | Ray DevRel    |

**[View the original repository on GitHub](https://github.com/ray-project/ray)**