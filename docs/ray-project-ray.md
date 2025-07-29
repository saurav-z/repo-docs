<img src="https://github.com/ray-project/ray/raw/master/doc/source/images/ray_header_logo.png" alt="Ray Logo" width="100%">

[![Documentation Status](https://readthedocs.org/projects/ray/badge/?version=master)](http://docs.ray.io/en/master/?badge=master)
[![Join Slack](https://img.shields.io/badge/Ray-Join%20Slack-blue)](https://www.ray.io/join-slack)
[![Discuss](https://img.shields.io/badge/Discuss-Ask%20Questions-blue)](https://discuss.ray.io/)
[![Twitter](https://img.shields.io/twitter/follow/raydistributed.svg?style=social&logo=twitter)](https://x.com/raydistributed)
[![Get Started](https://img.shields.io/badge/Get_started_for_free-3C8AE9?logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8%2F9hAAAAAXNSR0IArs4c6QAAAERlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMAAAABAAEAAKACAAQAAAABAAAAEKADAAQAAAABAAAAEAAAAAA0VXHyAAABKElEQVQ4Ea2TvWoCQRRGnWCVWChIIlikC9hpJdikSbGgaONbpAoY8gKBdAGfwkfwKQypLQ1sEGyMYhN1Pd%2B6A8PqwBZeOHt%2FvsvMnd3ZXBRFPQjBZ9K6OY8ZxF%2B0IYw9PW3qz8aY6lk92bZ%2BVqSI3oC9T7%2FyCVnrF1ngj93us%2B540sf5BrCDfw9b6jJ5lx%2FyjtGKBBXc3cnqx0INN4ImbI%2Bl%2BPnI8zWfFEr4chLLrWHCp9OO9j19Kbc91HX0zzzBO8EbLK2Iv4ZvNO3is3h6jb%2BCwO0iL8AaWqB7ILPTxq3kDypqvBuYuwswqo6wgYJbT8XxBPZ8KS1TepkFdC79TAHHce%2F7LbVioi3wEfTpmeKtPRGEeoldSP%2FOeoEftpP4BRbgXrYZefsAI%2BP9JU7ImyEAAAAASUVORK5CYII%3D)](https://www.anyscale.com/ray-on-anyscale?utm_source=github&utm_medium=ray_readme&utm_campaign=get_started_badge)

# Ray: A Unified Framework for Scaling AI and Python Applications

Ray is a powerful and versatile framework that simplifies scaling your AI and Python applications, allowing you to run them seamlessly from your laptop to the cloud.

<img src="https://github.com/ray-project/ray/raw/master/doc/source/images/what-is-ray-padded.svg" alt="What is Ray" width="600">

## Key Features

Ray offers a comprehensive suite of tools for building and scaling AI and Python applications, including:

*   **Ray AI Libraries:**
    *   **Data:** Scalable Datasets for ML.
    *   **Train:** Distributed Training.
    *   **Tune:** Scalable Hyperparameter Tuning.
    *   **RLlib:** Scalable Reinforcement Learning.
    *   **Serve:** Scalable and Programmable Serving.
*   **Ray Core:**
    *   **Tasks:** Stateless functions executed in the cluster.
    *   **Actors:** Stateful worker processes created in the cluster.
    *   **Objects:** Immutable values accessible across the cluster.
*   **Monitoring and Debugging:**
    *   **Ray Dashboard:** Monitor Ray applications and clusters.
    *   **Ray Distributed Debugger:** Debug Ray applications.
*   **Integration:**
    *   Runs on any machine, cluster, cloud provider, and Kubernetes.
    *   Growing [ecosystem of community integrations](https://docs.ray.io/en/latest/ray-overview/ray-libraries.html).

## Getting Started

Install Ray with: `pip install ray`

For nightly wheels, see the [Installation page](https://docs.ray.io/en/latest/ray-overview/installation.html).

## Why Ray?

Ray simplifies the process of scaling your Python and AI applications, allowing you to leverage the power of distributed computing without complex infrastructure.  Scale your Python code from your laptop to a cluster, no other infrastructure required.

## More Information

*   [Documentation](http://docs.ray.io/en/latest/index.html)
*   [Ray Architecture whitepaper](https://docs.google.com/document/d/1tBw9A4j62ruI5omIJbMxly-la5w4q_TjyJgJL_jN2fI/preview)
*   [Exoshuffle: large-scale data shuffle in Ray](https://arxiv.org/abs/2203.05072)
*   [Ownership: a distributed futures system for fine-grained tasks](https://www.usenix.org/system/files/nsdi21-wang.pdf)
*   [RLlib paper](https://arxiv.org/abs/1712.09381)
*   [Tune paper](https://arxiv.org/abs/1807.05118)

## Getting Involved

Ray has a welcoming community.  Connect with us on these platforms:

| Platform         | Purpose                                     | Estimated Response Time | Support Level |
| ---------------- | ------------------------------------------- | ----------------------- | ------------- |
| [Discourse Forum](https://discuss.ray.io/)        | Discussions about development and usage        | < 1 day                 | Community     |
| [GitHub Issues](https://github.com/ray-project/ray/issues)        | Reporting bugs and feature requests             | < 2 days                | Ray OSS Team  |
| [Slack](https://www.ray.io/join-slack?utm_source=github&utm_medium=ray_readme&utm_campaign=getting_involved)           | Collaborating with other Ray users          | < 2 days                | Community     |
| [StackOverflow](https://stackoverflow.com/questions/tagged/ray)      | Asking questions about how to use Ray              | 3-5 days                | Community     |
| [Meetup Group](https://www.meetup.com/Bay-Area-Ray-Meetup/)         | Learning about Ray projects and best practices     | Monthly                 | Ray DevRel    |
| [Twitter](https://x.com/raydistributed)             | Staying up-to-date on new features             | Daily                   | Ray DevRel    |

[Back to Top](#ray-a-unified-framework-for-scaling-ai-and-python-applications)