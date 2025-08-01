<div align="center">
  <img src="https://github.com/ray-project/ray/raw/master/doc/source/images/ray_header_logo.png" alt="Ray Logo" width="400"/>
  <h1>Ray: Scale Your Python and AI Applications</h1>
</div>

Ray is a unified, open-source framework that simplifies scaling your AI and Python applications from your laptop to the cloud.

[![Documentation Status](http://docs.ray.io/en/master/?badge=master)](http://docs.ray.io/en/master/?badge=master)
[![Join Slack](https://img.shields.io/badge/Ray-Join%20Slack-blue)](https://www.ray.io/join-slack)
[![Ask Questions](https://img.shields.io/badge/Discuss-Ask%20Questions-blue)](https://discuss.ray.io/)
[![Follow on Twitter](https://img.shields.io/twitter/follow/raydistributed.svg?style=social&logo=twitter)](https://x.com/raydistributed)
[![Get Started](https://img.shields.io/badge/Get_started_for_free-3C8AE9?logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8%2F9hAAAAAXNSR0IArs4c6QAAAERlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMAAAABAAEAAKACAAQAAAABAAAAEKADAAQAAAABAAAAEAAAAAA0VXHyAAABKElEQVQ4Ea2TvWoCQRRGnWCVWChIIlikC9hpJdikSbGgaONbpAoY8gKBdAGfwkfwKQypLQ1sEGyMYhN1Pd%2B6A8PqwBZeOHt%2FvsvMnd3ZXBRFPQjBZ9K6OY8ZxF%2B0IYw9PW3qz8aY6lk92bZ%2BVqSI3oC9T7%2FyCVnrF1ngj93us%2B540sf5BrCDfw9b6jJ5lx%2FyjtGKBBXc3cnqx0INN4ImbI%2Bl%2BPnI8zWfFEr4chLLrWHCp9OO9j19Kbc91HX0zzzBO8EbLK2Iv4ZvNO3is3h6jb%2BCwO0iL8AaWqB7ILPTxq3kDypqvBuYuwswqo6wgYJbT8XxBPZ8KS1TepkFdC79TAHHce%2F7LbVioi3wEfTpmeKtPRGEeoldSP%2FOeoEftpP4BRbgXrYZefsAI%2BP9JU7ImyEAAAAASUVORK5CYII%3D)](https://www.anyscale.com/ray-on-anyscale?utm_source=github&utm_medium=ray_readme&utm_campaign=get_started_badge)

Ray is a powerful framework designed for scaling AI and Python applications. It provides a distributed runtime and a suite of AI libraries to streamline your machine learning workflows.

<img src="https://github.com/ray-project/ray/raw/master/doc/source/images/what-is-ray-padded.svg" alt="What is Ray?" width="700"/>

## Key Features

*   **Unified Framework:** Simplify scaling from a laptop to a cluster.
*   **General-Purpose:** Run any Python workload with optimized performance.
*   **AI Libraries:** Leverage specialized libraries for ML compute.
*   **Scalable Datasets:**  Use `Data` for scalable datasets.
*   **Distributed Training:**  Train your models efficiently with `Train`.
*   **Hyperparameter Tuning:** Optimize your models with `Tune`.
*   **Reinforcement Learning:** Scale your RL applications with `RLlib`.
*   **Model Serving:** Deploy and serve your models with `Serve`.
*   **Core Abstractions:** Utilize `Tasks`, `Actors`, and `Objects` for distributed computing.
*   **Monitoring and Debugging:** Monitor and debug your Ray applications with the `Ray Dashboard <https://docs.ray.io/en/latest/ray-core/ray-dashboard.html>`__ and `Ray Distributed Debugger <https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html>`__.
*   **Extensive Ecosystem:** Integrate with a growing `ecosystem of community integrations`_.

## Installation

Install Ray with:

```bash
pip install ray
```

For nightly builds, please consult the `Installation page <https://docs.ray.io/en/latest/ray-overview/installation.html>`__.

## Why Ray?

Ray is the unified solution for scaling Python and AI applications.  Single-node environments are often insufficient for today's compute-intensive ML workloads. Ray solves this by enabling seamless scaling of your code from your laptop to a cluster, requiring no infrastructure changes.  Ray's general-purpose design supports any Python-based workload, making it a versatile tool for various applications.

## More Information

*   [Documentation](http://docs.ray.io/en/latest/index.html)
*   [Ray Architecture Whitepaper](https://docs.google.com/document/d/1tBw9A4j62ruI5omIJbMxly-la5w4q_TjyJgJL_jN2fI/preview)
*   [Exoshuffle: large-scale data shuffle in Ray](https://arxiv.org/abs/2203.05072)
*   [Ownership: a distributed futures system for fine-grained tasks](https://www.usenix.org/system/files/nsdi21-wang.pdf)
*   [RLlib paper](https://arxiv.org/abs/1712.09381)
*   [Tune paper](https://arxiv.org/abs/1807.05118)
*   [Ray Paper](https://arxiv.org/abs/1712.05889)
*   [Ray HotOS paper](https://arxiv.org/abs/1703.03924)
*   [Ray Architecture v1 whitepaper](https://docs.google.com/document/d/1lAy0Owi-vPz2jEqBSaHNQcy2IBSDEHyXNOQZlGuj93c/preview)

## Getting Involved

| Platform           | Purpose                                      | Estimated Response Time | Support Level |
| ------------------ | -------------------------------------------- | ----------------------- | ------------- |
| [Discourse Forum](https://discuss.ray.io/) | Discussions about development and usage | < 1 day               | Community     |
| [GitHub Issues](https://github.com/ray-project/ray/issues)   | Bug reports and feature requests   | < 2 days              | Ray OSS Team  |
| [Slack](https://www.ray.io/join-slack?utm_source=github&utm_medium=ray_readme&utm_campaign=getting_involved)         | Collaborate with other Ray users | < 2 days               | Community     |
| [StackOverflow](https://stackoverflow.com/questions/tagged/ray)     | Ask questions about Ray usage   | 3-5 days               | Community     |
| [Meetup Group](https://www.meetup.com/Bay-Area-Ray-Meetup/)        | Learn about projects & best practices  | Monthly               | Ray DevRel    |
| [Twitter](https://x.com/raydistributed)            | Stay up-to-date on new features        | Daily                 | Ray DevRel    |

[Back to top](#) ([Ray Project](https://github.com/ray-project/ray))
```
Key improvements and SEO considerations:

*   **Clear, Concise Title:** Includes the primary keywords "Ray" and "Python" and "AI."
*   **One-Sentence Hook:** Immediately grabs the reader's attention and explains the core value proposition.
*   **Detailed Headings:**  Uses clear headings like "Key Features," "Installation," "Why Ray?" and "Getting Involved" to organize the information.
*   **Bulleted Key Features:**  Provides a scannable list of Ray's main benefits.  Includes keywords like "distributed training", "hyperparameter tuning" etc.
*   **SEO-Friendly Language:**  Uses terms that users would likely search for (e.g., "scale Python applications," "distributed computing," "ML workflows").
*   **Call to Action:** Encourages users to install and get involved.
*   **Back to Top & Repo link:** Added at the end for improved navigation and a direct link back to the repository.
*   **Image Alt Text:** Images have meaningful alt text for accessibility and SEO.
*   **Clean Formatting:** Uses Markdown formatting for readability.
*   **Concise:** Removes unnecessary text while retaining essential information.
*   **Hyperlinking:** Links are direct and informative and use descriptive anchor text.
*   **Getting Involved Section:** Highlights various ways to connect with the community, with estimated response times and support levels.