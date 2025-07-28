<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![Ray Logo](https://github.com/ray-project/ray/raw/master/doc/source/images/ray_header_logo.png)](https://github.com/ray-project/ray)

# Ray: Scale Your AI and Python Applications

**Ray is a unified, open-source framework that lets you easily scale your AI and Python applications from a laptop to a cluster.**

Ray provides a powerful and flexible platform for distributed computing, enabling you to tackle even the most demanding AI and machine learning workloads. Visit the [Ray GitHub repository](https://github.com/ray-project/ray) to explore the source code and contribute.

## Key Features

*   **Unified Framework:** Simplifies distributed computing with a single framework for various AI and Python applications.
*   **Scalable AI Libraries:** Offers a suite of libraries designed to simplify ML compute:
    *   **Data:** Scalable Datasets for ML
    *   **Train:** Distributed Training
    *   **Tune:** Scalable Hyperparameter Tuning
    *   **RLlib:** Scalable Reinforcement Learning
    *   **Serve:** Scalable and Programmable Serving
*   **Ray Core for Distributed Computing:** Provides the core abstractions for building distributed applications:
    *   **Tasks:** Stateless functions executed in the cluster.
    *   **Actors:** Stateful worker processes created in the cluster.
    *   **Objects:** Immutable values accessible across the cluster.
*   **Flexible Deployment:** Runs on any machine, cluster, cloud provider, and Kubernetes.
*   **Monitoring and Debugging:**
    *   **Ray Dashboard:** Monitor Ray apps and clusters.
    *   **Ray Distributed Debugger:** Debug Ray applications.
*   **Community Integrations:** Growing ecosystem of community integrations.

## Why Ray?

Modern ML workloads are increasingly compute-intensive, requiring the ability to scale beyond single-node development environments. Ray provides a unified way to scale Python and AI applications from a laptop to a cluster, using the same code. Ray's general-purpose design enables it to efficiently run diverse workloads, allowing you to scale any Python application with ease.

## Getting Started

Install Ray:

```bash
pip install ray
```

For nightly wheels, see the [Installation page](https://docs.ray.io/en/latest/ray-overview/installation.html).

## More Information

*   [Documentation](http://docs.ray.io/en/latest/index.html)
*   [Ray Architecture whitepaper](https://docs.google.com/document/d/1tBw9A4j62ruI5omIJbMxly-la5w4q_TjyJgJL_jN2fI/preview)
*   [Exoshuffle: large-scale data shuffle in Ray](https://arxiv.org/abs/2203.05072)
*   [Ownership: a distributed futures system for fine-grained tasks](https://www.usenix.org/system/files/nsdi21-wang.pdf)
*   [RLlib paper](https://arxiv.org/abs/1712.09381)
*   [Tune paper](https://arxiv.org/abs/1807.05118)

*Older documents:*

*   [Ray paper](https://arxiv.org/abs/1712.05889)
*   [Ray HotOS paper](https://arxiv.org/abs/1703.03924)
*   [Ray Architecture v1 whitepaper](https://docs.google.com/document/d/1lAy0Owi-vPz2jEqBSaHNQcy2IBSDEHyXNOQZlGuj93c/preview)

## Getting Involved

| Platform           | Purpose                                        | Estimated Response Time | Support Level |
| ------------------ | ---------------------------------------------- | ----------------------- | ------------- |
| [Discourse Forum](https://discuss.ray.io/)    | For discussions and questions about usage.       | < 1 day              | Community     |
| [GitHub Issues](https://github.com/ray-project/ray/issues)     | For reporting bugs and filing feature requests. | < 2 days              | Ray OSS Team  |
| [Slack](https://www.ray.io/join-slack?utm_source=github&utm_medium=ray_readme&utm_campaign=getting_involved)          | For collaborating with other Ray users.         | < 2 days              | Community     |
| [StackOverflow](https://stackoverflow.com/questions/tagged/ray)   | For asking questions about how to use Ray.    | 3-5 days              | Community     |
| [Meetup Group](https://www.meetup.com/Bay-Area-Ray-Meetup/)      | For learning about Ray projects and best practices. | Monthly               | Ray DevRel    |
| [Twitter](https://x.com/raydistributed)      | For staying up-to-date on new features.         | Daily                 | Ray DevRel    |