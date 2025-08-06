<div align="center">
  <img src="https://github.com/ray-project/ray/raw/master/doc/source/images/ray_header_logo.png" alt="Ray Logo">
</div>

# Ray: The Unified Framework for Scaling AI and Python Applications

**Ray empowers you to effortlessly scale your Python and AI applications from a single laptop to a massive cluster.**

Ray is a powerful open-source framework designed to simplify the process of building and scaling distributed applications. It provides a unified platform for a wide range of tasks, including:

## Key Features

*   **Scalable AI Libraries:** Leverage specialized libraries for machine learning tasks:
    *   **Data:** Scalable Datasets for ML
    *   **Train:** Distributed Training
    *   **Tune:** Scalable Hyperparameter Tuning
    *   **RLlib:** Scalable Reinforcement Learning
    *   **Serve:** Scalable and Programmable Serving
*   **Ray Core Abstractions:** Utilize core components for distributed computing:
    *   **Tasks:** Stateless functions executed in the cluster.
    *   **Actors:** Stateful worker processes created in the cluster.
    *   **Objects:** Immutable values accessible across the cluster.
*   **Flexible Deployment:** Run Ray on any machine, cluster, cloud provider, and Kubernetes.
*   **Monitoring and Debugging:** Monitor and debug your Ray applications:
    *   **Ray Dashboard:** Monitor Ray apps and clusters.
    *   **Ray Distributed Debugger:** Debug Ray applications.
*   **Extensive Ecosystem:** Benefit from a growing ecosystem of community integrations.

## Why Choose Ray?

Traditional single-node development environments struggle to handle the demands of modern, compute-intensive ML workloads. Ray provides a unified solution for scaling Python and AI applications seamlessly, allowing you to:

*   **Scale effortlessly:** Move your code from your laptop to a cluster without significant changes.
*   **General Purpose:** Execute any Python workload.
*   **Maximize Performance:** Ray is designed to run any type of workload.

## Getting Started

Install Ray with:
```bash
pip install ray
```
For nightly builds, see the [Installation page](https://docs.ray.io/en/latest/ray-overview/installation.html).

## More Information

*   [Documentation](http://docs.ray.io/en/latest/index.html)
*   [Ray Architecture whitepaper](https://docs.google.com/document/d/1tBw9A4j62ruI5omIJbMxly-la5w4q_TjyJgJL_jN2fI/preview)
*   [Exoshuffle: large-scale data shuffle in Ray](https://arxiv.org/abs/2203.05072)
*   [Ownership: a distributed futures system for fine-grained tasks](https://www.usenix.org/system/files/nsdi21-wang.pdf)
*   [RLlib paper](https://arxiv.org/abs/1712.09381)
*   [Tune paper](https://arxiv.org/abs/1807.05118)

## Getting Involved

Join the Ray community and get help through:

| Platform          | Purpose                                       | Estimated Response Time | Support Level |
| :---------------- | :-------------------------------------------- | :---------------------- | :------------ |
| [Discourse Forum](https://discuss.ray.io/) | Discussions about development and usage | < 1 day               | Community     |
| [GitHub Issues](https://github.com/ray-project/ray/issues)  | Reporting bugs and feature requests       | < 2 days              | Ray OSS Team  |
| [Slack](https://www.ray.io/join-slack?utm_source=github&utm_medium=ray_readme&utm_campaign=getting_involved)     | Collaborating with other Ray users         | < 2 days              | Community     |
| [StackOverflow](https://stackoverflow.com/questions/tagged/ray)    | Asking questions about how to use Ray     | 3-5 days               | Community     |
| [Meetup Group](https://www.meetup.com/Bay-Area-Ray-Meetup/)       | Learning about Ray projects and best practices | Monthly                 | Ray DevRel    |
| [Twitter](https://x.com/raydistributed)        | Staying up-to-date on new features          | Daily                   | Ray DevRel    |

[Visit the Ray Project on GitHub](https://github.com/ray-project/ray)