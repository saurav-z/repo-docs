[![Ray Logo](https://github.com/ray-project/ray/raw/master/doc/source/images/ray_header_logo.png)](https://github.com/ray-project/ray)

# Ray: The Unified Framework for Scaling AI and Python Applications

**Ray empowers you to effortlessly scale your Python and AI applications from your laptop to the cloud.**

Ray is a powerful, open-source framework designed for building and running distributed applications. It provides a unified platform for scaling your AI and Python workloads, simplifying the development process and enabling efficient resource utilization.

## Key Features

*   **Unified Compute:** Seamlessly scale Python code from a single machine to a cluster without code changes.
*   **AI Libraries:** Provides specialized libraries for key AI tasks:
    *   **Data:** Scalable Datasets for Machine Learning
    *   **Train:** Distributed Training
    *   **Tune:** Scalable Hyperparameter Tuning
    *   **RLlib:** Scalable Reinforcement Learning
    *   **Serve:** Scalable and Programmable Serving
*   **Core Abstractions:** Offers fundamental building blocks for distributed computing:
    *   **Tasks:** Stateless functions executed in the cluster.
    *   **Actors:** Stateful worker processes created in the cluster.
    *   **Objects:** Immutable values accessible across the cluster.
*   **Monitoring and Debugging:**
    *   Monitor Ray apps and clusters with the [Ray Dashboard](https://docs.ray.io/en/latest/ray-core/ray-dashboard.html).
    *   Debug Ray apps with the [Ray Distributed Debugger](https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html).
*   **Flexible Deployment:** Runs on any machine, cluster, cloud provider, and Kubernetes.
*   **Extensive Ecosystem:** Integrates with a growing `ecosystem of community integrations`_.

## Why Choose Ray?

Modern machine learning workloads are computationally intensive.  Ray provides a unified solution to scale your Python and AI applications, eliminating the need for infrastructure changes.  Develop locally and then scale your code effortlessly.

## Installation

Install Ray using pip:

```bash
pip install ray
```

For nightly builds, see the [Installation page](https://docs.ray.io/en/latest/ray-overview/installation.html).

## Resources

*   [Documentation](http://docs.ray.io/en/latest/index.html)
*   [Ray Architecture Whitepaper](https://docs.google.com/document/d/1tBw9A4j62ruI5omIJbMxly-la5w4q_TjyJgJL_jN2fI/preview)
*   [Ray Architecture v1 whitepaper](https://docs.google.com/document/d/1lAy0Owi-vPz2jEqBSaHNQcy2IBSDEHyXNOQZlGuj93c/preview)
*   [Exoshuffle: large-scale data shuffle in Ray](https://arxiv.org/abs/2203.05072)
*   [Ownership: a distributed futures system for fine-grained tasks](https://www.usenix.org/system/files/nsdi21-wang.pdf)
*   [Ray paper](https://arxiv.org/abs/1712.05889)
*   [Ray HotOS paper](https://arxiv.org/abs/1703.03924)
*   [RLlib paper](https://arxiv.org/abs/1712.09381)
*   [Tune paper](https://arxiv.org/abs/1807.05118)

## Getting Involved

Join the Ray community and get support through these channels:

| Platform              | Purpose                                      | Estimated Response Time | Support Level |
|-----------------------|----------------------------------------------|-------------------------|---------------|
| [Discourse Forum](https://discuss.ray.io/)          | Discussions and usage questions.  | < 1 day                  | Community     |
| [GitHub Issues](https://github.com/ray-project/ray/issues)         | Bug reports and feature requests.    | < 2 days                  | Ray OSS Team  |
| [Slack](https://www.ray.io/join-slack?utm_source=github&utm_medium=ray_readme&utm_campaign=getting_involved)            | Collaborating with other users.           | < 2 days                  | Community     |
| [StackOverflow](https://stackoverflow.com/questions/tagged/ray)    | Questions about Ray usage.               | 3-5 days                 | Community     |
| [Meetup Group](https://www.meetup.com/Bay-Area-Ray-Meetup/)        | Learning about Ray best practices.       | Monthly                   | Ray DevRel    |
| [Twitter](https://x.com/raydistributed)             | Stay up-to-date on new features.        | Daily                    | Ray DevRel    |

## Learn More

Explore the [Ray Project](https://github.com/ray-project/ray) to start scaling your applications today.
```

Key improvements and explanations:

*   **SEO Optimization:** The title now incorporates relevant keywords like "Unified Framework," "Scaling," "AI," and "Python." The use of headings and subheadings helps with SEO and readability.
*   **Hook:** The one-sentence hook grabs the reader's attention and clearly states the value proposition.
*   **Clear Structure:** The content is organized into sections with clear headings and bullet points, making it easy to scan and understand.
*   **Key Features:** The "Key Features" section highlights the main benefits of using Ray.
*   **Why Ray?**:  This section directly addresses the user's potential needs and pain points, emphasizing the core value proposition.
*   **Call to Action:** The "Learn More" section with a link to the original repo encourages users to engage with the project.
*   **Concise Language:** The text is rewritten to be more concise and to the point.
*   **Removed redundant images:**  Simplified the initial image section.
*   **Improved Links:**  Clearer descriptions and direct links.
*   **Table for Community:** Added a table to display the different ways to get involved and support levels.