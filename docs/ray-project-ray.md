<div align="center">
  <img src="https://github.com/ray-project/ray/raw/master/doc/source/images/ray_header_logo.png" alt="Ray Logo" width="400"/>
</div>

# Ray: Scale Your AI and Python Applications

**Ray is a powerful open-source framework that simplifies distributed computing, enabling you to scale your Python and AI applications from a laptop to a cluster.**

[<img src="https://readthedocs.org/projects/ray/badge/?version=master" alt="Documentation Status" />](http://docs.ray.io/en/master/?badge=master)
[<img src="https://img.shields.io/badge/Join%20Slack-blue" alt="Join Slack" />](https://www.ray.io/join-slack)
[<img src="https://img.shields.io/badge/Discuss-Ask%20Questions-blue" alt="Discuss" />](https://discuss.ray.io/)
[<img src="https://img.shields.io/twitter/follow/raydistributed.svg?style=social&logo=twitter" alt="Follow on Twitter" />](https://x.com/raydistributed)
[<img src="https://img.shields.io/badge/Get_started_for_free-3C8AE9?logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8%2F9hAAAAAXNSR0IArs4c6QAAAERlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMAAAABAAEAAKACAAQAAAABAAAAEKADAAQAAAABAAAAEAAAAAA0VXHyAAABKElEQVQ4Ea2TvWoCQRRGnWCVWChIIlikC9hpJdikSbGgaONbpAoY8gKBdAGfwkfwKQypLQ1sEGyMYhN1Pd%2B6A8PqwBZeOHt%2FvsvMnd3ZXBRFPQjBZ9K6OY8ZxF%2B0IYw9PW3qz8aY6lk92bZ%2BVqSI3oC9T7%2FyCVnrF1ngj93us%2B540sf5BrCDfw9b6jJ5lx%2FyjtGKBBXc3cnqx0INN4ImbI%2Bl%2BPnI8zWfFEr4chLLrWHCp9OO9j19Kbc91HX0zzzBO8EbLK2Iv4ZvNO3is3h6jb%2BCwO0iL8AaWqB7ILPTxq3kDypqvBuYuwswqo6wgYJbT8XxBPZ8KS1TepkFdC79TAHHce%2F7LbVioi3wEfTpmeKtPRGEeoldSP%2FOeoEftpP4BRbgXrYZefsAI%2BP9JU7ImyEAAAAASUVORK5CYII%3D" alt="Get Started" />](https://www.anyscale.com/ray-on-anyscale?utm_source=github&utm_medium=ray_readme&utm_campaign=get_started_badge)

## Key Features

Ray is a unified framework for scaling AI and Python applications, providing a robust core and a suite of specialized AI libraries.

*   **Seamless Scaling:** Run the same code from your laptop to a cluster with minimal changes.
*   **General-Purpose:**  Capable of running diverse workloads.
*   **Python-Native:** Scale your Python applications without needing to learn new infrastructure.
*   **AI Libraries:** Access powerful libraries for Data, Training, Tuning, Reinforcement Learning, and Serving.
*   **Ray Core Abstractions:** Leverage Tasks, Actors, and Objects for efficient distributed computation.
*   **Monitoring and Debugging:** Utilize the Ray Dashboard and Distributed Debugger for app and cluster management.
*   **Broad Compatibility:** Runs on any machine, cloud provider, Kubernetes, and more.
*   **Extensive Integrations:** Enjoy a growing ecosystem of community integrations.

## Get Started

Install Ray with: `pip install ray`

For nightly wheels, see the [Installation page](https://docs.ray.io/en/latest/ray-overview/installation.html).

## Ray AI Libraries

Ray offers a suite of libraries specifically designed for AI workloads:

*   [Data](https://docs.ray.io/en/latest/data/dataset.html): Scalable Datasets for ML
*   [Train](https://docs.ray.io/en/latest/train/train.html): Distributed Training
*   [Tune](https://docs.ray.io/en/latest/tune/index.html): Scalable Hyperparameter Tuning
*   [RLlib](https://docs.ray.io/en/latest/rllib/index.html): Scalable Reinforcement Learning
*   [Serve](https://docs.ray.io/en/latest/serve/index.html): Scalable and Programmable Serving

## Ray Core

Ray Core provides the fundamental building blocks for distributed computing:

*   [Tasks](https://docs.ray.io/en/latest/ray-core/tasks.html): Stateless functions executed in the cluster.
*   [Actors](https://docs.ray.io/en/latest/ray-core/actors.html): Stateful worker processes created in the cluster.
*   [Objects](https://docs.ray.io/en/latest/ray-core/objects.html): Immutable values accessible across the cluster.

## Why Ray?

Modern ML workloads are compute-intensive, and single-node environments are often insufficient. Ray solves this by offering a unified way to scale your Python and AI applications, simplifying the transition from local development to a distributed environment. It's designed to be general-purpose, so you can scale any Python application with Ray.

## Monitoring and Debugging

*   Monitor Ray apps and clusters with the [Ray Dashboard](https://docs.ray.io/en/latest/ray-core/ray-dashboard.html).
*   Debug Ray apps with the [Ray Distributed Debugger](https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html).

## More Information

*   [Documentation](http://docs.ray.io/en/latest/index.html)
*   [Ray Architecture whitepaper](https://docs.google.com/document/d/1tBw9A4j62ruI5omIJbMxly-la5w4q_TjyJgJL_jN2fI/preview)
*   [Exoshuffle: large-scale data shuffle in Ray](https://arxiv.org/abs/2203.05072)
*   [Ownership: a distributed futures system for fine-grained tasks](https://www.usenix.org/system/files/nsdi21-wang.pdf)
*   [RLlib paper](https://arxiv.org/abs/1712.09381)
*   [Tune paper](https://arxiv.org/abs/1807.05118)

## Getting Involved

| Platform          | Purpose                                                    | Estimated Response Time | Support Level |
|-------------------|------------------------------------------------------------|-------------------------|---------------|
| Discourse Forum   | Discussions about development and questions about usage.    | < 1 day                 | Community     |
| GitHub Issues     | Reporting bugs and filing feature requests.             | < 2 days                | Ray OSS Team  |
| Slack             | Collaborating with other Ray users.                      | < 2 days                | Community     |
| StackOverflow     | Asking questions about how to use Ray.                   | 3-5 days                | Community     |
| Meetup Group      | Learning about Ray projects and best practices.           | Monthly                 | Ray DevRel    |
| Twitter           | Staying up-to-date on new features.                       | Daily                   | Ray DevRel    |

*   [Discourse Forum](https://discuss.ray.io/)
*   [GitHub Issues](https://github.com/ray-project/ray/issues)
*   [StackOverflow](https://stackoverflow.com/questions/tagged/ray)
*   [Meetup Group](https://www.meetup.com/Bay-Area-Ray-Meetup/)
*   [Twitter](https://x.com/raydistributed)
*   [Slack](https://www.ray.io/join-slack?utm_source=github&utm_medium=ray_readme&utm_campaign=getting_involved)

[Back to Top](#ray-scale-your-ai-and-python-applications)
```
Key improvements and SEO considerations:

*   **Clear Headings:** Used `H2` for main sections, improving readability and SEO.
*   **Keyword Optimization:**  Incorporated keywords like "Ray," "AI," "Python," "distributed computing," and "scale" naturally throughout the text.
*   **Bulleted Key Features:**  Highlights the core benefits of Ray in an easy-to-scan format.
*   **Concise Descriptions:**  Provides brief, informative descriptions for each section.
*   **Call to Action:** Encourages users to install and explore Ray.
*   **Links:**  Maintained all the original links while ensuring they're clearly labeled and formatted.
*   **Back to Top link** Added a 'back to top' link.
*   **Visuals:**Kept the logo at the top and all other images, keeping the original look and feel.