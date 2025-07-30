<div align="center">
  <a href="https://github.com/ray-project/ray">
    <img src="https://github.com/ray-project/ray/raw/master/doc/source/images/ray_header_logo.png" alt="Ray Logo" width="400"/>
  </a>
</div>

<div align="center">
  <a href="http://docs.ray.io/en/master/?badge=master">
    <img src="https://readthedocs.org/projects/ray/badge/?version=master" alt="Documentation"/>
  </a>
  <a href="https://www.ray.io/join-slack">
    <img src="https://img.shields.io/badge/Ray-Join%20Slack-blue" alt="Join Slack"/>
  </a>
  <a href="https://discuss.ray.io/">
    <img src="https://img.shields.io/badge/Discuss-Ask%20Questions-blue" alt="Discuss"/>
  </a>
  <a href="https://x.com/raydistributed">
    <img src="https://img.shields.io/twitter/follow/raydistributed.svg?style=social&logo=twitter" alt="Follow on Twitter"/>
  </a>
  <a href="https://www.anyscale.com/ray-on-anyscale?utm_source=github&utm_medium=ray_readme&utm_campaign=get_started_badge">
    <img src="https://img.shields.io/badge/Get_started_for_free-3C8AE9?logo=data%3Aimage%2Fpng%3B64%2CiVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8%2F9hAAAAAXNSR0IArs4c6QAAAERlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMAAAABAAEAAKACAAQAAAABAAAAEKADAAQAAAABAAAAEAAAAAA0VXHyAAABKElEQVQ4Ea2TvWoCQRRGnWCVWChIIlikC9hpJdikSbGgaONbpAoY8gKBdAGfwkfwKQypLQ1sEGyMYhN1Pd%2B6A8PqwBZeOHt%2FvsvMnd3ZXBRFPQjBZ9K6OY8ZxF%2B0IYw9PW3qz8aY6lk92bZ%2BVqSI3oC9T7%2FyCVnrF1ngj93us%2B540sf5BrCDfw9b6jJ5lx%2FyjtGKBBXc3cnqx0INN4ImbI%2Bl%2BPnI8zWfFEr4chLLrWHCp9OO9j19Kbc91HX0zzzBO8EbLK2Iv4ZvNO3is3h6jb%2BCwO0iL8AaWqB7ILPTxq3kDypqvBuYuwswqo6wgYJbT8XxBPZ8KS1TepkFdC79TAHHce%2F7LbVioi3wEfTpmeKtPRGEeoldSP%2FOeoEftpP4BRbgXrYZefsAI%2BP9JU7ImyEAAAAASUVORK5CYII%3D" alt="Get Started"/>
  </a>
</div>

**Ray is a unified framework that lets you easily scale your Python and AI applications, from your laptop to the cloud.**

## Key Features

*   **Unified Framework:** Ray provides a single, consistent framework for scaling diverse workloads, including machine learning, reinforcement learning, and general Python applications.
*   **Scalable AI Libraries:** Ray offers a suite of libraries to simplify ML compute:
    *   **Data:** Scalable Datasets for ML.
    *   **Train:** Distributed Training.
    *   **Tune:** Scalable Hyperparameter Tuning.
    *   **RLlib:** Scalable Reinforcement Learning.
    *   **Serve:** Scalable and Programmable Serving.
*   **Ray Core Abstractions:**
    *   **Tasks:** Stateless functions executed in the cluster.
    *   **Actors:** Stateful worker processes created in the cluster.
    *   **Objects:** Immutable values accessible across the cluster.
*   **Flexible Deployment:** Ray runs on any machine, cluster, cloud provider, and Kubernetes.
*   **Monitoring and Debugging:** Monitor and debug your Ray applications using the Ray Dashboard and Distributed Debugger.

## Getting Started

Install Ray using pip:

```bash
pip install ray
```

For nightly builds, refer to the [Installation Guide](https://docs.ray.io/en/latest/ray-overview/installation.html).

## Why Ray?

Ray is designed to address the increasing compute demands of modern ML workloads. It enables you to seamlessly scale your Python code from a single machine to a distributed cluster, without significant code changes. Ray's general-purpose design ensures it can handle a wide range of workloads, eliminating the need for complex infrastructure setups.

## More Information

*   [Documentation](http://docs.ray.io/en/latest/index.html)
*   [Ray Architecture Whitepaper](https://docs.google.com/document/d/1tBw9A4j62ruI5omIJbMxly-la5w4q_TjyJgJL_jN2fI/preview)

## Getting Involved

| Platform           | Purpose                                      | Estimated Response Time | Support Level |
| ------------------ | -------------------------------------------- | ----------------------- | ------------- |
| [Discourse Forum](https://discuss.ray.io/) | For discussions and usage questions | < 1 day                 | Community     |
| [GitHub Issues](https://github.com/ray-project/ray/issues) | Report bugs and feature requests  | < 2 days                | Ray OSS Team  |
| [Slack](https://www.ray.io/join-slack?utm_source=github&utm_medium=ray_readme&utm_campaign=getting_involved)      | Collaborate with other users       | < 2 days                | Community     |
| [StackOverflow](https://stackoverflow.com/questions/tagged/ray)     | Ask usage questions                | 3-5 days                | Community     |
| [Meetup Group](https://www.meetup.com/Bay-Area-Ray-Meetup/)   | Learn about projects & best practices     | Monthly                 | Ray DevRel    |
| [Twitter](https://x.com/raydistributed)       | Stay up-to-date on new features         | Daily                   | Ray DevRel    |
```
Key improvements and explanations:

*   **SEO Optimization:** Added relevant keywords like "Python", "AI", "machine learning", and "distributed" throughout the README. Included more descriptive section headings.
*   **Concise Hook:**  The one-sentence hook immediately grabs the reader's attention and explains the core value proposition of Ray.
*   **Clear Headings:**  Uses descriptive headings to structure the information, making it easier to scan and understand.
*   **Bulleted Key Features:**  Highlights the most important features of Ray in a clear and easy-to-read format.
*   **Direct Links:** All links are included for easy access to documentation, community resources, and installation instructions.
*   **"Why Ray?" Section:**  Provides a concise explanation of the problem Ray solves and the benefits it offers.
*   **Getting Involved Section:** Presents community resources in a structured, easy-to-scan table format.
*   **Concise & Focused:** Removed less critical information from the original and prioritized the most important details to improve scannability.
*   **Image Alt Text:** Added descriptive alt text to all images for accessibility and SEO.