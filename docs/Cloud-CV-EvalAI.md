<p align="center"><img width="65%" src="docs/source/_static/img/evalai_logo.png"/></p>

# EvalAI: The Open-Source Platform for AI Algorithm Evaluation and Benchmarking

**EvalAI** is your all-in-one solution for evaluating and comparing machine learning and AI algorithms, fostering collaboration and accelerating innovation.  Find the original repository [here](https://github.com/Cloud-CV/EvalAI).

[![Join the chat on Slack](https://img.shields.io/badge/Join%20Slack-Chat-blue?logo=slack)](https://join.slack.com/t/cloudcv-community/shared_invite/zt-3252n6or8-e0QuZKIZFLB0zXtQ6XgxfA)
[![Build Status](https://travis-ci.org/Cloud-CV/EvalAI.svg?branch=master)](https://travis-ci.org/Cloud-CV/EvalAI)
[![Coverage](https://img.shields.io/codecov/c/github/Cloud-CV/EvalAI?label=Coverage&style=flat-square)](https://codecov.io/gh/Cloud-CV/EvalAI)
[![Backend Coverage](https://img.shields.io/codecov/c/github/Cloud-CV/EvalAI?flag=backend&label=Backend&style=flat-square)](https://codecov.io/gh/Cloud-CV/EvalAI?flag=backend)
[![Frontend Coverage](https://img.shields.io/codecov/c/github/Cloud-CV/EvalAI?flag=frontend&label=Frontend&style=flat-square)](https://codecov.io/gh/Cloud-CV/EvalAI?flag=frontend)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/markdown-guide/badge/?version=latest)](http://evalai.readthedocs.io/en/latest/)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/Cloud-CV/EvalAI?style=flat-square)](https://github.com/Cloud-CV/EvalAI/tree/master)
[![Open Collective](https://opencollective.com/evalai/backers/badge.svg)](https://opencollective.com/evalai#backers)
[![Open Collective](https://opencollective.com/evalai/sponsors/badge.svg)](https://opencollective.com/evalai#sponsors)
[![Twitter Follow](https://img.shields.io/twitter/follow/eval_ai?style=social)](https://twitter.com/eval_ai)

## Key Features

*   **Custom Evaluation Protocols:** Define flexible evaluation phases, dataset splits, and leaderboards tailored to your specific needs.
*   **Remote Evaluation:** Leverage dedicated compute resources for large-scale challenges, allowing you to scale your evaluation infrastructure seamlessly.
*   **Environment-Based Evaluation:** Utilize Docker containers to safely evaluate submissions within defined environments, ensuring reproducibility.
*   **CLI Support:** Enhance your workflow with the evalai-cli, providing command-line access and simplifying platform interaction.
*   **Portability:** Built on open-source technologies like Docker, Django, Node.js, and PostgreSQL, EvalAI ensures scalability and platform independence.
*   **Faster Evaluation:** Optimize evaluation speed through techniques like worker node pre-warming, challenge code importing, and dataset chunking, reducing evaluation time significantly.

## Why Use EvalAI?

EvalAI simplifies the process of evaluating and comparing AI algorithms by providing a centralized platform. This enables researchers to:

*   **Reproduce Results:** Easily reproduce results from research papers.
*   **Perform Reliable Analysis:** Conduct accurate and reliable quantitative analyses.
*   **Benchmark Progress:** Contribute to and track advancements in the field of AI.

## Installation

Getting EvalAI up and running locally is straightforward:

1.  **Prerequisites:** Install [Docker](https://docs.docker.com/install/) and [Docker Compose](https://docs.docker.com/compose/install/).
2.  **Get the Source Code:**
    ```bash
    git clone https://github.com/Cloud-CV/EvalAI.git evalai && cd evalai
    ```
3.  **Build and Run:**
    ```bash
    docker-compose up --build
    ```
    (To include worker services, use `docker-compose --profile worker up --build`.)
4.  **Access EvalAI:** Open your web browser and navigate to <http://127.0.0.1:8888>. Default users:
    *   **SUPERUSER:** `admin` / `password`
    *   **HOST:** `host` / `password`
    *   **PARTICIPANT:** `participant` / `password`

If you encounter any issues, consult the [common errors during installation](https://evalai.readthedocs.io/en/latest/faq(developers).html#common-errors-during-installation) page.

## Contributing to EvalAI

We welcome contributions!  Please refer to our [contribution guidelines](https://github.com/Cloud-CV/EvalAI/blob/master/.github/CONTRIBUTING.md).

## Citing EvalAI

If you use EvalAI for hosting challenges, please cite the following technical report:

```
@article{EvalAI,
    title   =  {EvalAI: Towards Better Evaluation Systems for AI Agents},
    author  =  {Deshraj Yadav and Rishabh Jain and Harsh Agrawal and Prithvijit
                Chattopadhyay and Taranjeet Singh and Akash Jain and Shiv Baran
                Singh and Stefan Lee and Dhruv Batra},
    year    =  {2019},
    volume  =  arXiv:1902.03570
}
```
<p>
    <a href="http://learningsys.org/sosp19/assets/papers/23_CameraReadySubmission_EvalAI_SOSP_2019%20(8)%20(1).pdf" target="_blank"><img src="docs/source/_static/img/evalai-paper.jpg"/></a>
</p>

## Team

EvalAI is maintained by [Rishabh Jain](https://rishabhjain.xyz/), [Gunjan Chhablani](https://gchhablani.github.io/), and [Dhruv Batra](https://www.cc.gatech.edu/~dbatra/).

## Contributors

[//]: contributor-faces

```
(Contributor faces here)