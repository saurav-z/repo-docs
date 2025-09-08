<p align="center">
  <img width="65%" src="docs/source/_static/img/evalai_logo.png" alt="EvalAI Logo"/>
</p>

# EvalAI: The Open-Source Platform for AI and Machine Learning Evaluation

**EvalAI is a powerful, open-source platform designed to streamline the evaluation and comparison of machine learning (ML) and artificial intelligence (AI) algorithms at scale, offering a centralized hub for AI challenge participation and collaboration.** ([See the original repository](https://github.com/Cloud-CV/EvalAI))

[![Join Slack](https://img.shields.io/badge/Join%20Slack-Chat-blue?logo=slack)](https://join.slack.com/t/cloudcv-community/shared_invite/zt-3252n6or8-e0QuZKIZFLB0zXtQ6XgxfA)
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

EvalAI simplifies the process of comparing AI algorithms by providing a centralized platform for challenges, leaderboards, and standardized evaluation, ensuring reliable and accurate results.

## Key Features of EvalAI:

*   **Customizable Evaluation Protocols:** Define multiple evaluation phases, dataset splits, and use any programming language, with public and private leaderboards.
*   **Remote Evaluation:** Leverage specialized compute capabilities for large-scale challenges with easy integration of custom worker nodes.
*   **Dockerized Evaluation:** Submit your AI agent's code as Docker images, allowing for consistent and reproducible evaluation in isolated environments.
*   **CLI Support:** Enhance platform accessibility with the `evalai-cli` for command-line interaction.
*   **Scalability & Portability:** Built with open-source technologies (Docker, Django, Node.js, PostgreSQL) for easy deployment and scalability.
*   **Faster Evaluation:** Optimized performance through worker node warm-up, challenge code import, in-memory dataset preloading and parallel evaluation using dataset chunking, reducing evaluation time.

## Goal

Our ultimate goal is to establish a centralized platform for hosting, participating in, and collaborating on AI challenges globally, contributing to significant progress in AI benchmarking.

## Installation Instructions

Get started with EvalAI quickly using Docker:

1.  **Install Docker and Docker Compose:** Ensure you have [Docker](https://docs.docker.com/install/) and [Docker Compose](https://docs.docker.com/compose/install/) installed on your machine.
2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Cloud-CV/EvalAI.git evalai && cd evalai
    ```
3.  **Build and Run Containers:**
    ```bash
    docker-compose up --build
    ```
    *   For worker services: `docker-compose --profile worker up --build`
    *   For statsd-exporter: `docker-compose --profile statsd up --build`
    *   For both optional services: `docker-compose --profile worker --profile statsd up --build`

4.  **Access EvalAI:** Open your web browser and navigate to <http://127.0.0.1:8888>. Default user credentials are provided in the original README.

*   **SUPERUSER:** username: `admin` password: `password`
*   **HOST USER:** username: `host` password: `password`
*   **PARTICIPANT USER:** username: `participant` password: `password`

For troubleshooting, refer to the [common errors during installation](https://evalai.readthedocs.io/en/latest/faq(developers).html#common-errors-during-installation) documentation.

## Documentation Setup for Contributions

Refer to `docs/README.md` within the repository for detailed instructions on setting up the documentation builder if you wish to contribute.

## Citing EvalAI

If you use EvalAI, please cite the following:

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

## Contribution Guidelines

Contribute to EvalAI by following the [contribution guidelines](https://github.com/Cloud-CV/EvalAI/blob/master/.github/CONTRIBUTING.md).

## Contributors

[//]: contributor-faces
(List of contributors - see original README)

```