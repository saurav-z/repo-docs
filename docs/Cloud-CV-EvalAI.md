# EvalAI: The Open-Source Platform for AI and Machine Learning Evaluation

**EvalAI is your one-stop solution for evaluating and comparing machine learning (ML) and artificial intelligence (AI) algorithms at scale.**

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

## Key Features

*   **Customizable Evaluation:** Define your own evaluation protocols, phases, and dataset splits. Compatible with any programming language.
*   **Remote Evaluation:** Leverage specialized compute for large-scale challenges.
*   **Dockerized Environments:** Submit code as Docker images for consistent evaluation across different environments.
*   **CLI Support:** Use the `evalai-cli` for streamlined interaction from your command line.
*   **Scalable and Portable:** Built with open-source technologies like Docker, Django, Node.js, and PostgreSQL.
*   **Optimized Performance:** Faster evaluation through worker node warm-up and dataset chunking.

## Why Use EvalAI?

EvalAI streamlines the process of comparing AI and ML algorithms by providing a centralized platform for hosting challenges, managing submissions, and generating reliable leaderboards. This enables researchers to reproduce results and perform accurate analyses.

## Installation

Get started with EvalAI by following these simple steps:

1.  **Install Docker and Docker Compose:** Make sure you have both Docker and Docker Compose installed on your machine.  Refer to the [Docker documentation](https://docs.docker.com/install/) and [Docker Compose documentation](https://docs.docker.com/compose/install/) for details.
2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Cloud-CV/EvalAI.git evalai && cd evalai
    ```
3.  **Build and Run:**
    ```bash
    docker-compose up --build
    ```
    *   To include worker services: `docker-compose --profile worker up --build`
    *   To include statsd-exporter: `docker-compose --profile statsd up --build`
    *   To include both: `docker-compose --profile worker --profile statsd up --build`
4.  **Access EvalAI:** Open your web browser and go to [http://127.0.0.1:8888](http://127.0.0.1:8888).  Use the following default credentials:
    *   **SUPERUSER:** username: `admin`, password: `password`
    *   **HOST USER:** username: `host`, password: `password`
    *   **PARTICIPANT USER:** username: `participant`, password: `password`

For troubleshooting, see the [common errors during installation](https://evalai.readthedocs.io/en/latest/faq(developers).html#common-errors-during-installation) page.

## Documentation

Refer to the `docs/README.md` file for instructions on setting up the documentation builder locally if you're contributing to the EvalAI documentation.

## Citing EvalAI

If you use EvalAI, please cite our technical report:

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

We welcome contributions! Please review our [contribution guidelines](https://github.com/Cloud-CV/EvalAI/blob/master/.github/CONTRIBUTING.md) before submitting.

## Get Involved

Explore the platform and contribute to the development of EvalAI!

**[Visit the EvalAI Repository on GitHub](https://github.com/Cloud-CV/EvalAI)**