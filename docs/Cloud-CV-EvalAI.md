# EvalAI: The Open-Source Platform for AI Algorithm Evaluation

**EvalAI empowers researchers to evaluate and compare machine learning and AI algorithms at scale, fostering reproducible research and accelerating AI advancements.**  Check out the original repository [here](https://github.com/Cloud-CV/EvalAI).

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

*   **Customizable Evaluation:** Create and manage evaluation phases, dataset splits, and leaderboards with support for any programming language.
*   **Remote Evaluation:** Leverage powerful compute resources for challenges requiring extensive computational power.
*   **Dockerized Evaluation:** Submit code as Docker images for secure and reproducible evaluation within defined environments.
*   **Command-Line Interface (CLI):**  Utilize the `evalai-cli` to interact with EvalAI from your terminal, streamlining workflows.
*   **Scalability & Portability:** Built on open-source technologies like Docker, Django, Node.js, and PostgreSQL, ensuring flexibility and adaptability.
*   **Optimized Performance:** Achieve faster evaluation times through worker node warm-up, dataset chunking, and parallel processing.

## Goal

Our mission is to build a central hub for AI challenges, fostering collaboration and accelerating progress in the field of artificial intelligence.

## Installation

Easily set up EvalAI locally using Docker:

1.  **Install Docker and Docker Compose:** Follow the instructions for your operating system:
    *   [Docker Installation](https://docs.docker.com/install/)
    *   [Docker Compose Installation](https://docs.docker.com/compose/install/)
2.  **Get the Source Code:**
    ```bash
    git clone https://github.com/Cloud-CV/EvalAI.git evalai && cd evalai
    ```
3.  **Build and Run Containers:**
    ```bash
    docker-compose up --build
    ```
    *   To start worker services: `docker-compose --profile worker up --build`
    *   To start statsd-exporter: `docker-compose --profile statsd up --build`
    *   To start both optional services: `docker-compose --profile worker --profile statsd up --build`
4.  **Access EvalAI:** Open your web browser and go to `http://127.0.0.1:8888`.

    Default user credentials:
    *   **SUPERUSER:** username: `admin`, password: `password`
    *   **HOST USER:** username: `host`, password: `password`
    *   **PARTICIPANT USER:** username: `participant`, password: `password`

    If you encounter any issues, see the [common errors during installation](https://evalai.readthedocs.io/en/latest/faq(developers).html#common-errors-during-installation) page.

## Documentation

Find specific setup instructions for the documentation in `docs/README.md`.

## Citing EvalAI

If you use EvalAI in your research, please cite the following technical report:

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

EvalAI is actively maintained by [Rishabh Jain](https://rishabhjain.xyz/), [Gunjan Chhablani](https://gchhablani.github.io/), and [Dhruv Batra](https://www.cc.gatech.edu/~dbatra/).

## Contributions

Contribute to EvalAI by following the [contribution guidelines](https://github.com/Cloud-CV/EvalAI/blob/master/.github/CONTRIBUTING.md).

## Contributors

<!-- //]: contributor-faces -->
<!-- Added a full list of contributors here -->