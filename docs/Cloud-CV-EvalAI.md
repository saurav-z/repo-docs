<p align="center"><img width="65%" src="docs/source/_static/img/evalai_logo.png"/></p>

# EvalAI: Your Open-Source Platform for AI Algorithm Evaluation

**EvalAI is the go-to open-source platform for evaluating and comparing machine learning and AI algorithms at scale, streamlining the AI challenge process for researchers and developers.**  [Visit the EvalAI Repository](https://github.com/Cloud-CV/EvalAI)

[![Join the chat on Slack](https://img.shields.io/badge/Join%20Slack-Chat-blue?logo=slack)](https://join.slack.com/t/cloudcv-community/shared_invite/zt-3252n6or8-e0QuZKIZFLB0zXtQ6XgxfA)
[![Build Status](https://travis-ci.org/Cloud-CV/EvalAI.svg?branch=master)](https://travis-ci.org/Cloud-CV/EvalAI)
[![Coverage](https://img.shields.io/codecov/c/github/Cloud-CV/EvalAI?label=Coverage&style=flat-square)](https://codecov.io/gh/Cloud-CV/EvalAI)
[![Backend Coverage](https://img.shields.io/codecov/c/github/Cloud-CV/EvalAI?flag=backend&label=Backend&style=flat-square)](https://codecov.io/gh/Cloud-CV/EvalAI?flag=backend)
[![Frontend Coverage](https://img.shields.io/codecov/c/github/Cloud-CV/EvalAI?flag=frontend&label=Frontend&style=flat-square)](https://codecov.io/gh/EvalAI?flag=frontend)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/markdown-guide/badge/?version=latest)](http://evalai.readthedocs.io/en/latest/)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/Cloud-CV/EvalAI?style=flat-square)](https://github.com/Cloud-CV/EvalAI/tree/master)
[![Open Collective](https://opencollective.com/evalai/backers/badge.svg)](https://opencollective.com/evalai#backers)
[![Open Collective](https://opencollective.com/evalai/sponsors/badge.svg)](https://opencollective.com/evalai#sponsors)
[![Twitter Follow](https://img.shields.io/twitter/follow/eval_ai?style=social)](https://twitter.com/eval_ai)


## Key Features of EvalAI

*   **Customizable Evaluation Protocols:** Create flexible evaluation phases, dataset splits, and leaderboards with support for any programming language.
*   **Remote Evaluation:** Leverage specialized compute resources for large-scale challenges with worker node integration.
*   **Containerized Evaluation:** Submit and evaluate code within Docker images, ensuring consistent and reproducible results within defined test environments.
*   **Command-Line Interface (CLI) Support:** Enhance your workflow with the `evalai-cli`, making the platform more accessible from your terminal.
*   **Scalability and Portability:** Built with open-source technologies like Docker, Django, Node.js, and PostgreSQL for easy deployment and scalability.
*   **Optimized Performance:** Achieve faster evaluation times through worker node warm-up, dataset chunking, and multi-core processing.

## Goal

The primary objective is to build a centralized platform to host, participate, and collaborate on AI challenges worldwide, facilitating progress in AI benchmarking.

## Installation

Get started with EvalAI quickly using Docker:

1.  **Install Dependencies:** Ensure you have [Docker](https://docs.docker.com/install/) and [Docker Compose](https://docs.docker.com/compose/install/) installed on your machine.
2.  **Clone the Repository:**
    ```shell
    git clone https://github.com/Cloud-CV/EvalAI.git evalai && cd evalai
    ```
3.  **Build and Run:**
    ```
    docker-compose up --build
    ```
    *To start optional services, use profiles like* `docker-compose --profile worker up --build` *or* `docker-compose --profile statsd up --build`

4.  **Access EvalAI:** Open your web browser and go to `http://127.0.0.1:8888`.

    *   **Default Users:**
        *   **SUPERUSER:** username: `admin`, password: `password`
        *   **HOST USER:** username: `host`, password: `password`
        *   **PARTICIPANT USER:** username: `participant`, password: `password`

    *   Refer to the [common errors during installation](https://evalai.readthedocs.io/en/latest/faq(developers).html#common-errors-during-installation) if any issues arise.

## Documentation Setup

Refer to `docs/README.md` within the repository for specific instructions on setting up the EvalAI documentation builder locally if you are looking to contribute.

## Citing EvalAI

If you use EvalAI for your research, please cite the following:

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

A non-exhaustive list of past contributors includes: [Deshraj Yadav](http://deshraj.xyz/), [Ram Ramrakhya](https://ram81.github.io/), [Akash Jain](http://www.jainakash.in/), [Taranjeet Singh](https://taranjeet.cc/), [Shiv Baran Singh](https://github.com/spyshiv), [Harsh Agarwal](https://dexter1691.github.io/), [Prithvijit Chattopadhyay](https://prithv1.github.io/), and [Devi Parikh](https://www.cc.gatech.edu/~parikh/).

## Contribution Guidelines

We welcome contributions! Please review our [contribution guidelines](https://github.com/Cloud-CV/EvalAI/blob/master/.github/CONTRIBUTING.md) to get started.

## Contributors

[//]: contributor-faces
```
Include the contributor faces here.