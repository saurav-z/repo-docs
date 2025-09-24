<p align="center"><img width="65%" src="docs/source/_static/img/evalai_logo.png" alt="EvalAI Logo"/></p>

# EvalAI: The Open-Source Platform for AI Challenge Evaluation

**EvalAI is the leading open-source platform designed to streamline the evaluation and comparison of machine learning and AI algorithms, fostering collaboration and accelerating advancements in the field.**  Discover more at the [original repo](https://github.com/Cloud-CV/EvalAI).

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

*   **Customizable Evaluation Protocols:** Define evaluation phases, dataset splits, and metrics to fit your specific needs using any programming language. Public and private leaderboards allow for flexible result organization.
*   **Remote Evaluation:** Leverage external compute resources for resource-intensive challenges. Easily integrate your own cluster of worker nodes.
*   **Dockerized Environment Evaluation:**  Submit code as Docker images for secure and reproducible evaluations against test environments on the EvalAI server.
*   **Command-Line Interface (CLI) Support:**  The `evalai-cli` extends the platform's functionality to your command line for enhanced accessibility and terminal-based interactions.
*   **Scalability & Portability:** Built with open-source technologies like Docker, Django, Node.js, and PostgreSQL, ensuring scalability and easy deployment.
*   **Optimized Evaluation Speed:**  Improve evaluation times with pre-loading of challenge code and dataset, alongside the use of multiple cores.

##  Installation

Get started with EvalAI quickly using Docker:

1.  **Install Docker and Docker Compose:** Ensure Docker and Docker Compose are installed on your machine.
2.  **Get the Source Code:** Clone the EvalAI repository:
    ```bash
    git clone https://github.com/Cloud-CV/EvalAI.git evalai && cd evalai
    ```
3.  **Build and Run:** Build and start the Docker containers:
    ```bash
    docker-compose up --build
    ```
    *To include workers, stats, etc., follow the commands in the original documentation.*

4.  **Access EvalAI:** Open your web browser and go to [http://127.0.0.1:8888](http://127.0.0.1:8888).  Use the following default credentials:

    *   **SUPERUSER:** username: `admin`, password: `password`
    *   **HOST USER:** username: `host`, password: `password`
    *   **PARTICIPANT USER:** username: `participant`, password: `password`

*Refer to the FAQ for troubleshooting common installation issues.*

## Documentation Setup

If you wish to contribute to the EvalAI documentation, find detailed setup instructions in `docs/README.md`.

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
    <a href="http://learningsys.org/sosp19/assets/papers/23_CameraReadySubmission_EvalAI_SOSP_2019%20(8)%20(1).pdf" target="_blank"><img src="docs/source/_static/img/evalai-paper.jpg" alt="EvalAI Paper"/></a>
</p>

## Team

EvalAI is maintained by Rishabh Jain, Gunjan Chhablani, and Dhruv Batra.

*A non-exhaustive list of past contributors is provided below.*

## Contribution Guidelines

Contribute to EvalAI by following our [contribution guidelines](https://github.com/Cloud-CV/EvalAI/blob/master/.github/CONTRIBUTING.md).

## Contributors

[//]: contributor-faces
*   *List of contributors with GitHub profile images (as in original README)*