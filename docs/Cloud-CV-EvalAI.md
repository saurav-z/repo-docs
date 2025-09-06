<p align="center"><img width="65%" src="docs/source/_static/img/evalai_logo.png"/></p>

# EvalAI: The Open-Source Platform for AI Challenge Hosting and Evaluation

**EvalAI is your go-to platform for hosting and participating in machine learning and AI challenges, fostering collaboration and accelerating progress in the field.**  This open-source platform provides a centralized hub for evaluating, comparing, and benchmarking AI algorithms at scale.  Explore the [EvalAI GitHub Repository](https://github.com/Cloud-CV/EvalAI) for more details.

## Key Features

*   **Customizable Evaluation:** Define evaluation phases, dataset splits, and metrics to match your specific challenge requirements.  Supports any programming language.
*   **Remote Evaluation:** Leverage cloud resources for compute-intensive challenges, enabling evaluation using custom clusters.
*   **Dockerized Evaluation:** Submit code as Docker images for secure and reproducible evaluations within controlled environments.
*   **CLI Support:** Interact with EvalAI seamlessly from your terminal using the `evalai-cli` tool.
*   **Scalability & Portability:** Built with open-source technologies like Docker, Django, Node.js, and PostgreSQL for easy deployment and scalability.
*   **Faster Evaluation:** Optimized for speed using techniques like worker node warm-up and dataset chunking, reducing evaluation time significantly.

## Goals

EvalAI aims to be a leading platform for AI challenge hosting, participation, and collaboration, accelerating advancements in the field by providing a robust and reliable evaluation framework.

## Installation

Get started with EvalAI quickly using Docker:

1.  **Install Prerequisites:** Install [Docker](https://docs.docker.com/install/) and [Docker Compose](https://docs.docker.com/compose/install/) on your machine.
2.  **Get the Code:** Clone the repository:

    ```bash
    git clone https://github.com/Cloud-CV/EvalAI.git evalai && cd evalai
    ```
3.  **Build and Run:** Build and start the necessary Docker containers:

    ```bash
    docker-compose up --build
    ```
    *   To include worker services: `docker-compose --profile worker up --build`
    *   To include statsd-exporter: `docker-compose --profile statsd up --build`
    *   To include both optional services: `docker-compose --profile worker --profile statsd up --build`

4.  **Access EvalAI:** Open your web browser and navigate to [http://127.0.0.1:8888](http://127.0.0.1:8888).

    *   **Default User Credentials:**
        *   SUPERUSER: username: `admin`, password: `password`
        *   HOST USER: username: `host`, password: `password`
        *   PARTICIPANT USER: username: `participant`, password: `password`

If you encounter issues, consult the [common errors during installation](https://evalai.readthedocs.io/en/latest/faq(developers).html#common-errors-during-installation) page.

## Documentation

Refer to the `docs/README.md` for specific instructions on setting up the documentation locally if you want to contribute.

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

## Contribution Guidelines

Interested in contributing? Review our [contribution guidelines](https://github.com/Cloud-CV/EvalAI/blob/master/.github/CONTRIBUTING.md).

## Contributors

[//]: contributor-faces
<!--  The contributor faces section has been preserved. -->