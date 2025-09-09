<p align="center">
  <img width="65%" src="docs/source/_static/img/evalai_logo.png" alt="EvalAI Logo"/>
</p>

---

# EvalAI: The Open-Source Platform for AI/ML Algorithm Evaluation

**EvalAI is the leading open-source platform designed to help researchers, developers, and challenge organizers to evaluate and compare machine learning (ML) and artificial intelligence (AI) algorithms at scale.**  [Explore the EvalAI repository](https://github.com/Cloud-CV/EvalAI)

## Key Features

*   **Customizable Evaluation:** Define your own evaluation protocols with multiple phases, dataset splits, and support for any programming language.
*   **Remote Evaluation:** Leverage the power of remote clusters for compute-intensive challenges.
*   **Dockerized Evaluation:** Submit and evaluate your code within isolated Docker containers, ensuring reproducibility.
*   **Command-Line Interface (CLI):**  Interact with EvalAI directly from your terminal using the `evalai-cli` tool.
*   **Scalability and Portability:** Built with open-source technologies like Docker, Django, and PostgreSQL for flexible deployment.
*   **Optimized Performance:** Experience faster evaluation times through techniques like worker node warm-up and dataset chunking.

## Goal

EvalAI aims to be the central platform for hosting, participating in, and collaborating on AI challenges worldwide, accelerating progress in the field.

## Installation

Get up and running with EvalAI using Docker:

1.  **Install Docker and Docker Compose:** Follow the instructions for your operating system ([Docker Install](https://docs.docker.com/install/), [Docker Compose Install](https://docs.docker.com/compose/install/)).
2.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Cloud-CV/EvalAI.git evalai && cd evalai
    ```
3.  **Build and Run:**

    ```bash
    docker-compose up --build
    ```

    *   For worker services: `docker-compose --profile worker up --build`
    *   For statsd-exporter: `docker-compose --profile statsd up --build`
    *   For both optional services: `docker-compose --profile worker --profile statsd up --build`
4.  **Access EvalAI:** Open your web browser and go to <http://127.0.0.1:8888>.

    *   **Superuser:** `admin` / `password`
    *   **Host User:** `host` / `password`
    *   **Participant User:** `participant` / `password`

    If you encounter any issues, consult the [common errors documentation](https://evalai.readthedocs.io/en/latest/faq(developers).html#common-errors-during-installation).

## Documentation

*   For EvalAI Documentation setup, refer to the docs specific setup instructions in `docs/README.md`.

## Citing EvalAI

If you use EvalAI for hosting challenges, please cite this technical report:

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

### Past Contributors

Deshraj Yadav, Ram Ramrakhya, Akash Jain, Taranjeet Singh, Shiv Baran Singh, Harsh Agarwal, Prithvijit Chattopadhyay, and Devi Parikh.

## Contribution Guidelines

Interested in contributing?  Review the [contribution guidelines](https://github.com/Cloud-CV/EvalAI/blob/master/.github/CONTRIBUTING.md).

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->