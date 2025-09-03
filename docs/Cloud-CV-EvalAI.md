<p align="center">
  <img width="65%" src="docs/source/_static/img/evalai_logo.png" alt="EvalAI Logo"/>
</p>

# EvalAI: The Open-Source Platform for AI and Machine Learning Algorithm Evaluation

**EvalAI** is a powerful, open-source platform designed to streamline the evaluation and comparison of machine learning (ML) and artificial intelligence (AI) algorithms.  Explore the code and contribute on [GitHub](https://github.com/Cloud-CV/EvalAI).

## Key Features

*   **Customizable Evaluation:** Create an unlimited number of evaluation phases, dataset splits, and leaderboards.
*   **Flexible Language Support:** Compatible with any programming language.
*   **Remote Evaluation:** Leverage dedicated compute resources for large-scale challenges.
*   **Dockerized Environments:** Submit code as Docker images for secure and reproducible evaluations.
*   **CLI Support:** Interact with EvalAI directly from your terminal using the `evalai-cli` tool.
*   **Scalable Architecture:** Built with open-source technologies like Docker, Django, and PostgreSQL for portability and scalability.
*   **Optimized Evaluation:** Achieve faster evaluation times through worker node warm-up and dataset chunking.

## Why Use EvalAI?

EvalAI addresses the challenges of comparing AI algorithms by providing a centralized platform for reproducible results and accurate quantitative analysis. It simplifies the process of organizing and participating in AI challenges, allowing researchers to focus on innovation.

## Getting Started: Installation

Install and run EvalAI with Docker using the following steps:

1.  **Install Docker and Docker Compose:** Ensure you have Docker and Docker Compose installed on your machine.
2.  **Get the Source Code:** Clone the EvalAI repository:

    ```bash
    git clone https://github.com/Cloud-CV/EvalAI.git evalai && cd evalai
    ```

3.  **Build and Run Containers:**

    ```bash
    docker-compose up --build
    ```

    *   To start **worker** services:  `docker-compose --profile worker up --build`
    *   To start **statsd-exporter**: `docker-compose --profile statsd up --build`
    *   To start **both optional services**: `docker-compose --profile worker --profile statsd up --build`

4.  **Access EvalAI:** Open your web browser and go to [http://127.0.0.1:8888](http://127.0.0.1:8888).
    *   Default user accounts:
        *   **SUPERUSER:** username: `admin`, password: `password`
        *   **HOST USER:** username: `host`, password: `password`
        *   **PARTICIPANT USER:** username: `participant`, password: `password`

If you encounter any installation issues, consult the [common errors during installation](https://evalai.readthedocs.io/en/latest/faq(developers).html#common-errors-during-installation) page.

## Documentation

Find detailed instructions and information in the `docs/README.md` file.

## Citing EvalAI

If you use EvalAI in your research, please cite the following technical report:

```bibtex
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

## Team & Contributors

EvalAI is a collaborative effort. It is maintained by [Rishabh Jain](https://rishabhjain.xyz/), [Gunjan Chhablani](https://gchhablani.github.io/), and [Dhruv Batra](https://www.cc.gatech.edu/~dbatra/).  A huge thank you to all the contributors!

[//]: contributor-faces
```