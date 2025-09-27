html
<p align="center">
  <img width="65%" src="docs/source/_static/img/evalai_logo.png" alt="EvalAI Logo">
</p>

<!--  Removed the Shields, as they have the same content as the original -->

## EvalAI: The Open-Source Platform for AI Algorithm Evaluation

EvalAI is an open-source platform designed to streamline the evaluation and comparison of machine learning and artificial intelligence algorithms at scale. **Tired of inconsistent results and hard-to-reproduce research?**  EvalAI provides a centralized solution for hosting, participating in, and collaborating on AI challenges. Check out the original repo: [https://github.com/Cloud-CV/EvalAI](https://github.com/Cloud-CV/EvalAI)

### Key Features

*   **Customizable Evaluation Protocols:** Define evaluation phases, dataset splits, and metrics to match your specific needs. Supports any programming language.
*   **Remote Evaluation:** Leverage dedicated compute resources for resource-intensive challenges.
*   **Dockerized Environments:** Submit and evaluate code within Docker containers, ensuring consistent and reproducible results.
*   **Command-Line Interface (CLI):**  The `evalai-cli` streamlines interaction with the platform via your terminal.
*   **Scalable and Portable:** Built with open-source technologies like Docker, Django, Node.js, and PostgreSQL for scalability and ease of deployment.
*   **Optimized Evaluation:** Employing techniques like worker warm-up and dataset chunking for significantly faster evaluation times.

### Goal

Our mission is to build a central hub for AI challenges, fostering collaboration and accelerating progress in the field.

### Installation

Getting started with EvalAI is straightforward using Docker:

1.  **Install Dependencies:** Ensure you have <a href="https://docs.docker.com/install/" target="_blank">Docker</a> and <a href="https://docs.docker.com/compose/install/" target="_blank">Docker Compose</a> installed.
2.  **Get the Source Code:** Clone the repository:

    ```bash
    git clone https://github.com/Cloud-CV/EvalAI.git evalai && cd evalai
    ```

3.  **Build and Run:** Execute the following command. This may take a while.
    ```bash
    docker-compose up --build
    ```
    For worker services:
    ```bash
    docker-compose --profile worker up --build
    ```
    For statsd-exporter:
    ```bash
    docker-compose --profile statsd up --build
    ```
    For both optional services:
    ```bash
    docker-compose --profile worker --profile statsd up --build
    ```

4.  **Access EvalAI:** Open your web browser and go to <a href="http://127.0.0.1:8888" target="_blank">http://127.0.0.1:8888</a>.

    Default User Accounts:

    *   **SUPERUSER:**  username: `admin`, password: `password`
    *   **HOST USER:**  username: `host`, password: `password`
    *   **PARTICIPANT USER:** username: `participant`, password: `password`

    For troubleshooting, see the  <a href="https://evalai.readthedocs.io/en/latest/faq(developers).html#common-errors-during-installation" target="_blank">common errors during installation</a> page.

### Documentation

For contributing to the documentation, follow the instructions in `docs/README.md`.

### Citing EvalAI

If using EvalAI for hosting challenges, please cite the following:

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

### Team

EvalAI is maintained by <a href="https://rishabhjain.xyz/" target="_blank">Rishabh Jain</a>, <a href="https://gchhablani.github.io/" target="_blank">Gunjan Chhablani</a>, and <a href="https://www.cc.gatech.edu/~dbatra/" target="_blank">Dhruv Batra</a>.

### Contributors

(List of contributors)
```
```
(List of contributors - kept in original form for compatibility with the repo)
```
```

### Contribution Guidelines

We welcome contributions!  See our <a href="https://github.com/Cloud-CV/EvalAI/blob/master/.github/CONTRIBUTING.md" target="_blank">contribution guidelines</a> for details.
```