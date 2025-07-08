<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/skypilot-wide-dark-1k.png">
    <img alt="SkyPilot" src="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/skypilot-wide-light-1k.png" width=55%>
  </picture>
</p>

<p align="center">
  <a href="https://docs.skypilot.co/">
    <img alt="Documentation" src="https://img.shields.io/badge/docs-gray?logo=readthedocs&logoColor=f5f5f5">
  </a>

  <a href="https://github.com/skypilot-org/skypilot/releases">
    <img alt="GitHub Release" src="https://img.shields.io/github/release/skypilot-org/skypilot.svg">
  </a>

  <a href="http://slack.skypilot.co">
    <img alt="Join Slack" src="https://img.shields.io/badge/SkyPilot-Join%20Slack-blue?logo=slack">
  </a>

  <a href="https://github.com/skypilot-org/skypilot/releases">
    <img alt="Downloads" src="https://img.shields.io/pypi/dm/skypilot">
  </a>
</p>

## SkyPilot: Run Your AI & Batch Workloads on Any Infrastructure – Faster, Cheaper, Easier

SkyPilot is an open-source framework that simplifies running AI and batch workloads across various infrastructure providers.  

[**View the SkyPilot Repository on GitHub**](https://github.com/skypilot-org/skypilot)

### Key Features

*   **Unified Infrastructure:** Run workloads on Kubernetes, AWS, GCP, Azure, and more with a single interface.
*   **Cost Optimization:** Leverage spot instances and autostop for significant cloud cost savings.
*   **Simplified Job Management:** Easily queue, run, and auto-recover jobs with environment and job as code.
*   **Accelerated Development:** Quickly spin up compute resources and streamline your AI workflows.
*   **Flexible Resource Provisioning:**  Access GPUs, TPUs, and CPUs with auto-retry and intelligent scheduling to find the best prices and availability.
*   **Team Collaboration**: Facilitate team deployments and resource sharing for more efficient utilization.

### What's New

*   [Apr 2025] Spin up **Qwen3** on your cluster/cloud: [**example**](./llm/qwen/)
*   [Mar 2025] Run and serve **Google Gemma 3** using SkyPilot [**example**](./llm/gemma3/)
*   [Feb 2025] Prepare and serve **Retrieval Augmented Generation (RAG) with DeepSeek-R1**: [**blog post**](https://blog.skypilot.co/deepseek-rag), [**example**](./llm/rag/)
*   [Feb 2025] Run and serve **DeepSeek-R1 671B** using SkyPilot and SGLang with high throughput: [**example**](./llm/deepseek-r1/)
*   [Feb 2025] Prepare and serve large-scale image search with **vector databases**: [**blog post**](https://blog.skypilot.co/large-scale-vector-database/), [**example**](./examples/vector_database/)
*   [Jan 2025] Launch and serve distilled models from **[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)** and **[Janus](https://github.com/deepseek-ai/DeepSeek-Janus)** on Kubernetes or any cloud: [**R1 example**](./llm/deepseek-r1-distilled/) and [**Janus example**](./llm/deepseek-janus/)
*   [Oct 2024] :tada: **SkyPilot crossed 1M+ downloads** :tada:: Thank you to our community! [**Twitter/X**](https://x.com/skypilot_org/status/1844770841718067638)
*   [Sep 2024] Point, launch and serve **Llama 3.2** on Kubernetes or any cloud: [**example**](./llm/llama-3_2/)


**LLM Finetuning Cookbooks**: Finetuning Llama 2 / Llama 3.1 in your own cloud environment, privately: Llama 2 [**example**](./llm/vicuna-llama-2/) and [**blog**](https://blog.skypilot.co/finetuning-llama2-operational-guide/); Llama 3.1 [**example**](./llm/llama-3_1-finetuning/) and [**blog**](https://blog.skypilot.co/finetune-llama-3_1-on-your-infra/)

### Get Started

1.  **Installation:** Install SkyPilot using pip:
    ```bash
    pip install -U "skypilot[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
    ```
    For the latest features and fixes, use the nightly build or [install from source](https://docs.skypilot.co/en/latest/getting-started/installation.html):
    ```bash
    pip install "skypilot-nightly[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
    ```

2.  **Documentation:** Explore the official documentation for detailed guides and examples:

    *   [Getting Started](https://docs.skypilot.co/en/latest/getting-started/quickstart.html)
    *   [Installation](https://docs.skypilot.co/en/latest/getting-started/installation.html)
    *   [CLI Reference](https://docs.skypilot.co/en/latest/reference/cli.html)

### SkyPilot in 1 Minute:  A Quickstart

SkyPilot simplifies launching AI workloads with a simple YAML-based task definition.

1.  **Define Your Task:**  Create a YAML file (e.g., `my_task.yaml`) specifying resources, commands, and data.

    ```yaml
    resources:
      accelerators: A100:8  # 8x NVIDIA A100 GPU

    num_nodes: 1  # Number of VMs to launch

    # Working directory (optional) containing the project codebase.
    # Its contents are synced to ~/sky_workdir/ on the cluster.
    workdir: ~/torch_examples

    # Commands to be run before executing the job.
    # Typical use: pip install -r requirements.txt, git clone, etc.
    setup: |
      cd mnist
      pip install -r requirements.txt

    # Commands to run as a job.
    # Typical use: launch the main program.
    run: |
      cd mnist
      python main.py --epochs 1
    ```

2.  **Prepare Your Workdir:**  Clone your project code (or sync local files).

    ```bash
    git clone https://github.com/pytorch/examples.git ~/torch_examples
    ```

3.  **Launch Your Task:**  Use the `sky launch` command to deploy your task.

    ```bash
    sky launch my_task.yaml
    ```

    SkyPilot handles the rest: finding the best infrastructure, provisioning resources, syncing your code, and running your task.

### Runnable Examples

SkyPilot offers a variety of examples to get you started with different AI tasks.

*   **Training:** PyTorch, DeepSpeed, Finetune Llama 3, NeMo, Ray, Unsloth, Jax/TPU
*   **Serving:** vLLM, SGLang, Ollama
*   **Models:** DeepSeek-R1, Llama 3, CodeLlama, Qwen, Mixtral
*   **AI Apps:** RAG, vector databases (ChromaDB, CLIP)
*   **Common Frameworks:** Airflow, Jupyter

Browse the [SkyPilot Examples](https://docs.skypilot.co/en/docs-examples/examples/index.html) for details.  Source files and additional examples are in the [`llm/`](https://github.com/skypilot-org/skypilot/tree/master/llm) and [`examples/`](https://github.com/skypilot-org/skypilot/tree/master/examples) directories.

### Supported Infrastructure

SkyPilot supports a wide range of infrastructure providers:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-dark.png">
    <img alt="SkyPilot" src="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-light.png" width=85%>
  </picture>
</p>

Kubernetes, AWS, GCP, Azure, OCI, Lambda Cloud, Fluidstack, RunPod, Cudo, Digital Ocean, Paperspace, Cloudflare, Samsung, IBM, Vast.ai, VMware vSphere, Nebius.

### Learn More

*   [SkyPilot Overview](https://docs.skypilot.co/en/latest/overview.html)
*   [SkyPilot Docs](https://docs.skypilot.co/en/latest/)
*   [SkyPilot Blog](https://blog.skypilot.co/)

### Case Studies and Community

*   [Community Spotlights](https://blog.skypilot.co/community/)

### Follow SkyPilot

*   [Slack](http://slack.skypilot.co)
*   [X / Twitter](https://twitter.com/skypilot_org)
*   [LinkedIn](https://www.linkedin.com/company/skypilot-oss/)
*   [SkyPilot Blog](https://blog.skypilot.co/) ([Introductory blog post](https://blog.skypilot.co/introducing-skypilot/))

### Research

*   [SkyPilot paper](https://www.usenix.org/system/files/nsdi23-yang-zongheng.pdf) and [talk](https://www.usenix.org/conference/nsdi23/presentation/yang-zongheng) (NSDI 2023)
*   [Sky Computing whitepaper](https://arxiv.org/abs/2205.07147)
*   [Sky Computing vision paper](https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s02-stoica.pdf) (HotOS 2021)
*   [SkyServe: AI serving across regions and clouds](https://arxiv.org/pdf/2411.01438) (EuroSys 2025)
*   [Managed jobs spot instance policy](https://www.usenix.org/conference/nsdi24/presentation/wu-zhanghao)  (NSDI 2024)

### Origin

SkyPilot was initially started at the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley.

### Get Involved

*   **Issues & Feature Requests:** [Open a GitHub issue](https://github.com/skypilot-org/skypilot/issues/new).
*   **Questions:** Use [GitHub Discussions](https://github.com/skypilot-org/skypilot/discussions).
*   **General Discussion:** Join the [SkyPilot Slack](http://slack.skypilot.co).
*   **Contributing:**  See [CONTRIBUTING](CONTRIBUTING.md) for guidelines.
```
Key improvements and SEO considerations:

*   **Clear Title & Hook:** Uses a strong title and a one-sentence hook to immediately grab attention.
*   **SEO-Friendly Headings:**  Uses H2 and H3 headings for better structure and keyword targeting.
*   **Keyword Optimization:** Naturally incorporates relevant keywords like "AI," "batch workloads," "infrastructure," "cloud," and specific provider names.
*   **Bulleted Lists:**  Highlights key features in a clear, easy-to-scan format, improving readability and SEO.
*   **Concise Language:** Streamlines the text to convey information quickly and effectively.
*   **Call to Action:** Includes clear calls to action, such as "View the SkyPilot Repository on GitHub".
*   **Internal Linking:**  Provides links to the documentation, examples, and GitHub repository, encouraging exploration and engagement.
*   **Updated Information:** Kept the "What's New" section.
*   **Complete & Comprehensive:** Includes all the original information, but in a more organized and accessible way.