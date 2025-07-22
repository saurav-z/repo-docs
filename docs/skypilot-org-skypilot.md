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

## SkyPilot: Run AI on Any Infrastructure – Unified, Faster, and Cheaper

SkyPilot empowers you to run your AI workloads seamlessly across any infrastructure, optimizing cost and accelerating deployment. Check out the [original repo](https://github.com/skypilot-org/skypilot).

**Key Features:**

*   **Unified Infrastructure:** Deploy AI and batch workloads across 16+ clouds, Kubernetes, and on-premise resources with a single interface.
*   **Simplified Deployment:** Define your environment and jobs as code for easy portability and reproducibility.
*   **Cost Optimization:** Leverage spot instances for up to 6x cost savings and automatic resource cleanup to minimize idle resource costs.
*   **Intelligent Scheduling:** SkyPilot automatically runs your workloads on the most cost-effective and available infrastructure.
*   **Flexible Resource Provisioning:** Supports GPUs, TPUs, and CPUs with auto-retry for robust job execution.

---

:fire: **What's New:** :fire:

*   **[Apr 2025]** Spin up **Qwen3** on your cluster/cloud: [**example**](./llm/qwen/)
*   **[Mar 2025]** Run and serve **Google Gemma 3** using SkyPilot [**example**](./llm/gemma3/)
*   **[Feb 2025]** Prepare and serve **Retrieval Augmented Generation (RAG) with DeepSeek-R1**: [**blog post**](https://blog.skypilot.co/deepseek-rag), [**example**](./llm/rag/)
*   **[Feb 2025]** Run and serve **DeepSeek-R1 671B** using SkyPilot and SGLang with high throughput: [**example**](./llm/deepseek-r1/)
*   **[Feb 2025]** Prepare and serve large-scale image search with **vector databases**: [**blog post**](https://blog.skypilot.co/large-scale-vector-database/), [**example**](./examples/vector_database/)
*   **[Jan 2025]** Launch and serve distilled models from **[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)** and **[Janus](https://github.com/deepseek-ai/DeepSeek-Janus)** on Kubernetes or any cloud: [**R1 example**](./llm/deepseek-r1-distilled/) and [**Janus example**](./llm/deepseek-janus/)
*   **[Oct 2024]** :tada: **SkyPilot crossed 1M+ downloads** :tada:: Thank you to our community! [**Twitter/X**](https://x.com/skypilot_org/status/1844770841718067638)
*   **[Sep 2024]** Point, launch and serve **Llama 3.2** on Kubernetes or any cloud: [**example**](./llm/llama-3_2/)

**LLM Finetuning Cookbooks**: Finetuning Llama 2 / Llama 3.1 in your own cloud environment, privately: Llama 2 [**example**](./llm/vicuna-llama-2/) and [**blog**](https://blog.skypilot.co/finetuning-llama2-operational-guide/); Llama 3.1 [**example**](./llm/llama-3_1-finetuning/) and [**blog**](https://blog.skypilot.co/finetune-llama-3_1-on-your-infra/)

---

### Getting Started

Find comprehensive documentation and examples to help you get started:

*   [Installation](https://docs.skypilot.co/en/latest/getting-started/installation.html)
*   [Quickstart](https://docs.skypilot.co/en/latest/getting-started/quickstart.html)
*   [CLI Reference](https://docs.skypilot.co/en/latest/reference/cli.html)

### SkyPilot in 1 Minute

SkyPilot tasks are defined with resource requirements, data to be synced, setup and run commands within a unified YAML or Python API, enabling seamless deployment on any cloud provider.

Here's a quick example:

1.  **Create `my_task.yaml`:**

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

2.  **Prepare the working directory:**

    ```bash
    git clone https://github.com/pytorch/examples.git ~/torch_examples
    ```

3.  **Launch the task:**

    ```bash
    sky launch my_task.yaml
    ```

SkyPilot automatically handles the underlying infrastructure:

*   Finds the most cost-effective VM instance.
*   Provisions the VM and handles auto-failover.
*   Syncs your working directory.
*   Runs setup and run commands.

### Runnable Examples

Explore a wide range of ready-to-use examples covering various use cases:

*   **Training:** PyTorch, DeepSpeed, Finetune Llama 3, NeMo, Ray, Unsloth, Jax/TPU.
*   **Serving:** vLLM, SGLang, Ollama.
*   **Models:** DeepSeek-R1, Llama 3, CodeLlama, Qwen, Mixtral.
*   **AI Apps:** RAG, Vector databases (ChromaDB, CLIP).
*   **Common Frameworks:** Airflow, Jupyter.

Find all examples in [`llm/`](https://github.com/skypilot-org/skypilot/tree/master/llm) and [`examples/`](https://github.com/skypilot-org/skypilot/tree/master/examples).

### Supported Infrastructure

SkyPilot supports a broad range of infrastructure providers: Kubernetes, AWS, GCP, Azure, OCI, Lambda Cloud, Fluidstack, RunPod, Cudo, Digital Ocean, Paperspace, Cloudflare, Samsung, IBM, Vast.ai, VMware vSphere, and Nebius.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-dark.png">
    <img alt="SkyPilot" src="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-light.png" width=85%>
  </picture>
</p>

### Installation

Install SkyPilot using `pip`:

```bash
# Choose your clouds:
pip install -U "skypilot[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```

For the latest features or to install from source:

```bash
# Choose your clouds:
pip install "skypilot-nightly[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```

### Learn More

*   [SkyPilot Overview](https://docs.skypilot.co/en/latest/overview.html)
*   [SkyPilot Documentation](https://docs.skypilot.co/en/latest/)
*   [SkyPilot Blog](https://blog.skypilot.co/)
*   [Community Spotlights](https://blog.skypilot.co/community/)

### Stay Updated

*   [Slack](http://slack.skypilot.co)
*   [X / Twitter](https://twitter.com/skypilot_org)
*   [LinkedIn](https://www.linkedin.com/company/skypilot-oss/)
*   [SkyPilot Blog](https://blog.skypilot.co/)
*   [Introductory blog post](https://blog.skypilot.co/introducing-skypilot/)

### Research

*   [SkyPilot paper](https://www.usenix.org/system/files/nsdi23-yang-zongheng.pdf) and [talk](https://www.usenix.org/conference/nsdi23/presentation/yang-zongheng) (NSDI 2023)
*   [Sky Computing whitepaper](https://arxiv.org/abs/2205.07147)
*   [Sky Computing vision paper](https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s02-stoica.pdf) (HotOS 2021)
*   [SkyServe: AI serving across regions and clouds](https://arxiv.org/pdf/2411.01438) (EuroSys 2025)
*   [Managed jobs spot instance policy](https://www.usenix.org/conference/nsdi24/presentation/wu-zhanghao)  (NSDI 2024)

### Origin

SkyPilot was initially started at the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley. Learn more about Sky Computing [here](https://docs.skypilot.co/en/latest/sky-computing.html).

### Get Involved

*   For issues and feature requests, [open a GitHub issue](https://github.com/skypilot-org/skypilot/issues/new).
*   For questions, use [GitHub Discussions](https://github.com/skypilot-org/skypilot/discussions).
*   For general discussions, join the [SkyPilot Slack](http://slack.skypilot.co).

### Contributing

We welcome contributions! See [CONTRIBUTING](CONTRIBUTING.md) for details.