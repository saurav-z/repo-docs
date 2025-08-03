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

## SkyPilot: Run Your AI Workloads Anywhere, Easily and Affordably

SkyPilot is an open-source framework that empowers you to run AI and batch workloads across any infrastructure, optimizing for speed, cost, and ease of use.  [**Get started with SkyPilot on GitHub**](https://github.com/skypilot-org/skypilot).

### Key Features

*   **Unified Infrastructure Management:** Run AI workloads on Kubernetes, AWS, GCP, Azure, and more, all from a single interface.
*   **Cost Optimization:** Leverage spot instances for up to 6x cost savings, with automatic preemption recovery and intelligent scheduling.
*   **Simplified Deployment:**  Define your jobs and environments as code, ensuring portability and easy management.
*   **Automated Resource Provisioning:** SkyPilot handles VM provisioning, auto-retry, and data syncing, saving you valuable time.
*   **Scalable & Flexible:**  Supports diverse hardware including GPUs, TPUs, and CPUs, enabling you to scale your AI projects.
*   **Team Collaboration:** Easily manage resources and share them within your team.

### What's New?

*   **[July 2025]** Run distributed **RL training for LLMs** with Verl (PPO, GRPO) on any cloud: [**example**](./llm/verl/)
*   **[July 2025]** 🎉 SkyPilot v0.10.0 released! [**blog post**](https://blog.skypilot.co/announcing-skypilot-0.10.0/), [**release notes**](https://github.com/skypilot-org/skypilot/releases/tag/v0.10.0)
*   **[July 2025]** Finetune **Llama4** on any distributed cluster/cloud: [**example**](./llm/llama-4-finetuning/)
*   **[July 2025]** Two-part blog series, `The Evolution of AI Job Orchestration`: (1) [Running AI jobs on GPU Neoclouds](https://blog.skypilot.co/ai-job-orchestration-pt1-gpu-neoclouds/), (2) [The AI-Native Control Plane & Orchestration that Finally Works for ML](https://blog.skypilot.co/ai-job-orchestration-pt2-ai-control-plane/)
*   **[Apr 2025]** Spin up **Qwen3** on your cluster/cloud: [**example**](./llm/qwen/)
*   **[Mar 2025]** Run and serve **Google Gemma 3** using SkyPilot [**example**](./llm/gemma3/)
*   **[Feb 2025]** Prepare and serve **Retrieval Augmented Generation (RAG) with DeepSeek-R1**: [**blog post**](https://blog.skypilot.co/deepseek-rag), [**example**](./llm/rag/)
*   **[Feb 2025]** Run and serve **DeepSeek-R1 671B** using SkyPilot and SGLang with high throughput: [**example**](./llm/deepseek-r1/)
*   **[Feb 2025]** Prepare and serve large-scale image search with **vector databases**: [**blog post**](https://blog.skypilot.co/large-scale-vector-database/), [**example**](./examples/vector_database/)
*   **[Jan 2025]** Launch and serve distilled models from **[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)** and **[Janus](https://github.com/deepseek-ai/DeepSeek-Janus)** on Kubernetes or any cloud: [**R1 example**](./llm/deepseek-r1-distilled/) and [**Janus example**](./llm/deepseek-janus/)
*   **[Oct 2024]** :tada: **SkyPilot crossed 1M+ downloads** :tada:: Thank you to our community! [**Twitter/X**](https://x.com/skypilot_org/status/1844770841718067638)

### LLM Finetuning Cookbooks

Finetuning Llama 2 / Llama 3.1 in your own cloud environment, privately: Llama 2 [**example**](./llm/vicuna-llama-2/) and [**blog**](https://blog.skypilot.co/finetuning-llama2-operational-guide/); Llama 3.1 [**example**](./llm/llama-3_1-finetuning/) and [**blog**](https://blog.skypilot.co/finetune-llama-3_1-on-your-infra/)

### Get Started: Installation

Install SkyPilot using pip:

```bash
pip install -U "skypilot[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```

For the latest features, use the nightly build or install from source:

```bash
pip install "skypilot-nightly[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```

### Example:  Running a PyTorch Training Job

Here's a quick example demonstrating how to launch a training job with SkyPilot:

1.  **Create a YAML file (`my_task.yaml`)**:

    ```yaml
    resources:
      accelerators: A100:8  # 8x NVIDIA A100 GPU
    num_nodes: 1  # Number of VMs to launch
    workdir: ~/torch_examples
    setup: |
      cd mnist
      pip install -r requirements.txt
    run: |
      cd mnist
      python main.py --epochs 1
    ```

2.  **Prepare the workdir**:

    ```bash
    git clone https://github.com/pytorch/examples.git ~/torch_examples
    ```

3.  **Launch the task**:

    ```bash
    sky launch my_task.yaml
    ```

###  SkyPilot in 1 Minute

*   Define a task specifying resources, data, setup commands, and the job commands.
*   Launch the task using the unified interface (YAML or Python API).
*   SkyPilot handles:
    *   Finding the most cost-effective VM instance.
    *   Provisioning the VM with auto-failover.
    *   Syncing your local workdir.
    *   Running setup and job commands.

See [Quickstart](https://docs.skypilot.co/en/latest/getting-started/quickstart.html) for a full guide.

### Runnable Examples

Explore a wide range of examples for training, serving, and AI app development:

*   **Training:** PyTorch, DeepSpeed, Finetune Llama 3, NeMo, Ray, Unsloth, Jax/TPU
*   **Serving:** vLLM, SGLang, Ollama
*   **Models:** DeepSeek-R1, Llama 3, CodeLlama, Qwen, Mixtral
*   **AI Apps:** RAG, vector databases (ChromaDB, CLIP)
*   **Frameworks:** Airflow, Jupyter

Find source files and more examples in [`llm/`](https://github.com/skypilot-org/skypilot/tree/master/llm) and [`examples/`](https://github.com/skypilot-org/skypilot/tree/master/examples).

### Supported Infrastructure

SkyPilot supports a wide range of infrastructure providers:  Kubernetes, AWS, GCP, Azure, OCI, Lambda Cloud, Fluidstack, RunPod, Cudo, Digital Ocean, Paperspace, Cloudflare, Samsung, IBM, Vast.ai, VMware vSphere, Nebius.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-dark.png">
    <img alt="SkyPilot" src="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-light.png" width=85%>
  </picture>
</p>

### Resources

*   **Documentation:** [SkyPilot Documentation](https://docs.skypilot.co/en/latest/)
*   **Getting Started:** [Installation](https://docs.skypilot.co/en/latest/getting-started/installation.html), [Quickstart](https://docs.skypilot.co/en/latest/getting-started/quickstart.html)
*   **CLI Reference:** [CLI Reference](https://docs.skypilot.co/en/latest/reference/cli.html)
*   **SkyPilot Overview:** [Overview](https://docs.skypilot.co/en/latest/overview.html)
*   **Sky Computing:** [Concept: Sky Computing](https://docs.skypilot.co/en/latest/sky-computing.html)
*   **Blog:** [SkyPilot Blog](https://blog.skypilot.co/) ([Introductory blog post](https://blog.skypilot.co/introducing-skypilot/))

###  Connect with the Community

*   **Slack:** [Join SkyPilot Slack](http://slack.skypilot.co)
*   **X / Twitter:** [SkyPilot on Twitter](https://twitter.com/skypilot_org)
*   **LinkedIn:** [SkyPilot on LinkedIn](https://www.linkedin.com/company/skypilot-oss/)

###  Research

*   **SkyPilot Paper:** [SkyPilot paper](https://www.usenix.org/system/files/nsdi23-yang-zongheng.pdf) and [talk](https://www.usenix.org/conference/nsdi23/presentation/yang-zongheng) (NSDI 2023)
*   **Sky Computing whitepaper:** [Sky Computing whitepaper](https://arxiv.org/abs/2205.07147)
*   **Sky Computing vision paper:** [Sky Computing vision paper](https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s02-stoica.pdf) (HotOS 2021)
*   **SkyServe: AI serving across regions and clouds:** [SkyServe: AI serving across regions and clouds](https://arxiv.org/pdf/2411.01438) (EuroSys 2025)
*   **Managed jobs spot instance policy:** [Managed jobs spot instance policy](https://www.usenix.org/conference/nsdi24/presentation/wu-zhanghao)  (NSDI 2024)

###  Contribute

We welcome contributions!  See [CONTRIBUTING](CONTRIBUTING.md) for guidelines.

###  Questions and Feedback

*   **Issues and Feature Requests:** [Open a GitHub issue](https://github.com/skypilot-org/skypilot/issues/new).
*   **Questions:** Use [GitHub Discussions](https://github.com/skypilot-org/skypilot/discussions).