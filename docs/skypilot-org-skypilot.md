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

## SkyPilot: Run AI Workloads Faster, Cheaper, and Anywhere

SkyPilot is an open-source platform that simplifies and optimizes running AI and batch workloads across various infrastructure providers, including Kubernetes, AWS, GCP, and more. Visit the [SkyPilot GitHub repository](https://github.com/skypilot-org/skypilot) for the latest updates.

### Key Features

*   **Unified Infrastructure:** Run your AI tasks seamlessly on Kubernetes, 16+ clouds, and various hardware configurations (GPUs, TPUs, CPUs).
*   **Cost Optimization:** Leverage spot instances for cost savings (3-6x) with automatic preemption recovery and intelligent scheduling.  Autostop features ensure idle resources are cleaned up automatically.
*   **Simplified Deployment:**  Easily define your environment and jobs as code for portability and reproducibility.  The SkyPilot API simplifies deployment across different cloud providers.
*   **Kubernetes Made Easy for AI:** Experience Slurm-like ease of use, cloud-native robustness, and local development capabilities directly on your Kubernetes clusters.
*   **Accelerated AI Development:** Efficiently manage and queue multiple AI jobs, accelerating your development workflow.

### Latest Updates

*   [Aug 2025] Serve and finetune **OpenAI GPT-OSS models** (gpt-oss-120b, gpt-oss-20b) with one command on any infra: [**serve**](./llm/gpt-oss/) + [**LoRA and full finetuning**](./llm/gpt-oss-finetuning/)
*   [Jul 2025] Run distributed **RL training for LLMs** with Verl (PPO, GRPO) on any cloud: [**example**](./llm/verl/)
*   [Jul 2025] 🎉 SkyPilot v0.10.0 released! [**blog post**](https://blog.skypilot.co/announcing-skypilot-0.10.0/), [**release notes**](https://github.com/skypilot-org/skypilot/releases/tag/v0.10.0)
*   [Jul 2025] Finetune **Llama4** on any distributed cluster/cloud: [**example**](./llm/llama-4-finetuning/)
*   [Jul 2025] Two-part blog series, `The Evolution of AI Job Orchestration`: (1) [Running AI jobs on GPU Neoclouds](https://blog.skypilot.co/ai-job-orchestration-pt1-gpu-neoclouds/), (2) [The AI-Native Control Plane & Orchestration that Finally Works for ML](https://blog.skypilot.co/ai-job-orchestration-pt2-ai-control-plane/)
*   [Apr 2025] Spin up **Qwen3** on your cluster/cloud: [**example**](./llm/qwen/)
*   [Mar 2025] Run and serve **Google Gemma 3** using SkyPilot [**example**](./llm/gemma3/)
*   [Feb 2025] Prepare and serve **Retrieval Augmented Generation (RAG) with DeepSeek-R1**: [**blog post**](https://blog.skypilot.co/deepseek-rag), [**example**](./llm/rag/)
*   [Feb 2025] Run and serve **DeepSeek-R1 671B** using SkyPilot and SGLang with high throughput: [**example**](./llm/deepseek-r1/)
*   [Feb 2025] Prepare and serve large-scale image search with **vector databases**: [**blog post**](https://blog.skypilot.co/large-scale-vector-database/), [**example**](./examples/vector_database/)
*   [Jan 2025] Launch and serve distilled models from **[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)** and **[Janus](https://github.com/deepseek-ai/DeepSeek-Janus)** on Kubernetes or any cloud: [**R1 example**](./llm/deepseek-r1-distilled/) and [**Janus example**](./llm/deepseek-janus/)
*   [Oct 2024] :tada: **SkyPilot crossed 1M+ downloads** :tada:: Thank you to our community! [**Twitter/X**](https://x.com/skypilot_org/status/1844770841718067638)

**LLM Finetuning Cookbooks**: Finetuning Llama 2 / Llama 3.1 in your own cloud environment, privately: Llama 2 [**example**](./llm/vicuna-llama-2/) and [**blog**](https://blog.skypilot.co/finetuning-llama2-operational-guide/); Llama 3.1 [**example**](./llm/llama-3_1-finetuning/) and [**blog**](https://blog.skypilot.co/finetune-llama-3_1-on-your-infra/)

### SkyPilot in 1 Minute

SkyPilot simplifies AI workload execution with a unified interface.  A SkyPilot task is defined by: resource requirements, data to be synced, setup commands, and task commands. Once written in this [**unified interface**](https://docs.skypilot.co/en/latest/reference/yaml-spec.html) (YAML or Python API), the task can be launched on any available infrastructure (Kubernetes, cloud, etc.).

Here's how to get started:

1.  Create a file named `my_task.yaml` and paste the following:

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

2.  Prepare the working directory:

    ```bash
    git clone https://github.com/pytorch/examples.git ~/torch_examples
    ```

3.  Launch your task using the `sky launch` command:

    ```bash
    sky launch my_task.yaml
    ```

SkyPilot then handles the following:

1.  Find the most cost-effective and available infrastructure.
2.  Provision the necessary resources (GPUs or VMs).
3.  Sync your local working directory.
4.  Install dependencies.
5.  Execute your task commands and stream the logs.

See the [Quickstart Guide](https://docs.skypilot.co/en/latest/getting-started/quickstart.html) for detailed instructions.

### Runnable Examples

Explore a wide range of SkyPilot examples covering training, serving, LLM models, AI applications, and popular frameworks in the [SkyPilot examples](https://docs.skypilot.co/en/docs-examples/examples/index.html)

**Featured Examples:**

| Task | Examples |
|----------|----------|
| Training | [Verl](https://docs.skypilot.co/en/latest/examples/training/verl.html), [Finetune Llama 4](https://docs.skypilot.co/en/latest/examples/training/llama-4-finetuning.html), [PyTorch](https://docs.skypilot.co/en/latest/getting-started/tutorial.html), [DeepSpeed](https://docs.skypilot.co/en/latest/examples/training/deepspeed.html), [NeMo](https://docs.skypilot.co/en/latest/examples/training/nemo.html), [Ray](https://docs.skypilot.co/en/latest/examples/training/ray.html), [Unsloth](https://docs.skypilot.co/en/latest/examples/training/unsloth.html), [Jax/TPU](https://docs.skypilot.co/en/latest/examples/training/tpu.html) |
| Serving | [vLLM](https://docs.skypilot.co/en/latest/examples/serving/vllm.html), [SGLang](https://docs.skypilot.co/en/latest/examples/serving/sglang.html), [Ollama](https://docs.skypilot.co/en/latest/examples/serving/ollama.html) |
| Models | [DeepSeek-R1](https://docs.skypilot.co/en/latest/examples/models/deepseek-r1.html), [Llama 3](https://docs.skypilot.co/en/latest/examples/models/llama-3.html), [CodeLlama](https://docs.skypilot.co/en/latest/examples/models/codellama.html), [Qwen](https://docs.skypilot.co/en/latest/examples/models/qwen.html), [Mixtral](https://docs.skilot.co/en/latest/examples/models/mixtral.html) |
| AI apps | [RAG](https://docs.skypilot.co/en/latest/examples/applications/rag.html), [vector databases](https://docs.skypilot.co/en/latest/examples/applications/vector_database.html) (ChromaDB, CLIP) |
| Common frameworks | [Airflow](https://docs.skypilot.co/en/latest/examples/frameworks/airflow.html), [Jupyter](https://docs.skypilot.co/en/latest/examples/frameworks/jupyter.html) |

Source code can be found in the [`llm/`](https://github.com/skypilot-org/skypilot/tree/master/llm) and [`examples/`](https://github.com/skypilot-org/skypilot/tree/master/examples) directories.

### Getting Started

*   **[Documentation](https://docs.skypilot.co/)**: Comprehensive documentation covering installation, quickstart, and CLI reference.
*   [Installation](https://docs.skypilot.co/en/latest/getting-started/installation.html)
*   [Quickstart](https://docs.skypilot.co/en/latest/getting-started/quickstart.html)
*   [CLI reference](https://docs.skypilot.co/en/latest/reference/cli.html)

### Supported Infrastructure

SkyPilot currently supports a wide array of infrastructure providers, including: Kubernetes, AWS, GCP, Azure, OCI, Lambda Cloud, Fluidstack, RunPod, Cudo, Digital Ocean, Paperspace, Cloudflare, Samsung, IBM, Vast.ai, VMware vSphere, and Nebius.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-dark.png">
    <img alt="SkyPilot" src="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-light.png" width=85%>
  </picture>
</p>

### Additional Resources

*   **[SkyPilot Overview](https://docs.skypilot.co/en/latest/overview.html)**
*   **[SkyPilot Blog](https://blog.skypilot.co/)**: Stay updated with the latest news, tutorials, and community spotlights.
*   **[Community Spotlights](https://blog.skypilot.co/community/)**: Case studies and integrations.
*   **[Introductory Blog Post](https://blog.skypilot.co/introducing-skypilot/)**

### Connect with the Community

*   [Slack](http://slack.skypilot.co)
*   [X / Twitter](https://twitter.com/skypilot_org)
*   [LinkedIn](https://www.linkedin.com/company/skypilot-oss/)

### Research Papers

*   [SkyPilot paper](https://www.usenix.org/system/files/nsdi23-yang-zongheng.pdf) (NSDI 2023)
*   [Sky Computing whitepaper](https://arxiv.org/abs/2205.07147)
*   [Sky Computing vision paper](https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s02-stoica.pdf) (HotOS 2021)
*   [SkyServe: AI serving across regions and clouds](https://arxiv.org/pdf/2411.01438) (EuroSys 2025)
*   [Managed jobs spot instance policy](https://www.usenix.org/conference/nsdi24/presentation/wu-zhanghao) (NSDI 2024)

### Contributing

We welcome contributions!  Please review the [CONTRIBUTING](CONTRIBUTING.md) guidelines to get involved.

### Questions and Feedback

*   For issues and feature requests, please [open a GitHub issue](https://github.com/skypilot-org/skypilot/issues/new).
*   For questions, please use [GitHub Discussions](https://github.com/skypilot-org/skypilot/discussions).

For general discussions, join us on the [SkyPilot Slack](http://slack.skypilot.co).