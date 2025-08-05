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

<h1 align="center">SkyPilot: Run Your AI Workloads Anywhere, Easier and Cheaper</h1>

<p align="center">
  SkyPilot simplifies running AI workloads on any infrastructure, from cloud providers to on-premise clusters.
</p>

<div align="center">

#### [🌟 **SkyPilot Demo** 🌟: Click to see a 1-minute tour](https://demo.skypilot.co/dashboard/)

</div>

---

## Key Features

*   **Unified Interface:** Run AI workloads seamlessly across multiple clouds and hardware platforms with a single, easy-to-use interface.
*   **Cost Optimization:** Automatically find the cheapest and most available infrastructure and supports spot instances for significant cost savings (3-6x).
*   **Automated Infrastructure Management:**  SkyPilot handles provisioning, setup, and auto-recovery for your AI jobs.
*   **Portability:** Avoid vendor lock-in and easily move your workloads between different providers.
*   **Scalability:** Run distributed training and batch workloads with ease.
*   **Easy-to-Use for AI Users:** Quickly spin up compute, environment and job as code, and simplified job management.
*   **Team Deployment and Resource Sharing:**  Supports team collaboration and resource sharing across various infrastructures.

---

:fire: *News* :fire:
- [Jul 2025] Run distributed **RL training for LLMs** with Verl (PPO, GRPO) on any cloud: [**example**](./llm/verl/)
- [Jul 2025] 🎉 SkyPilot v0.10.0 released! [**blog post**](https://blog.skypilot.co/announcing-skypilot-0.10.0/), [**release notes**](https://github.com/skypilot-org/skypilot/releases/tag/v0.10.0)
- [Jul 2025] Finetune **Llama4** on any distributed cluster/cloud: [**example**](./llm/llama-4-finetuning/)
- [Jul 2025] Two-part blog series, `The Evolution of AI Job Orchestration`: (1) [Running AI jobs on GPU Neoclouds](https://blog.skypilot.co/ai-job-orchestration-pt1-gpu-neoclouds/), (2) [The AI-Native Control Plane & Orchestration that Finally Works for ML](https://blog.skypilot.co/ai-job-orchestration-pt2-ai-control-plane/)
- [Apr 2025] Spin up **Qwen3** on your cluster/cloud: [**example**](./llm/qwen/)
- [Mar 2025] Run and serve **Google Gemma 3** using SkyPilot [**example**](./llm/gemma3/)
- [Feb 2025] Prepare and serve **Retrieval Augmented Generation (RAG) with DeepSeek-R1**: [**blog post**](https://blog.skypilot.co/deepseek-rag), [**example**](./llm/rag/)
- [Feb 2025] Run and serve **DeepSeek-R1 671B** using SkyPilot and SGLang with high throughput: [**example**](./llm/deepseek-r1/)
- [Feb 2025] Prepare and serve large-scale image search with **vector databases**: [**blog post**](https://blog.skypilot.co/large-scale-vector-database/), [**example**](./examples/vector_database/)
- [Jan 2025] Launch and serve distilled models from **[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)** and **[Janus](https://github.com/deepseek-ai/DeepSeek-Janus)** on Kubernetes or any cloud: [**R1 example**](./llm/deepseek-r1-distilled/) and [**Janus example**](./llm/deepseek-janus/)
- [Oct 2024] :tada: **SkyPilot crossed 1M+ downloads** :tada:: Thank you to our community! [**Twitter/X**](https://x.com/skypilot_org/status/1844770841718067638)

**LLM Finetuning Cookbooks**: Finetuning Llama 2 / Llama 3.1 in your own cloud environment, privately: Llama 2 [**example**](./llm/vicuna-llama-2/) and [**blog**](https://blog.skypilot.co/finetuning-llama2-operational-guide/); Llama 3.1 [**example**](./llm/llama-3_1-finetuning/) and [**blog**](https://blog.skypilot.co/finetune-llama-3_1-on-your-infra/)

---

## Getting Started

SkyPilot is an open-source framework that streamlines running your AI workloads.

### Installation

Install SkyPilot using pip:

```bash
# Choose your clouds:
pip install -U "skypilot[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```

For the latest features, use the nightly build or [install from source](https://docs.skypilot.co/en/latest/getting-started/installation.html):

```bash
# Choose your clouds:
pip install "skypilot-nightly[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```

### Quickstart

SkyPilot tasks are defined by resource requirements, data syncing, setup commands, and task commands. You can launch these tasks on any available cloud, simplifying your AI infrastructure management.

1.  **Define your task** in a YAML file (e.g., `my_task.yaml`):

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

2.  **Prepare your working directory:**

    ```bash
    git clone https://github.com/pytorch/examples.git ~/torch_examples
    ```

3.  **Launch your task:**

    ```bash
    sky launch my_task.yaml
    ```

SkyPilot will then handle:

*   Finding the most cost-effective infrastructure.
*   Provisioning the VM, with automatic failover.
*   Syncing your working directory.
*   Running setup and run commands.

Get started with SkyPilot by following the [Quickstart Guide](https://docs.skypilot.co/en/latest/getting-started/quickstart.html).

## Runnable Examples

Explore these examples to learn how to use SkyPilot for various AI tasks:

| Task        | Examples                                                                                                                                                                                                                                                                    |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Training    | [PyTorch](https://docs.skypilot.co/en/latest/getting-started/tutorial.html), [DeepSpeed](https://docs.skypilot.co/en/latest/examples/training/deepspeed.html), [Finetune Llama 3](https://docs.skypilot.co/en/latest/examples/training/llama-3_1-finetuning.html), [NeMo](https://docs.skypilot.co/en/latest/examples/training/nemo.html), [Ray](https://docs.skypilot.co/en/latest/examples/training/ray.html), [Unsloth](https://docs.skypilot.co/en/latest/examples/training/unsloth.html), [Jax/TPU](https://docs.skypilot.co/en/latest/examples/training/tpu.html) |
| Serving     | [vLLM](https://docs.skypilot.co/en/latest/examples/serving/vllm.html), [SGLang](https://docs.skypilot.co/en/latest/examples/serving/sglang.html), [Ollama](https://docs.skypilot.co/en/latest/examples/serving/ollama.html)                                                                                                                                                                           |
| Models      | [DeepSeek-R1](https://docs.skypilot.co/en/latest/examples/models/deepseek-r1.html), [Llama 3](https://docs.skypilot.co/en/latest/examples/models/llama-3.html), [CodeLlama](https://docs.skypilot.co/en/latest/examples/models/codellama.html), [Qwen](https://docs.skypilot.co/en/latest/examples/models/qwen.html), [Mixtral](https://docs.skypilot.co/en/latest/examples/models/mixtral.html)  |
| AI Apps     | [RAG](https://docs.skypilot.co/en/latest/examples/applications/rag.html), [vector databases](https://docs.skypilot.co/en/latest/examples/applications/vector_database.html) (ChromaDB, CLIP) |
| Frameworks  | [Airflow](https://docs.skypilot.co/en/latest/examples/frameworks/airflow.html), [Jupyter](https://docs.skypilot.co/en/latest/examples/frameworks/jupyter.html)                                                                                                                  |

Find more examples in the [`llm/`](https://github.com/skypilot-org/skypilot/tree/master/llm) and [`examples/`](https://github.com/skypilot-org/skypilot/tree/master/examples) directories.

## Supported Infrastructure

SkyPilot supports a wide range of infrastructure providers, including:

*   Kubernetes
*   AWS
*   GCP
*   Azure
*   OCI
*   Lambda Cloud
*   Fluidstack
*   RunPod
*   Cudo
*   Digital Ocean
*   Paperspace
*   Cloudflare
*   Samsung
*   IBM
*   Vast.ai
*   VMware vSphere
*   Nebius

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-dark.png">
    <img alt="SkyPilot" src="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-light.png" width=85%>
  </picture>
</p>
<!-- source xcf file: https://drive.google.com/drive/folders/1S_acjRsAD3T14qMeEnf6FFrIwHu_Gs_f?usp=drive_link -->

## More Information

*   [SkyPilot Overview](https://docs.skypilot.co/en/latest/overview.html)
*   [SkyPilot Documentation](https://docs.skypilot.co/en/latest/)
*   [SkyPilot Blog](https://blog.skypilot.co/)
*   [Community Spotlights](https://blog.skypilot.co/community/)
*   [SkyPilot Paper](https://www.usenix.org/system/files/nsdi23-yang-zongheng.pdf)
*   [Sky Computing Whitepaper](https://arxiv.org/abs/2205.07147)
*   [Sky Computing Vision Paper](https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s02-stoica.pdf)
*   [SkyServe: AI serving across regions and clouds](https://arxiv.org/pdf/2411.01438) (EuroSys 2025)
*   [Managed jobs spot instance policy](https://www.usenix.org/conference/nsdi24/presentation/wu-zhanghao)  (NSDI 2024)

SkyPilot was originally developed at the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley.

## Get Involved

*   [GitHub Repository](https://github.com/skypilot-org/skypilot)
*   [Open a GitHub issue](https://github.com/skypilot-org/skypilot/issues/new) for issues and feature requests.
*   Use [GitHub Discussions](https://github.com/skypilot-org/skypilot/discussions) for questions.
*   Join the [SkyPilot Slack](http://slack.skypilot.co) for general discussions.
*   Follow us on:
    *   [X / Twitter](https://twitter.com/skypilot_org)
    *   [LinkedIn](https://www.linkedin.com/company/skypilot-oss/)
    *   [SkyPilot Blog](https://blog.skypilot.co/)

## Contribute

We welcome contributions!  See the [CONTRIBUTING](CONTRIBUTING.md) guide.