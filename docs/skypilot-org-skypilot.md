<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/skypilot-wide-dark-1k.png">
    <img alt="SkyPilot" src="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/skypilot-wide-light-1k.png" width=55%>
  </picture>
</p>

<h1 align="center">SkyPilot: Run AI on Any Infrastructure, Faster and Cheaper</h1>

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

SkyPilot is an open-source framework that simplifies running AI and batch workloads on any infrastructure, offering a unified, cost-effective, and efficient solution.  **[Explore the SkyPilot repository](https://github.com/skypilot-org/skypilot) to accelerate your AI workflows!**

## Key Features of SkyPilot

*   **Unified Infrastructure:** Run your AI workloads seamlessly across various clouds (AWS, GCP, Azure, and more), Kubernetes, and on-premise resources.
*   **Cost Optimization:** Leverage spot instances for significant cost savings (3-6x) and automatic resource scaling and cleanup to reduce cloud expenses.
*   **Simplified Deployment:** Use a unified YAML-based interface to define, launch, and manage your AI jobs, enabling easy portability and reproducibility.
*   **Automated Resource Management:** SkyPilot automatically handles resource provisioning, including GPU and TPU selection, with intelligent scheduling and auto-retry mechanisms.
*   **Fast & Flexible:** Quickly spin up compute, manage jobs efficiently (queue, run, auto-recover), and support your existing GPU, TPU, and CPU workloads without code changes.

## What's New (Recent Updates)

*   [Apr 2025] Spin up **Qwen3** on your cluster/cloud: [**example**](./llm/qwen/)
*   [Mar 2025] Run and serve **Google Gemma 3** using SkyPilot [**example**](./llm/gemma3/)
*   [Feb 2025] Prepare and serve **Retrieval Augmented Generation (RAG) with DeepSeek-R1**: [**blog post**](https://blog.skypilot.co/deepseek-rag), [**example**](./llm/rag/)
*   [Feb 2025] Run and serve **DeepSeek-R1 671B** using SkyPilot and SGLang with high throughput: [**example**](./llm/deepseek-r1/)
*   [Feb 2025] Prepare and serve large-scale image search with **vector databases**: [**blog post**](https://blog.skypilot.co/large-scale-vector-database/), [**example**](./examples/vector_database/)
*   [Jan 2025] Launch and serve distilled models from **[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)** and **[Janus](https://github.com/deepseek-ai/DeepSeek-Janus)** on Kubernetes or any cloud: [**R1 example**](./llm/deepseek-r1-distilled/) and [**Janus example**](./llm/deepseek-janus/)
*   [Oct 2024] :tada: **SkyPilot crossed 1M+ downloads** :tada:: Thank you to our community! [**Twitter/X**](https://x.com/skypilot_org/status/1844770841718067638)
*   [Sep 2024] Point, launch and serve **Llama 3.2** on Kubernetes or any cloud: [**example**](./llm/llama-3_2/)

**LLM Finetuning Cookbooks**: Finetuning Llama 2 / Llama 3.1 in your own cloud environment, privately: Llama 2 [**example**](./llm/vicuna-llama-2/) and [**blog**](https://blog.skypilot.co/finetuning-llama2-operational-guide/); Llama 3.1 [**example**](./llm/llama-3_1-finetuning/) and [**blog**](https://blog.skypilot.co/finetune-llama-3_1-on-your-infra/)

## Getting Started

### Installation

```bash
# Choose your clouds:
pip install -U "skypilot[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```

For the latest features and fixes, use the nightly build or [install from source](https://docs.skypilot.co/en/latest/getting-started/installation.html):

```bash
# Choose your clouds:
pip install "skypilot-nightly[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```

### Quickstart

For detailed instructions and examples, refer to:
*   [SkyPilot Documentation](https://docs.skypilot.co/)
*   [Quickstart Guide](https://docs.skypilot.co/en/latest/getting-started/quickstart.html)
*   [CLI Reference](https://docs.skypilot.co/en/latest/reference/cli.html)

## SkyPilot in 1 Minute: An Example

Define your task, specify resource requirements, sync data, and run commands:

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
  pip install "torch<2.2" torchvision --index-url https://download.pytorch.org/whl/cu121

# Commands to run as a job.
# Typical use: launch the main program.
run: |
  cd mnist
  python main.py --epochs 1
```

Prepare the workdir by cloning:
```bash
git clone https://github.com/pytorch/examples.git ~/torch_examples
```

Launch with `sky launch` (note: [access to GPU instances](https://docs.skypilot.co/en/latest/cloud-setup/quota.html) is needed for this example):
```bash
sky launch my_task.yaml
```

SkyPilot automatically handles the following:

1.  Find the lowest-priced VM instance.
2.  Provision the VM.
3.  Sync the `workdir`.
4.  Run the `setup` commands.
5.  Run the `run` commands.

## Runnable Examples

Explore comprehensive examples covering training, serving, LLM models, AI apps, and common frameworks:

*   **Training:** PyTorch, DeepSpeed, Finetune Llama 3, NeMo, Ray, Unsloth, Jax/TPU
*   **Serving:** vLLM, SGLang, Ollama
*   **Models:** DeepSeek-R1, Llama 3, CodeLlama, Qwen, Mixtral
*   **AI apps:** RAG, vector databases (ChromaDB, CLIP)
*   **Common Frameworks:** Airflow, Jupyter

Find more examples in [`llm/`](https://github.com/skypilot-org/skypilot/tree/master/llm) and [`examples/`](https://github.com/skypilot-org/skypilot/tree/master/examples).

## Supported Infrastructure

SkyPilot supports a wide range of infrastructure providers:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-dark.png">
    <img alt="SkyPilot" src="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-light.png" width=85%>
  </picture>
</p>

Currently supported: Kubernetes, AWS, GCP, Azure, OCI, Lambda Cloud, Fluidstack, RunPod, Cudo, Digital Ocean, Paperspace, Cloudflare, Samsung, IBM, Vast.ai, VMware vSphere, Nebius.

## More Information

*   [SkyPilot Overview](https://docs.skypilot.co/en/latest/overview.html)
*   [SkyPilot Documentation](https://docs.skypilot.co/en/latest/)
*   [SkyPilot Blog](https://blog.skypilot.co/)

**Stay Updated:**

*   [Slack](http://slack.skypilot.co)
*   [X / Twitter](https://twitter.com/skypilot_org)
*   [LinkedIn](https://www.linkedin.com/company/skypilot-oss/)
*   [SkyPilot Blog](https://blog.skypilot.co/)

**Research:**

*   [SkyPilot paper](https://www.usenix.org/system/files/nsdi23-yang-zongheng.pdf) (NSDI 2023)
*   [Sky Computing whitepaper](https://arxiv.org/abs/2205.07147)
*   [Sky Computing vision paper](https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s02-stoica.pdf) (HotOS 2021)
*   [SkyServe: AI serving across regions and clouds](https://arxiv.org/pdf/2411.01438) (EuroSys 2025)
*   [Managed jobs spot instance policy](https://www.usenix.org/conference/nsdi24/presentation/wu-zhanghao)  (NSDI 2024)

## Get Involved

*   **Issues and Feature Requests:** [Open a GitHub issue](https://github.com/skypilot-org/skypilot/issues/new).
*   **Questions and Discussions:** Use [GitHub Discussions](https://github.com/skypilot-org/skypilot/discussions).
*   **General Discussions:** Join the [SkyPilot Slack](http://slack.skypilot.co).
*   **Contributing:** See [CONTRIBUTING](CONTRIBUTING.md).