<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/skypilot-wide-dark-1k.png">
    <img alt="SkyPilot" src="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/skypilot-wide-light-1k.png" width=55%>
  </picture>
</p>

<h1 align="center">SkyPilot: Run Your AI Workloads, Anywhere, Easily, and Cost-Effectively</h1>

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

<p align="center">
  <a href="https://demo.skypilot.co/dashboard/">
    <img alt="SkyPilot Demo" src="https://img.shields.io/badge/Demo-See%20it%20in%20Action-blue?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAACXBIWXMAAAFiAAABYgU7x5qMAAAAGXRFWHRTb2Z0d2FyZQB3d3cubm9uZS5vcmc//hLwAAACSSURBVEiJ7ZCxCsAgDEO/v+0O4uT/82E+N66/6Xf24M664w+7WjYg8v6G2D+gJ8kRkQdYmQk6jHlq4eLg+bW9Lh+vLw7n1dK7X67nQJ8M39yN6w19M6gX95s51O8N0L6P8g/n3BvYFz2sH/QAAAABJRU5ErkJggg==">
  </a>
</p>

----

## Key Features

*   **Unified Platform:** Run AI workloads on any infrastructure, including Kubernetes, AWS, GCP, Azure, and more.
*   **Simplified Deployment:** Define your AI jobs and environments as code for easy portability and reproducibility.
*   **Cost Optimization:** Automatically leverages spot instances and intelligent scheduling for significant cost savings.
*   **Automated Management:**  Handles resource provisioning, job queuing, auto-retry, and auto-stop for efficient resource utilization.
*   **Scalability and Flexibility:** Supports a wide range of hardware including GPUs, TPUs, and CPUs, with flexible provisioning options.

----

### Recent Updates

*   **[Jul 2025]** 🎉 SkyPilot v0.10.0 released! [**blog post**](https://blog.skypilot.co/announcing-skypilot-0.10.0/), [**release notes**](https://github.com/skypilot-org/skypilot/releases/tag/v0.10.0)
*   **[Jul 2025]** Finetune **Llama4** on any distributed cluster/cloud: [**example**](./llm/llama-4-finetuning/)
*   **[Jul 2025]** Two-part blog series, `The Evolution of AI Job Orchestration`: (1) [Running AI jobs on GPU Neoclouds](https://blog.skypilot.co/ai-job-orchestration-pt1-gpu-neoclouds/), (2) [The AI-Native Control Plane & Orchestration that Finally Works for ML](https://blog.skypilot.co/ai-job-orchestration-pt2-ai-control-plane/)
*   **[Apr 2025]** Spin up **Qwen3** on your cluster/cloud: [**example**](./llm/qwen/)
*   **[Mar 2025]** Run and serve **Google Gemma 3** using SkyPilot [**example**](./llm/gemma3/)
*   **[Feb 2025]** Prepare and serve **Retrieval Augmented Generation (RAG) with DeepSeek-R1**: [**blog post**](https://blog.skypilot.co/deepseek-rag), [**example**](./llm/rag/)
*   **[Feb 2025]** Run and serve **DeepSeek-R1 671B** using SkyPilot and SGLang with high throughput: [**example**](./llm/deepseek-r1/)
*   **[Feb 2025]** Prepare and serve large-scale image search with **vector databases**: [**blog post**](https://blog.skypilot.co/large-scale-vector-database/), [**example**](./examples/vector_database/)
*   **[Jan 2025]** Launch and serve distilled models from **[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)** and **[Janus](https://github.com/deepseek-ai/DeepSeek-Janus)** on Kubernetes or any cloud: [**R1 example**](./llm/deepseek-r1-distilled/) and [**Janus example**](./llm/deepseek-janus/)
*   **[Oct 2024]** :tada: **SkyPilot crossed 1M+ downloads** :tada:: Thank you to our community! [**Twitter/X**](https://x.com/skypilot_org/status/1844770841718067638)

**LLM Finetuning Cookbooks**: Finetuning Llama 2 / Llama 3.1 in your own cloud environment, privately: Llama 2 [**example**](./llm/vicuna-llama-2/) and [**blog**](https://blog.skypilot.co/finetuning-llama2-operational-guide/); Llama 3.1 [**example**](./llm/llama-3_1-finetuning/) and [**blog**](https://blog.skypilot.co/finetune-llama-3_1-on-your-infra/)

----

SkyPilot is an open-source framework designed to streamline the process of running AI and batch workloads on diverse infrastructure, offering a unified, faster, and more cost-effective approach.  Whether you're training large language models, serving AI applications, or managing complex data pipelines, SkyPilot provides a robust and flexible solution.

### Why Use SkyPilot?

*   **Easy to Use:** Simplify your AI workflow with intuitive tools and a user-friendly interface.
*   **Infra Agnostic:**  Run your workloads across multiple clouds and hardware with a single command.
*   **Cost-Effective:**  Reduce your cloud spending with intelligent resource management and spot instance support.
*   **Scalable and Reliable:** Handle large-scale workloads with automated scaling and fault tolerance.

### Installation

Install SkyPilot using pip:

```bash
# Choose your clouds:
pip install -U "skypilot[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```

For the latest features and fixes, install from the nightly build or [from source](https://docs.skypilot.co/en/latest/getting-started/installation.html):

```bash
# Choose your clouds:
pip install "skypilot-nightly[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```

<p align="center">
  <img src="docs/source/_static/intro.gif" alt="SkyPilot">
</p>

**Supported Infrastructure:** Kubernetes, AWS, GCP, Azure, OCI, Lambda Cloud, Fluidstack, RunPod, Cudo, Digital Ocean, Paperspace, Cloudflare, Samsung, IBM, Vast.ai, VMware vSphere, Nebius.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-dark.png">
    <img alt="SkyPilot" src="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-light.png" width=85%>
  </picture>
</p>

## Getting Started

*   **[Documentation](https://docs.skypilot.co/):** Comprehensive documentation to get you started.
*   [Installation](https://docs.skypilot.co/en/latest/getting-started/installation.html)
*   [Quickstart](https://docs.skypilot.co/en/latest/getting-started/quickstart.html)
*   [CLI Reference](https://docs.skypilot.co/en/latest/reference/cli.html)

## SkyPilot in 1 Minute: A Simple Example

A SkyPilot task defines resource needs, data syncing instructions, setup and task commands, simplifying complex AI workflows.

Define the task in a YAML file, like `my_task.yaml`:

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

Prepare the workdir by cloning:
```bash
git clone https://github.com/pytorch/examples.git ~/torch_examples
```

Launch the task with `sky launch`:
```bash
sky launch my_task.yaml
```

SkyPilot will:

1.  Find the most cost-effective VM instance.
2.  Provision the VM, automatically recovering from capacity errors.
3.  Sync your working directory.
4.  Execute setup commands.
5.  Run your main task.

See the [Quickstart](https://docs.skypilot.co/en/latest/getting-started/quickstart.html) for a step-by-step guide.

## Example Workloads

Discover practical examples for a wide range of applications:

*   **Training:** PyTorch, DeepSpeed, Llama 3 Finetuning, NeMo, Ray, Unsloth, Jax/TPU
*   **Serving:** vLLM, SGLang, Ollama
*   **Models:** DeepSeek-R1, Llama 3, CodeLlama, Qwen, Mixtral
*   **AI Apps:** RAG, vector databases (ChromaDB, CLIP)
*   **Common Frameworks:** Airflow, Jupyter

Find source files and detailed examples in the [`llm/`](https://github.com/skypilot-org/skypilot/tree/master/llm) and [`examples/`](https://github.com/skypilot-org/skypilot/tree/master/examples) directories.

## Resources

*   **[SkyPilot Overview](https://docs.skypilot.co/en/latest/overview.html)**: Comprehensive guide to SkyPilot.
*   **[SkyPilot Documentation](https://docs.skypilot.co/en/latest/)**: Detailed documentation for all features.
*   **[SkyPilot Blog](https://blog.skypilot.co/)**: Stay up to date with the latest news and tutorials.
*   **[Community Spotlights](https://blog.skypilot.co/community/)**: Case studies and integrations.

### Stay Connected

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

SkyPilot was initially developed at the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley.

## Get Involved

*   **[Contribute](CONTRIBUTING.md)**:  Learn how to contribute to the project.
*   **[Open a GitHub issue](https://github.com/skypilot-org/skypilot/issues/new)**: Report issues and request features.
*   **[GitHub Discussions](https://github.com/skypilot-org/skypilot/discussions)**: Ask questions and engage in discussions.
*   **[SkyPilot Slack](http://slack.skypilot.co)**: Join the community for general discussions.

##  [SkyPilot - The Open Source Framework for Effortless AI Workload Management](https://github.com/skypilot-org/skypilot)
```
Key improvements:

*   **SEO-Optimized Title & Description:**  Includes keywords like "AI," "workloads," "cloud," "GPU," "training," "serving" and "cost-effective" to boost search visibility.
*   **Concise Hook:** A clear, benefit-driven sentence at the beginning to immediately grab the reader's attention.
*   **Organized Structure:** Uses headings, subheadings, and bullet points to improve readability and scannability.
*   **Clear Value Proposition:**  Highlights key features and benefits in an easy-to-understand format.
*   **Call to Action:**  Encourages users to explore the documentation and examples.
*   **Contextual Links:** Uses descriptive anchor text for all links and includes multiple links to the original repo.
*   **Updated News Section**: News is now included in a section titled "Recent Updates"
*   **Clear and concise language**: Removed extra fluff.