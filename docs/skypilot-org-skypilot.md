<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/skypilot-wide-dark-1k.png">
    <img alt="SkyPilot" src="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/skypilot-wide-light-1k.png" width=55%>
  </picture>
</p>

<h1 align="center">SkyPilot: Run AI Workloads Anywhere, Faster, and Cheaper</h1>

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
  <a href="https://github.com/skypilot-org/skypilot">
    <img alt="GitHub Repo" src="https://img.shields.io/badge/GitHub-Repo-black?logo=github">
  </a>
</p>


<div align="center">

#### [🌟 **SkyPilot Demo** 🌟: Click to see a 1-minute tour](https://demo.skypilot.co/dashboard/)

</div>

---

**SkyPilot** is an open-source framework that simplifies running AI and batch workloads across any infrastructure, enabling you to optimize your AI projects for cost, performance, and accessibility. **[Check out the GitHub repo](https://github.com/skypilot-org/skypilot) to get started!**

---

:fire: *News* :fire:
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

## Key Features of SkyPilot:

*   **Unified Infrastructure:** Run AI workloads on any infrastructure, including Kubernetes, AWS, GCP, Azure, and more. SkyPilot abstracts away the complexities of different cloud providers.
*   **Simplified AI Workflows:** Easily define your AI tasks using YAML or a Python API, making them portable and reproducible across different environments.
*   **Cost Optimization:** SkyPilot helps you reduce cloud costs through automatic resource selection, spot instance support, and auto-stopping of idle resources.
*   **High Availability:** Benefit from automatic retry mechanisms and intelligent scheduling to maximize GPU availability and minimize downtime.
*   **Team Collaboration:** SkyPilot supports team deployment and resource sharing, streamlining collaboration on AI projects.

## Benefits:

*   **Faster Development:** Quickly spin up compute resources and iterate on your AI models.
*   **Reduced Costs:** Optimize your spending on cloud resources with smart scheduling and cost-saving features.
*   **Increased Productivity:** Focus on your AI research and development, not on managing infrastructure.
*   **Vendor Agnostic:** Avoid vendor lock-in by easily switching between different cloud providers.

## Installation

Install SkyPilot using pip:

```bash
# Choose your clouds:
pip install -U "skypilot[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```

For the latest features, consider installing the nightly build or from source:

```bash
# Choose your clouds:
pip install "skypilot-nightly[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```

<p align="center">
  <img src="docs/source/_static/intro.gif" alt="SkyPilot">
</p>

## Supported Infrastructure

SkyPilot supports a wide range of infrastructure providers, including:

Kubernetes, AWS, GCP, Azure, OCI, Lambda Cloud, Fluidstack, RunPod, Cudo, Digital Ocean, Paperspace, Cloudflare, Samsung, IBM, Vast.ai, VMware vSphere, Nebius.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-dark.png">
    <img alt="SkyPilot" src="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-light.png" width=85%>
  </picture>
</p>
<!-- source xcf file: https://drive.google.com/drive/folders/1S_acjRsAD3T14qMeEnf6FFrIwHu_Gs_f?usp=drive_link -->

## Getting Started

Explore our comprehensive documentation to quickly get started with SkyPilot:

*   [Installation](https://docs.skypilot.co/en/latest/getting-started/installation.html)
*   [Quickstart](https://docs.skypilot.co/en/latest/getting-started/quickstart.html)
*   [CLI Reference](https://docs.skypilot.co/en/latest/reference/cli.html)

## SkyPilot in 1 Minute

SkyPilot streamlines AI task execution with a user-friendly interface.  A SkyPilot task includes resource requirements, data syncing details, setup commands, and the run commands.  These tasks can then be deployed on a choice of available clouds.

Here's how you define and launch a task:

1.  **Define your task:** Create a YAML file (e.g., `my_task.yaml`) with the following content:

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

2.  **Prepare the workdir:**

```bash
git clone https://github.com/pytorch/examples.git ~/torch_examples
```

3.  **Launch with `sky launch`:**

```bash
sky launch my_task.yaml
```

SkyPilot will then automatically:

1.  Find the most cost-effective VM instance across different clouds.
2.  Provision the VM with auto-failover.
3.  Sync the `workdir`.
4.  Execute the `setup` commands.
5.  Run the `run` commands.

See the [Quickstart](https://docs.skypilot.co/en/latest/getting-started/quickstart.html) guide for more details.

## Runnable Examples

SkyPilot provides numerous examples for development, training, serving, LLM models, and more:

| Task | Examples |
|----------|----------|
| Training | [PyTorch](https://docs.skypilot.co/en/latest/getting-started/tutorial.html), [DeepSpeed](https://docs.skypilot.co/en/latest/examples/training/deepspeed.html), [Finetune Llama 3](https://docs.skypilot.co/en/latest/examples/training/llama-3_1-finetuning.html), [NeMo](https://docs.skypilot.co/en/latest/examples/training/nemo.html), [Ray](https://docs.skypilot.co/en/latest/examples/training/ray.html), [Unsloth](https://docs.skypilot.co/en/latest/examples/training/unsloth.html), [Jax/TPU](https://docs.skypilot.co/en/latest/examples/training/tpu.html) |
| Serving | [vLLM](https://docs.skypilot.co/en/latest/examples/serving/vllm.html), [SGLang](https://docs.skypilot.co/en/latest/examples/serving/sglang.html), [Ollama](https://docs.skypilot.co/en/latest/examples/serving/ollama.html) |
| Models | [DeepSeek-R1](https://docs.skypilot.co/en/latest/examples/models/deepseek-r1.html), [Llama 3](https://docs.skypilot.co/en/latest/examples/models/llama-3.html), [CodeLlama](https://docs.skypilot.co/en/latest/examples/models/codellama.html), [Qwen](https://docs.skypilot.co/en/latest/examples/models/qwen.html), [Mixtral](https://docs.skypilot.co/en/latest/examples/models/mixtral.html) |
| AI apps | [RAG](https://docs.skypilot.co/en/latest/examples/applications/rag.html), [vector databases](https://docs.skypilot.co/en/latest/examples/applications/vector_database.html) (ChromaDB, CLIP) |
| Common frameworks | [Airflow](https://docs.skypilot.co/en/latest/examples/frameworks/airflow.html), [Jupyter](https://docs.skypilot.co/en/latest/examples/frameworks/jupyter.html) |

Find more examples in the [`llm/`](https://github.com/skypilot-org/skypilot/tree/master/llm) and [`examples/`](https://github.com/skypilot-org/skypilot/tree/master/examples) directories.

## More Information

*   [SkyPilot Overview](https://docs.skypilot.co/en/latest/overview.html)
*   [SkyPilot Documentation](https://docs.skypilot.co/en/latest/)
*   [SkyPilot Blog](https://blog.skypilot.co/)

**Community Spotlights:** [Community Spotlights](https://blog.skypilot.co/community/)

**Stay Updated:**

*   [Slack](http://slack.skypilot.co)
*   [X / Twitter](https://twitter.com/skypilot_org)
*   [LinkedIn](https://www.linkedin.com/company/skypilot-oss/)
*   [SkyPilot Blog](https://blog.skypilot.co/) ([Introductory blog post](https://blog.skypilot.co/introducing-skypilot/))

**Read the Research:**

*   [SkyPilot paper](https://www.usenix.org/system/files/nsdi23-yang-zongheng.pdf) and [talk](https://www.usenix.org/conference/nsdi23/presentation/yang-zongheng) (NSDI 2023)
*   [Sky Computing whitepaper](https://arxiv.org/abs/2205.07147)
*   [Sky Computing vision paper](https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s02-stoica.pdf) (HotOS 2021)
*   [SkyServe: AI serving across regions and clouds](https://arxiv.org/pdf/2411.01438) (EuroSys 2025)
*   [Managed jobs spot instance policy](https://www.usenix.org/conference/nsdi24/presentation/wu-zhanghao)  (NSDI 2024)

## Origin and Vision

SkyPilot was initially developed at the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley. To learn more, see [Concept: Sky Computing](https://docs.skypilot.co/en/latest/sky-computing.html).

## Questions and Feedback

*   For issues and feature requests, please [open a GitHub issue](https://github.com/skypilot-org/skypilot/issues/new).
*   For questions, use [GitHub Discussions](https://github.com/skypilot-org/skypilot/discussions).

Join the [SkyPilot Slack](http://slack.skypilot.co) for general discussions.

## Contributing

Contributions are welcome! See [CONTRIBUTING](CONTRIBUTING.md) for details.