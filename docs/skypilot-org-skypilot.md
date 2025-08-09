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

## SkyPilot: Run AI Workloads Faster, Cheaper, and on Any Cloud 🚀

SkyPilot is an open-source platform that simplifies and optimizes running AI and batch workloads across various cloud providers and infrastructure, offering a unified and cost-effective solution.  [Check out the original repo!](https://github.com/skypilot-org/skypilot)

**Key Features:**

*   **Multi-Cloud Support:** Run AI workloads on Kubernetes, AWS, GCP, Azure, and more.
*   **Cost Optimization:** Automatically selects the most cost-effective infrastructure and supports spot instances for significant savings.
*   **Simplified Deployment:**  Deploy your AI applications with ease using a simple YAML or Python API.
*   **Unified Interface:**  Manage your jobs across different clouds and hardware from a single interface.
*   **Auto-Scaling & Resource Management:** Efficiently manage and scale your resources.
*   **Kubernetes Integration:**  Provides a user-friendly experience for AI teams using Kubernetes.

### News & Updates

Stay up-to-date with the latest SkyPilot developments:

*   **[Aug 2025]** Serve and finetune **OpenAI GPT-OSS models** (gpt-oss-120b, gpt-oss-20b) with one command: [**serve**](./llm/gpt-oss/) + [**LoRA and full finetuning**](./llm/gpt-oss-finetuning/)
*   **[Jul 2025]** Run distributed **RL training for LLMs** with Verl (PPO, GRPO) on any cloud: [**example**](./llm/verl/)
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

**LLM Finetuning Cookbooks:** Finetuning Llama 2 / Llama 3.1 in your own cloud environment, privately: Llama 2 [**example**](./llm/vicuna-llama-2/) and [**blog**](https://blog.skypilot.co/finetuning-llama2-operational-guide/); Llama 3.1 [**example**](./llm/llama-3_1-finetuning/) and [**blog**](https://blog.skypilot.co/finetune-llama-3_1-on-your-infra/)

### Demo

#### [🌟 **SkyPilot Demo** 🌟: Click to see a 1-minute tour](https://demo.skypilot.co/dashboard/)

### Getting Started

Easily deploy your AI workloads with SkyPilot.

*   **[Installation](https://docs.skypilot.co/en/latest/getting-started/installation.html)**
*   **[Quickstart](https://docs.skypilot.co/en/latest/getting-started/quickstart.html)**
*   **[CLI Reference](https://docs.skypilot.co/en/latest/reference/cli.html)**

**Installation:**

```bash
# Choose your clouds:
pip install -U "skypilot[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```

**Nightly Build (for latest features):**

```bash
# Choose your clouds:
pip install "skypilot-nightly[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```

### SkyPilot in 1 Minute

SkyPilot uses a unified interface to define AI tasks, making them portable across different infrastructures:

1.  **Define Your Task:** Specify resource requirements, data, setup commands, and run commands in YAML or Python.
2.  **Launch with `sky launch`:**  SkyPilot finds the best infrastructure, provisions resources (VMs or pods), syncs your code, installs dependencies, and runs your job.

**Example `my_task.yaml`:**

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

Launch with `sky launch` (note: [access to GPU instances](https://docs.skypilot.co/en/latest/cloud-setup/quota.html) is needed for this example):
```bash
sky launch my_task.yaml
```

SkyPilot will:

1.  Find the cheapest & available infra across your clusters or clouds
2.  Provision the GPUs (pods or VMs), with auto-failover if the infra returned capacity errors
3.  Sync your local `workdir` to the provisioned cluster
4.  Auto-install dependencies by running the task's `setup` commands
5.  Run the task's `run` commands, and stream logs

### Runnable Examples

Explore [**SkyPilot examples**](https://docs.skypilot.co/en/docs-examples/examples/index.html) for various applications:

| Task       | Examples                                                                                                                                                                                                                                                                     |
| :--------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Training   | [Verl](https://docs.skypilot.co/en/latest/examples/training/verl.html), [Finetune Llama 4](https://docs.skypilot.co/en/latest/examples/training/llama-4-finetuning.html), [PyTorch](https://docs.skypilot.co/en/latest/getting-started/tutorial.html), [DeepSpeed](https://docs.skypilot.co/en/latest/examples/training/deepspeed.html), [NeMo](https://docs.skypilot.co/en/latest/examples/training/nemo.html), [Ray](https://docs.skypilot.co/en/latest/examples/training/ray.html), [Unsloth](https://docs.skypilot.co/en/latest/examples/training/unsloth.html), [Jax/TPU](https://docs.skypilot.co/en/latest/examples/training/tpu.html) |
| Serving    | [vLLM](https://docs.skypilot.co/en/latest/examples/serving/vllm.html), [SGLang](https://docs.skypilot.co/en/latest/examples/serving/sglang.html), [Ollama](https://docs.skypilot.co/en/latest/examples/serving/ollama.html)                                                           |
| Models     | [DeepSeek-R1](https://docs.skypilot.co/en/latest/examples/models/deepseek-r1.html), [Llama 3](https://docs.skypilot.co/en/latest/examples/models/llama-3.html), [CodeLlama](https://docs.skypilot.co/en/latest/examples/models/codellama.html), [Qwen](https://docs.skypilot.co/en/latest/examples/models/qwen.html), [Mixtral](https://docs.skypilot.co/en/latest/examples/models/mixtral.html) |
| AI apps    | [RAG](https://docs.skypilot.co/en/latest/examples/applications/rag.html), [vector databases](https://docs.skypilot.co/en/latest/examples/applications/vector_database.html) (ChromaDB, CLIP)                                                                                |
| Frameworks | [Airflow](https://docs.skypilot.co/en/latest/examples/frameworks/airflow.html), [Jupyter](https://docs.skypilot.co/en/latest/examples/frameworks/jupyter.html)                                                                                                                  |

Source files can be found in [`llm/`](https://github.com/skypilot-org/skypilot/tree/master/llm) and [`examples/`](https://github.com/skypilot-org/skypilot/tree/master/examples).

### Learn More

*   [SkyPilot Overview](https://docs.skypilot.co/en/latest/overview.html)
*   [SkyPilot Documentation](https://docs.skypilot.co/en/latest/)
*   [SkyPilot Blog](https://blog.skypilot.co/)
*   [Concept: Sky Computing](https://docs.skypilot.co/en/latest/sky-computing.html)

**Community Spotlights:** [Community Spotlights](https://blog.skypilot.co/community/)

**Stay Connected:**

*   [Slack](http://slack.skypilot.co)
*   [X / Twitter](https://twitter.com/skypilot_org)
*   [LinkedIn](https://www.linkedin.com/company/skypilot-oss/)
*   [SkyPilot Blog](https://blog.skypilot.co/) ([Introductory blog post](https://blog.skypilot.co/introducing-skypilot/))

**Research Papers:**

*   [SkyPilot paper](https://www.usenix.org/system/files/nsdi23-yang-zongheng.pdf) and [talk](https://www.usenix.org/conference/nsdi23/presentation/yang-zongheng) (NSDI 2023)
*   [Sky Computing whitepaper](https://arxiv.org/abs/2205.07147)
*   [Sky Computing vision paper](https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s02-stoica.pdf) (HotOS 2021)
*   [SkyServe: AI serving across regions and clouds](https://arxiv.org/pdf/2411.01438) (EuroSys 2025)
*   [Managed jobs spot instance policy](https://www.usenix.org/conference/nsdi24/presentation/wu-zhanghao)  (NSDI 2024)

### Contribute

We welcome contributions!  See [CONTRIBUTING](CONTRIBUTING.md) for details.

### Get Involved & Support

*   [Open a GitHub issue](https://github.com/skypilot-org/skypilot/issues/new) for issues/feature requests.
*   Use [GitHub Discussions](https://github.com/skypilot-org/skypilot/discussions) for questions.
*   Join the [SkyPilot Slack](http://slack.skypilot.co) for general discussions.