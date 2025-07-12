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

<h1 align="center">SkyPilot: Run AI on Any Infrastructure, Simply and Affordably</h1>

---

## Key Features

*   **Unified Infrastructure**: Run your AI workloads across any cloud (AWS, GCP, Azure, and more) and on-premise infrastructure with a single, easy-to-use interface.
*   **Cost Optimization**: Automatically leverage spot instances for significant cost savings (3-6x) and intelligent scheduling to find the cheapest and most available resources.
*   **Simplified Deployment**: Define your AI tasks and environments as code, enabling portability and reproducibility across different infrastructures.
*   **Automated Resource Management**: SkyPilot handles provisioning, auto-scaling, and auto-recovery, freeing you from infrastructure management complexities.
*   **Accelerated AI Development**: Easily deploy and scale your AI models, with built-in support for common AI frameworks.

---

### What's New

*   **[Apr 2025]** Run **Qwen3** on your cluster/cloud: [**example**](./llm/qwen/)
*   **[Mar 2025]** Run and serve **Google Gemma 3** using SkyPilot [**example**](./llm/gemma3/)
*   **[Feb 2025]** Prepare and serve **Retrieval Augmented Generation (RAG) with DeepSeek-R1**: [**blog post**](https://blog.skypilot.co/deepseek-rag), [**example**](./llm/rag/)
*   **[Feb 2025]** Run and serve **DeepSeek-R1 671B** using SkyPilot and SGLang with high throughput: [**example**](./llm/deepseek-r1/)
*   **[Feb 2025]** Prepare and serve large-scale image search with **vector databases**: [**blog post**](https://blog.skypilot.co/large-scale-vector-database/), [**example**](./examples/vector_database/)
*   **[Jan 2025]** Launch and serve distilled models from **[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)** and **[Janus](https://github.com/deepseek-ai/DeepSeek-Janus)** on Kubernetes or any cloud: [**R1 example**](./llm/deepseek-r1-distilled/) and [**Janus example**](./llm/deepseek-janus/)
*   **[Oct 2024]** :tada: **SkyPilot crossed 1M+ downloads** :tada:: Thank you to our community! [**Twitter/X**](https://x.com/skypilot_org/status/1844770841718067638)
*   **[Sep 2024]** Point, launch and serve **Llama 3.2** on Kubernetes or any cloud: [**example**](./llm/llama-3_2/)

**LLM Finetuning Cookbooks**: Finetuning Llama 2 / Llama 3.1 in your own cloud environment, privately: Llama 2 [**example**](./llm/vicuna-llama-2/) and [**blog**](https://blog.skypilot.co/finetuning-llama2-operational-guide/); Llama 3.1 [**example**](./llm/llama-3_1-finetuning/) and [**blog**](https://blog.skypilot.co/finetune-llama-3_1-on-your-infra/)

---

## Getting Started

SkyPilot simplifies the process of running AI workloads, offering a unified platform for diverse infrastructure.

### Installation

Install SkyPilot using pip:

```bash
# Choose your clouds:
pip install -U "skypilot[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```

For the latest features, use the nightly build or install from source:

```bash
# Choose your clouds:
pip install "skypilot-nightly[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```

### Quickstart

1.  **Define Your Task**: Create a YAML file (e.g., `my_task.yaml`) to specify resource requirements, data, setup commands, and the job's commands.
2.  **Launch Your Task**: Use the `sky launch` command to launch the task on any cloud. SkyPilot handles the underlying infrastructure.

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

```bash
# Prepare the workdir by cloning:
git clone https://github.com/pytorch/examples.git ~/torch_examples

# Launch the task
sky launch my_task.yaml
```

For a more detailed tutorial, see the [Quickstart](https://docs.skypilot.co/en/latest/getting-started/quickstart.html) in the documentation.

---

## SkyPilot in 1 Minute

SkyPilot streamlines AI workload execution by abstracting away infrastructure complexities.  Define your task in YAML, and SkyPilot handles the rest:

1.  Finds the lowest-priced VM instance.
2.  Provisions the VM with auto-failover.
3.  Syncs your local work directory.
4.  Executes setup commands.
5.  Runs your job's commands.

## Runnable Examples

SkyPilot provides numerous examples to help you get started quickly.

*   **Training**: [PyTorch](https://docs.skypilot.co/en/latest/getting-started/tutorial.html), [DeepSpeed](https://docs.skypilot.co/en/latest/examples/training/deepspeed.html), [Finetune Llama 3](https://docs.skypilot.co/en/latest/examples/training/llama-3_1-finetuning.html), [NeMo](https://docs.skypilot.co/en/latest/examples/training/nemo.html), [Ray](https://docs.skypilot.co/en/latest/examples/training/ray.html), [Unsloth](https://docs.skypilot.co/en/latest/examples/training/unsloth.html), [Jax/TPU](https://docs.skypilot.co/en/latest/examples/training/tpu.html)
*   **Serving**: [vLLM](https://docs.skypilot.co/en/latest/examples/serving/vllm.html), [SGLang](https://docs.skypilot.co/en/latest/examples/serving/sglang.html), [Ollama](https://docs.skypilot.co/en/latest/examples/serving/ollama.html)
*   **Models**: [DeepSeek-R1](https://docs.skypilot.co/en/latest/examples/models/deepseek-r1.html), [Llama 3](https://docs.skypilot.co/en/latest/examples/models/llama-3.html), [CodeLlama](https://docs.skypilot.co/en/latest/examples/models/codellama.html), [Qwen](https://docs.skypilot.co/en/latest/examples/models/qwen.html), [Mixtral](https://docs.skypilot.co/en/latest/examples/models/mixtral.html)
*   **AI Apps**: [RAG](https://docs.skypilot.co/en/latest/examples/applications/rag.html), [vector databases](https://docs.skypilot.co/en/latest/examples/applications/vector_database.html) (ChromaDB, CLIP)
*   **Common Frameworks**: [Airflow](https://docs.skypilot.co/en/latest/examples/frameworks/airflow.html), [Jupyter](https://docs.skypilot.co/en/latest/examples/frameworks/jupyter.html)

More examples can be found in [`llm/`](https://github.com/skypilot-org/skypilot/tree/master/llm) and [`examples/`](https://github.com/skypilot-org/skypilot/tree/master/examples).

---

## Supported Infrastructure

SkyPilot supports a wide range of infrastructure providers:

<p align="center">
  <img src="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-light.png" alt="SkyPilot" width=85%>
</p>

---

## Learn More

*   [SkyPilot Documentation](https://docs.skypilot.co/en/latest/)
*   [SkyPilot Overview](https://docs.skypilot.co/en/latest/overview.html)
*   [SkyPilot Blog](https://blog.skypilot.co/)

### Additional Resources

*   [SkyPilot Paper](https://www.usenix.org/system/files/nsdi23-yang-zongheng.pdf) (NSDI 2023)
*   [Sky Computing Whitepaper](https://arxiv.org/abs/2205.07147)
*   [Sky Computing Vision Paper](https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s02-stoica.pdf) (HotOS 2021)

For more details, see [Concept: Sky Computing](https://docs.skypilot.co/en/latest/sky-computing.html), the foundation for SkyPilot.

---

## Get Involved

*   **[Contribute](CONTRIBUTING.md)**: We welcome contributions!
*   **[Open a GitHub issue](https://github.com/skypilot-org/skypilot/issues/new)** for issues and feature requests.
*   **[GitHub Discussions](https://github.com/skypilot-org/skypilot/discussions)** for questions.
*   **[SkyPilot Slack](http://slack.skypilot.co)** for general discussions.

---

**[Visit the SkyPilot GitHub Repository](https://github.com/skypilot-org/skypilot)**