<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/skypilot-wide-dark-1k.png">
    <img alt="SkyPilot" src="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/skypilot-wide-light-1k.png" width=55%>
  </picture>
</p>

<h1 align="center">SkyPilot: Run AI Workloads on Any Cloud, Faster & Cheaper</h1>

SkyPilot is an open-source framework that simplifies running AI and batch workloads across various cloud providers and infrastructure options. **[Explore the SkyPilot Repository](https://github.com/skypilot-org/skypilot)**

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

---

## Key Features of SkyPilot

*   **Unified Cloud Interface:** Run your workloads on any cloud or Kubernetes cluster with a single interface.
*   **Cost Optimization:** Automatically finds the most cost-effective infrastructure and supports spot instances for significant savings.
*   **Simplified Management:** Easily queue, run, and manage AI jobs, with built-in auto-recovery.
*   **Environment & Job as Code:** Define your environment and jobs in a portable, YAML-based format.
*   **Accelerated Workflows:** Supports a wide range of AI workloads (Training, Serving, LLMs, AI Apps, common frameworks).

---

## What's New

Stay up-to-date with the latest features and examples:

*   **[Apr 2025]** Spin up **Qwen3** on your cluster/cloud: [**example**](./llm/qwen/)
*   **[Mar 2025]** Run and serve **Google Gemma 3** using SkyPilot [**example**](./llm/gemma3/)
*   **[Feb 2025]** Prepare and serve **Retrieval Augmented Generation (RAG) with DeepSeek-R1**: [**blog post**](https://blog.skypilot.co/deepseek-rag), [**example**](./llm/rag/)
*   **[Feb 2025]** Run and serve **DeepSeek-R1 671B** using SkyPilot and SGLang with high throughput: [**example**](./llm/deepseek-r1/)
*   **[Feb 2025]** Prepare and serve large-scale image search with **vector databases**: [**blog post**](https://blog.skypilot.co/large-scale-vector-database/), [**example**](./examples/vector_database/)
*   **[Jan 2025]** Launch and serve distilled models from **[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)** and **[Janus](https://github.com/deepseek-ai/DeepSeek-Janus)** on Kubernetes or any cloud: [**R1 example**](./llm/deepseek-r1-distilled/) and [**Janus example**](./llm/deepseek-janus/)
*   **[Oct 2024]** :tada: **SkyPilot crossed 1M+ downloads** :tada:: Thank you to our community! [**Twitter/X**](https://x.com/skypilot_org/status/1844770841718067638)
*   **[Sep 2024]** Point, launch and serve **Llama 3.2** on Kubernetes or any cloud: [**example**](./llm/llama-3_2/)

**LLM Finetuning Cookbooks:** Finetuning Llama 2 / Llama 3.1 in your own cloud environment, privately: Llama 2 [**example**](./llm/vicuna-llama-2/) and [**blog**](https://blog.skypilot.co/finetuning-llama2-operational-guide/); Llama 3.1 [**example**](./llm/llama-3_1-finetuning/) and [**blog**](https://blog.skypilot.co/finetune-llama-3_1-on-your-infra/)

---

## SkyPilot in Action: How it Works

SkyPilot streamlines AI workload execution by abstracting away the complexities of cloud infrastructure.

Here's a quick overview:

1.  **Define Your Task:** Use a simple YAML file to specify your resource requirements, data, setup commands, and run commands.
2.  **Unified Interface:** Launch your task using `sky launch`, which can be run on any available cloud.
3.  **Automatic Management:** SkyPilot handles the heavy lifting:
    *   Finding the best-priced VM instance.
    *   Provisioning the VM with auto-failover.
    *   Syncing your code and data.
    *   Executing setup and run commands.

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
  pip install "torch<2.2" torchvision --index-url https://download.pytorch.org/whl/cu121

# Commands to run as a job.
# Typical use: launch the main program.
run: |
  cd mnist
  python main.py --epochs 1
```

To launch, run:

```bash
git clone https://github.com/pytorch/examples.git ~/torch_examples
sky launch my_task.yaml
```

---

## Supported Infrastructure

SkyPilot supports a wide range of cloud providers and infrastructure options:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-dark.png">
    <img alt="SkyPilot" src="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-light.png" width=85%>
  </picture>
</p>

Includes: Kubernetes, AWS, GCP, Azure, OCI, Lambda Cloud, Fluidstack, RunPod, Cudo, Digital Ocean, Paperspace, Cloudflare, Samsung, IBM, Vast.ai, VMware vSphere, Nebius.

---

## Getting Started

*   **Documentation:** [SkyPilot Documentation](https://docs.skypilot.co/)
    *   [Installation](https://docs.skypilot.co/en/latest/getting-started/installation.html)
    *   [Quickstart](https://docs.skypilot.co/en/latest/getting-started/quickstart.html)
    *   [CLI Reference](https://docs.skypilot.co/en/latest/reference/cli.html)
*   **Install SkyPilot:**

    ```bash
    # Choose your clouds:
    pip install -U "skypilot[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
    ```
    or the nightly build:
     ```bash
    # Choose your clouds:
    pip install "skypilot-nightly[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
    ```

---

## Examples & Use Cases

Explore ready-to-use examples for various AI tasks:

*   **Training:** PyTorch, DeepSpeed, Finetune Llama 3, NeMo, Ray, Unsloth, Jax/TPU
*   **Serving:** vLLM, SGLang, Ollama
*   **Models:** DeepSeek-R1, Llama 3, CodeLlama, Qwen, Mixtral
*   **AI Apps:** RAG, vector databases (ChromaDB, CLIP)
*   **Frameworks:** Airflow, Jupyter

Find more examples in the [`llm/`](https://github.com/skypilot-org/skypilot/tree/master/llm) and [`examples/`](https://github.com/skypilot-org/skypilot/tree/master/examples) directories.

---

## Resources

*   **SkyPilot Overview:** [SkyPilot Overview](https://docs.skypilot.co/en/latest/overview.html)
*   **SkyPilot Docs:** [SkyPilot Documentation](https://docs.skypilot.co/en/latest/)
*   **SkyPilot Blog:** [SkyPilot Blog](https://blog.skypilot.co/)
*   **Community Spotlights:** [Community Spotlights](https://blog.skypilot.co/community/)
*   **SkyPilot Paper:** [SkyPilot Paper](https://www.usenix.org/system/files/nsdi23-yang-zongheng.pdf) (NSDI 2023)
*   **Sky Computing Whitepaper:** [Sky Computing whitepaper](https://arxiv.org/abs/2205.07147)

---

## Get Involved

*   **Issues and Feature Requests:** [Open a GitHub Issue](https://github.com/skypilot-org/skypilot/issues/new)
*   **Discussions:** [GitHub Discussions](https://github.com/skypilot-org/skypilot/discussions)
*   **Join the Community:** [SkyPilot Slack](http://slack.skypilot.co)

---

## Contributing

Contributions are welcome! See [CONTRIBUTING](CONTRIBUTING.md) for guidelines.