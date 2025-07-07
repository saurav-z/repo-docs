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

<h2 align="center">SkyPilot: Run Your AI Workloads Anywhere, Faster, and Cheaper</h2>

SkyPilot is an open-source framework that simplifies and accelerates AI and batch workloads on any infrastructure, giving you maximum flexibility and cost savings.  [Check out the original repository](https://github.com/skypilot-org/skypilot) for more details.

----
:fire: *News* :fire:
- [Apr 2025] Spin up **Qwen3** on your cluster/cloud: [**example**](./llm/qwen/)
- [Mar 2025] Run and serve **Google Gemma 3** using SkyPilot [**example**](./llm/gemma3/)
- [Feb 2025] Prepare and serve **Retrieval Augmented Generation (RAG) with DeepSeek-R1**: [**blog post**](https://blog.skypilot.co/deepseek-rag), [**example**](./llm/rag/)
- [Feb 2025] Run and serve **DeepSeek-R1 671B** using SkyPilot and SGLang with high throughput: [**example**](./llm/deepseek-r1/)
- [Feb 2025] Prepare and serve large-scale image search with **vector databases**: [**blog post**](https://blog.skypilot.co/large-scale-vector-database/), [**example**](./examples/vector_database/)
- [Jan 2025] Launch and serve distilled models from **[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)** and **[Janus](https://github.com/deepseek-ai/DeepSeek-Janus)** on Kubernetes or any cloud: [**R1 example**](./llm/deepseek-r1-distilled/) and [**Janus example**](./llm/deepseek-janus/)
- [Oct 2024] :tada: **SkyPilot crossed 1M+ downloads** :tada:: Thank you to our community! [**Twitter/X**](https://x.com/skypilot_org/status/1844770841718067638)
- [Sep 2024] Point, launch and serve **Llama 3.2** on Kubernetes or any cloud: [**example**](./llm/llama-3_2/)


**LLM Finetuning Cookbooks**: Finetuning Llama 2 / Llama 3.1 in your own cloud environment, privately: Llama 2 [**example**](./llm/vicuna-llama-2/) and [**blog**](https://blog.skypilot.co/finetuning-llama2-operational-guide/); Llama 3.1 [**example**](./llm/llama-3_1-finetuning/) and [**blog**](https://blog.skypilot.co/finetune-llama-3_1-on-your-infra/)

----

## Key Features of SkyPilot

*   **Unified Infrastructure:**  Run AI workloads across 16+ clouds, Kubernetes, and existing infrastructure with a single interface, eliminating vendor lock-in.
*   **Simplified Workflow:**  Define your environment and jobs as code (YAML or Python), making your projects portable and reproducible.
*   **Cost Optimization:**  Leverage features like auto-stop, spot instance support, and intelligent scheduling to minimize cloud costs and maximize GPU availability, with potential savings of 3-6x.
*   **Easy Management:**  Manage your jobs with ease: queue, run, and auto-recover many jobs.
*   **Flexible Provisioning:** Easily provision GPU, TPU, and CPU resources, with automatic retry mechanisms for enhanced reliability.
*   **Team Deployment:** Support team deployment and resource sharing.

## Why Choose SkyPilot?

SkyPilot empowers you to:

*   **Accelerate AI Development:** Quickly spin up compute resources on your preferred infrastructure.
*   **Reduce Costs:** Optimize your cloud spending with automated cost-saving features.
*   **Maximize Availability:** Ensure consistent access to resources through intelligent scheduling and auto-recovery.

## Getting Started

**Installation:**

```bash
# Choose your clouds:
pip install -U "skypilot[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```

For the latest features and fixes, consider the nightly build or [install from source](https://docs.skypilot.co/en/latest/getting-started/installation.html):

```bash
# Choose your clouds:
pip install "skypilot-nightly[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```

<p align="center">
  <img src="docs/source/_static/intro.gif" alt="SkyPilot">
</p>

**Supported Infrastructure:** Kubernetes, AWS, GCP, Azure, OCI, Lambda Cloud, Fluidstack,
RunPod, Cudo, Digital Ocean, Paperspace, Cloudflare, Samsung, IBM, Vast.ai,
VMware vSphere, Nebius.
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-dark.png">
    <img alt="SkyPilot" src="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-light.png" width=85%>
  </picture>
</p>
<!-- source xcf file: https://drive.google.com/drive/folders/1S_acjRsAD3T14qMeEnf6FFrIwHu_Gs_f?usp=drive_link -->

## SkyPilot in 1 Minute

SkyPilot streamlines your AI workflows by allowing you to define tasks using a unified interface (YAML or Python API). These tasks specify resource requirements, data synchronization, setup commands, and the job's execution commands. You can then launch these tasks on any available cloud, eliminating vendor lock-in and enabling easy migration between providers.

**Example Task (my\_task.yaml):**

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

**Launch with:**
```bash
git clone https://github.com/pytorch/examples.git ~/torch_examples
sky launch my_task.yaml
```

SkyPilot automatically handles:

1.  Finding the most cost-effective VM instance.
2.  Provisioning the VM with automatic failover.
3.  Syncing the local `workdir`.
4.  Executing the `setup` commands.
5.  Running the `run` commands.

See the [Quickstart](https://docs.skypilot.co/en/latest/getting-started/quickstart.html) for more.

## Runnable Examples and Resources

Explore these resources to get started:

*   **SkyPilot Examples:** ([SkyPilot examples](https://docs.skypilot.co/en/docs-examples/examples/index.html)) for development, training, serving, LLM models, AI apps, and common frameworks.
*   **Latest Featured Examples:**

    | Task        | Examples                                                                                                                                   |
    | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
    | Training    | [PyTorch](https://docs.skypilot.co/en/latest/getting-started/tutorial.html), [DeepSpeed](https://docs.skypilot.co/en/latest/examples/training/deepspeed.html), [Finetune Llama 3](https://docs.skypilot.co/en/latest/examples/training/llama-3_1-finetuning.html), [NeMo](https://docs.skypilot.co/en/latest/examples/training/nemo.html), [Ray](https://docs.skypilot.co/en/latest/examples/training/ray.html), [Unsloth](https://docs.skypilot.co/en/latest/examples/training/unsloth.html), [Jax/TPU](https://docs.skypilot.co/en/latest/examples/training/tpu.html) |
    | Serving     | [vLLM](https://docs.skypilot.co/en/latest/examples/serving/vllm.html), [SGLang](https://docs.skypilot.co/en/latest/examples/serving/sglang.html), [Ollama](https://docs.skypilot.co/en/latest/examples/serving/ollama.html) |
    | Models      | [DeepSeek-R1](https://docs.skypilot.co/en/latest/examples/models/deepseek-r1.html), [Llama 3](https://docs.skypilot.co/en/latest/examples/models/llama-3.html), [CodeLlama](https://docs.skypilot.co/en/latest/examples/models/codellama.html), [Qwen](https://docs.skypilot.co/en/latest/examples/models/qwen.html), [Mixtral](https://docs.skypilot.co/en/latest/examples/models/mixtral.html) |
    | AI apps     | [RAG](https://docs.skypilot.co/en/latest/examples/applications/rag.html), [vector databases](https://docs.skypilot.co/en/latest/examples/applications/vector_database.html) (ChromaDB, CLIP) |
    | Frameworks  | [Airflow](https://docs.skypilot.co/en/latest/examples/frameworks/airflow.html), [Jupyter](https://docs.skypilot.co/en/latest/examples/frameworks/jupyter.html) |

*   **Example Code:** Find source files and more examples in the [`llm/`](https://github.com/skypilot-org/skypilot/tree/master/llm) and [`examples/`](https://github.com/skypilot-org/skypilot/tree/master/examples) directories.
*   **Documentation:** [SkyPilot Overview](https://docs.skypilot.co/en/latest/overview.html), [SkyPilot docs](https://docs.skypilot.co/en/latest/)
*   **Blog:** [SkyPilot blog](https://blog.skypilot.co/) and [Introductory blog post](https://blog.skypilot.co/introducing-skypilot/)
*   **Case Studies and Integrations:** [Community Spotlights](https://blog.skypilot.co/community/)

## Research and Publications

*   [SkyPilot paper](https://www.usenix.org/system/files/nsdi23-yang-zongheng.pdf) and [talk](https://www.usenix.org/conference/nsdi23/presentation/yang-zongheng) (NSDI 2023)
*   [Sky Computing whitepaper](https://arxiv.org/abs/2205.07147)
*   [Sky Computing vision paper](https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s02-stoica.pdf) (HotOS 2021)
*   [SkyServe: AI serving across regions and clouds](https://arxiv.org/pdf/2411.01438) (EuroSys 2025)
*   [Managed jobs spot instance policy](https://www.usenix.org/conference/nsdi24/presentation/wu-zhanghao)  (NSDI 2024)

SkyPilot was initially started at the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley. To understand the project's origin and vision, see [Concept: Sky Computing](https://docs.skypilot.co/en/latest/sky-computing.html).

## Get Involved

We welcome your feedback and contributions:

*   **Report Issues and Feature Requests:** [Open a GitHub issue](https://github.com/skypilot-org/skypilot/issues/new).
*   **Ask Questions:** Use [GitHub Discussions](https://github.com/skypilot-org/skypilot/discussions).
*   **Join the Community:**  Connect with us on the [SkyPilot Slack](http://slack.skypilot.co).
*   **Contribute:**  See [CONTRIBUTING](CONTRIBUTING.md) for guidance.

## Stay Connected

*   [Slack](http://slack.skypilot.co)
*   [X / Twitter](https://twitter.com/skypilot_org)
*   [LinkedIn](https://www.linkedin.com/company/skypilot-oss/)
*   [SkyPilot Blog](https://blog.skypilot.co/)