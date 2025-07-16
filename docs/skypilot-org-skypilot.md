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

<h1 align="center">SkyPilot: Run AI Workloads on Any Infrastructure</h1>

<p align="center">
  <b>SkyPilot simplifies running AI workloads by enabling you to run your code on any cloud, cluster, or hardware, faster and cheaper.</b>
</p>

----

:fire: *What's New* :fire:

*   [Apr 2025] Spin up **Qwen3** on your cluster/cloud: [**example**](./llm/qwen/)
*   [Mar 2025] Run and serve **Google Gemma 3** using SkyPilot [**example**](./llm/gemma3/)
*   [Feb 2025] Prepare and serve **Retrieval Augmented Generation (RAG) with DeepSeek-R1**: [**blog post**](https://blog.skypilot.co/deepseek-rag), [**example**](./llm/rag/)
*   [Feb 2025] Run and serve **DeepSeek-R1 671B** using SkyPilot and SGLang with high throughput: [**example**](./llm/deepseek-r1/)
*   [Feb 2025] Prepare and serve large-scale image search with **vector databases**: [**blog post**](https://blog.skypilot.co/large-scale-vector-database/), [**example**](./examples/vector_database/)
*   [Jan 2025] Launch and serve distilled models from **[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)** and **[Janus](https://github.com/deepseek-ai/DeepSeek-Janus)** on Kubernetes or any cloud: [**R1 example**](./llm/deepseek-r1-distilled/) and [**Janus example**](./llm/deepseek-janus/)
*   [Oct 2024] :tada: **SkyPilot crossed 1M+ downloads** :tada:: Thank you to our community! [**Twitter/X**](https://x.com/skypilot_org/status/1844770841718067638)
*   [Sep 2024] Point, launch and serve **Llama 3.2** on Kubernetes or any cloud: [**example**](./llm/llama-3_2/)

**LLM Finetuning Cookbooks**: Finetuning Llama 2 / Llama 3.1 in your own cloud environment, privately: Llama 2 [**example**](./llm/vicuna-llama-2/) and [**blog**](https://blog.skypilot.co/finetuning-llama2-operational-guide/); Llama 3.1 [**example**](./llm/llama-3_1-finetuning/) and [**blog**](https://blog.skypilot.co/finetune-llama-3_1-on-your-infra/)

----

## Key Features

*   **Unified Infrastructure Management:** Run AI workloads across diverse infrastructure, including public clouds, Kubernetes, and on-premise resources.
*   **Simplified Deployment:** Easily define and deploy AI tasks using a simple YAML or Python API.
*   **Cost Optimization:**  Utilize spot instances and intelligent scheduling to minimize cloud costs.
*   **Automated Resource Management:** SkyPilot handles provisioning, scaling, and auto-recovery, simplifying operations.
*   **Vendor Agnostic:** Avoid vendor lock-in by seamlessly switching between cloud providers.
*   **Broad Infrastructure Support:** Compatible with Kubernetes, AWS, GCP, Azure, and more (see supported infra below).
*   **Accelerated Development:** Quickly spin up compute and manage your AI jobs with ease.

## Why Choose SkyPilot?

SkyPilot is an open-source framework designed to streamline and optimize your AI and batch workloads:

*   **Ease of Use:** Get up and running quickly with a simple, intuitive interface.
*   **Portability:**  Move your workloads between different infrastructures without code changes.
*   **Cost-Effectiveness:** Reduce cloud expenses by leveraging spot instances and automated resource management.
*   **Flexibility:** Choose the best infrastructure for your needs, from public clouds to on-premise resources.

## Getting Started

### Installation

Install SkyPilot using pip:

```bash
# Choose your clouds:
pip install -U "skypilot[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```

For the latest features and fixes, consider the nightly build or [install from source](https://docs.skypilot.co/en/latest/getting-started/installation.html):

```bash
# Choose your clouds:
pip install "skypilot-nightly[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```

### Quickstart

1.  **Define Your Task:** Create a YAML file (e.g., `my_task.yaml`) specifying resources, setup commands, and the job to run.

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

2.  **Prepare the workdir by cloning:**
    ```bash
    git clone https://github.com/pytorch/examples.git ~/torch_examples
    ```

3.  **Launch the Task:** Run `sky launch my_task.yaml`. SkyPilot automatically handles provisioning, syncing, and execution on the most suitable infrastructure.

See the [Quickstart](https://docs.skypilot.co/en/latest/getting-started/quickstart.html) for a more detailed guide.

## SkyPilot in Action: 1-Minute Tutorial

SkyPilot uses a [**unified interface**](https://docs.skypilot.co/en/latest/reference/yaml-spec.html) (YAML or Python API) to launch tasks, making it easy to run your tasks across any supported cloud provider.

### Launching the Task

```bash
sky launch my_task.yaml
```

SkyPilot handles the heavy-lifting, including:

1.  Finding the lowest priced VM instance type across different clouds
2.  Provisioning the VM, with auto-failover if the cloud returned capacity errors
3.  Syncing your local `workdir` to the VM
4.  Running the task's `setup` commands to prepare the VM for running the task
5.  Running the task's `run` commands

## Examples and Tutorials

Explore a wide range of examples showcasing SkyPilot's capabilities:  [**SkyPilot examples**](https://docs.skypilot.co/en/docs-examples/examples/index.html).

**Featured Examples:**

| Task        | Examples                                                                                                                                           |
|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| Training    | [PyTorch](https://docs.skypilot.co/en/latest/getting-started/tutorial.html), [DeepSpeed](https://docs.skypilot.co/en/latest/examples/training/deepspeed.html), [Finetune Llama 3](https://docs.skypilot.co/en/latest/examples/training/llama-3_1-finetuning.html), [NeMo](https://docs.skypilot.co/en/latest/examples/training/nemo.html), [Ray](https://docs.skypilot.co/en/latest/examples/training/ray.html), [Unsloth](https://docs.skypilot.co/en/latest/examples/training/unsloth.html), [Jax/TPU](https://docs.skypilot.co/en/latest/examples/training/tpu.html) |
| Serving     | [vLLM](https://docs.skypilot.co/en/latest/examples/serving/vllm.html), [SGLang](https://docs.skypilot.co/en/latest/examples/serving/sglang.html), [Ollama](https://docs.skypilot.co/en/latest/examples/serving/ollama.html) |
| Models      | [DeepSeek-R1](https://docs.skypilot.co/en/latest/examples/models/deepseek-r1.html), [Llama 3](https://docs.skypilot.co/en/latest/examples/models/llama-3.html), [CodeLlama](https://docs.skypilot.co/en/latest/examples/models/codellama.html), [Qwen](https://docs.skypilot.co/en/latest/examples/models/qwen.html), [Mixtral](https://docs.skypilot.co/en/latest/examples/models/mixtral.html) |
| AI apps     | [RAG](https://docs.skypilot.co/en/latest/examples/applications/rag.html), [vector databases](https://docs.skypilot.co/en/latest/examples/applications/vector_database.html) (ChromaDB, CLIP) |
| Frameworks  | [Airflow](https://docs.skypilot.co/en/latest/examples/frameworks/airflow.html), [Jupyter](https://docs.skypilot.co/en/latest/examples/frameworks/jupyter.html) |

Source files and more examples can be found in [`llm/`](https://github.com/skypilot-org/skypilot/tree/master/llm) and [`examples/`](https://github.com/skypilot-org/skypilot/tree/master/examples).

## Supported Infrastructure

SkyPilot supports a wide range of infrastructure providers:

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
  <img src="docs/source/_static/intro.gif" alt="SkyPilot">
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-dark.png">
    <img alt="SkyPilot" src="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-light.png" width=85%>
  </picture>
</p>

## Resources and Documentation

*   **[SkyPilot Documentation](https://docs.skypilot.co/en/latest/)**: Comprehensive documentation, tutorials, and API reference.
*   **[SkyPilot Overview](https://docs.skypilot.co/en/latest/overview.html)**: Get a high-level understanding of SkyPilot's capabilities.
*   **[SkyPilot Blog](https://blog.skypilot.co/)**: Stay up-to-date with the latest news, case studies, and community spotlights.
*   **[SkyPilot GitHub Repository](https://github.com/skypilot-org/skypilot)**: Explore the source code, contribute, and report issues.

## Community and Support

*   **[Join the SkyPilot Slack](http://slack.skypilot.co)**: Connect with other users and the SkyPilot community.
*   **[GitHub Discussions](https://github.com/skypilot-org/skypilot/discussions)**: Ask questions and engage in discussions.
*   **[Open a GitHub Issue](https://github.com/skypilot-org/skypilot/issues/new)**: Report bugs, request features, and provide feedback.
*   **[X / Twitter](https://twitter.com/skypilot_org)**: Follow for updates.
*   **[LinkedIn](https://www.linkedin.com/company/skypilot-oss/)**
*   **[SkyPilot Blog](https://blog.skypilot.co/)** ([Introductory blog post](https://blog.skypilot.co/introducing-skypilot/))

## Research

Learn more about the underlying principles of SkyPilot:
*   [SkyPilot paper](https://www.usenix.org/system/files/nsdi23-yang-zongheng.pdf) and [talk](https://www.usenix.org/conference/nsdi23/presentation/yang-zongheng) (NSDI 2023)
*   [Sky Computing whitepaper](https://arxiv.org/abs/2205.07147)
*   [Sky Computing vision paper](https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s02-stoica.pdf) (HotOS 2021)
*   [SkyServe: AI serving across regions and clouds](https://arxiv.org/pdf/2411.01438) (EuroSys 2025)
*   [Managed jobs spot instance policy](https://www.usenix.org/conference/nsdi24/presentation/wu-zhanghao)  (NSDI 2024)

SkyPilot was initially started at the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley and has since gained many industry contributors. To read about the project's origin and vision, see [Concept: Sky Computing](https://docs.skypilot.co/en/latest/sky-computing.html).

## Contributing

We welcome contributions!  See the [CONTRIBUTING](CONTRIBUTING.md) guidelines for more details on how to contribute.

**[Back to Top](#)  [View the SkyPilot Repository](https://github.com/skypilot-org/skypilot)**