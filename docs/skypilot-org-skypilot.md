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

<h1 align="center">SkyPilot: Run AI Workloads on Any Infrastructure, Simply and Cost-Effectively</h1>

<p align="center">
  <a href="https://github.com/skypilot-org/skypilot">
      <img alt="GitHub Repo" src="https://img.shields.io/badge/GitHub-Repo-blue?logo=github">
  </a>
</p>

---

SkyPilot is your all-in-one solution for running AI and batch workloads, offering a unified platform for any infrastructure.  **SkyPilot empowers you to run your AI projects seamlessly and affordably, regardless of the underlying infrastructure.**

---

**Key Features:**

*   **Unified Infrastructure:** Run your workloads on Kubernetes, AWS, GCP, Azure, and more – all through a single interface.
*   **Cost Optimization:** Leverage spot instances for significant cost savings (3-6x) and autostop to avoid idle resource charges.
*   **Simplified Management:** Easily manage your AI jobs with features like job queuing, auto-recovery, and environment-as-code.
*   **Flexible Provisioning:** Supports GPUs, TPUs, and CPUs, with automatic resource retries and failover.
*   **Team Collaboration:**  Facilitates team deployments and resource sharing.

---

:fire: **Latest News & Examples:** :fire:

*   [Apr 2025] Spin up **Qwen3** on your cluster/cloud: [**example**](./llm/qwen/)
*   [Mar 2025] Run and serve **Google Gemma 3** using SkyPilot [**example**](./llm/gemma3/)
*   [Feb 2025] Prepare and serve **Retrieval Augmented Generation (RAG) with DeepSeek-R1**: [**blog post**](https://blog.skypilot.co/deepseek-rag), [**example**](./llm/rag/)
*   [Feb 2025] Run and serve **DeepSeek-R1 671B** using SkyPilot and SGLang with high throughput: [**example**](./llm/deepseek-r1/)
*   [Feb 2025] Prepare and serve large-scale image search with **vector databases**: [**blog post**](https://blog.skypilot.co/large-scale-vector-database/), [**example**](./examples/vector_database/)
*   [Jan 2025] Launch and serve distilled models from **[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)** and **[Janus](https://github.com/deepseek-ai/DeepSeek-Janus)** on Kubernetes or any cloud: [**R1 example**](./llm/deepseek-r1-distilled/) and [**Janus example**](./llm/deepseek-janus/)
*   [Oct 2024] :tada: **SkyPilot crossed 1M+ downloads** :tada:: Thank you to our community! [**Twitter/X**](https://x.com/skypilot_org/status/1844770841718067638)
*   [Sep 2024] Point, launch and serve **Llama 3.2** on Kubernetes or any cloud: [**example**](./llm/llama-3_2/)

**LLM Finetuning Cookbooks**: Finetuning Llama 2 / Llama 3.1 in your own cloud environment, privately: Llama 2 [**example**](./llm/vicuna-llama-2/) and [**blog**](https://blog.skypilot.co/finetuning-llama2-operational-guide/); Llama 3.1 [**example**](./llm/llama-3_1-finetuning/) and [**blog**](https://blog.skypilot.co/finetune-llama-3_1-on-your-infra/)

---

## Getting Started

Explore the [SkyPilot Documentation](https://docs.skypilot.co/) for comprehensive guides:

*   [Installation](https://docs.skypilot.co/en/latest/getting-started/installation.html)
*   [Quickstart](https://docs.skypilot.co/en/latest/getting-started/quickstart.html)
*   [CLI Reference](https://docs.skypilot.co/en/latest/reference/cli.html)

## SkyPilot in 1 Minute:

Define your task with a YAML or Python configuration, specifying resources, setup commands, and the workload itself. Launch the task, and SkyPilot takes care of the rest.  The unified interface lets you easily run your tasks on any supported cloud.

1.  **Define**:
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

2.  **Prepare**:
    ```bash
    git clone https://github.com/pytorch/examples.git ~/torch_examples
    ```

3.  **Launch**:
    ```bash
    sky launch my_task.yaml
    ```

SkyPilot will then:

*   Find the most cost-effective VM instance across different clouds.
*   Provision the VM with automatic failover.
*   Sync your local workdir.
*   Run your setup and run commands.

## Runnable Examples

Access a wide array of pre-built examples in the [SkyPilot examples](https://docs.skypilot.co/en/docs-examples/examples/index.html) to quickly develop, train, serve, and deploy AI models.

**Featured Examples:**

| Task      | Examples                                                                                                                                                                                |
| :-------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Training  | [PyTorch](https://docs.skypilot.co/en/latest/getting-started/tutorial.html), [DeepSpeed](https://docs.skypilot.co/en/latest/examples/training/deepspeed.html), [Finetune Llama 3](https://docs.skypilot.co/en/latest/examples/training/llama-3_1-finetuning.html), [NeMo](https://docs.skypilot.co/en/latest/examples/training/nemo.html), [Ray](https://docs.skypilot.co/en/latest/examples/training/ray.html), [Unsloth](https://docs.skypilot.co/en/latest/examples/training/unsloth.html), [Jax/TPU](https://docs.skypilot.co/en/latest/examples/training/tpu.html) |
| Serving   | [vLLM](https://docs.skypilot.co/en/latest/examples/serving/vllm.html), [SGLang](https://docs.skypilot.co/en/latest/examples/serving/sglang.html), [Ollama](https://docs.skypilot.co/en/latest/examples/serving/ollama.html)                                                                                                             |
| Models    | [DeepSeek-R1](https://docs.skypilot.co/en/latest/examples/models/deepseek-r1.html), [Llama 3](https://docs.skypilot.co/en/latest/examples/models/llama-3.html), [CodeLlama](https://docs.skypilot.co/en/latest/examples/models/codellama.html), [Qwen](https://docs.skypilot.co/en/latest/examples/models/qwen.html), [Mixtral](https://docs.skypilot.co/en/latest/examples/models/mixtral.html) |
| AI apps   | [RAG](https://docs.skypilot.co/en/latest/examples/applications/rag.html), [vector databases](https://docs.skypilot.co/en/latest/examples/applications/vector_database.html) (ChromaDB, CLIP)                                                                                                                                           |
| Frameworks | [Airflow](https://docs.skypilot.co/en/latest/examples/frameworks/airflow.html), [Jupyter](https://docs.skypilot.co/en/latest/examples/frameworks/jupyter.html)                                                                                                                                                                         |

Explore more examples in the [`llm/`](https://github.com/skypilot-org/skypilot/tree/master/llm) and [`examples/`](https://github.com/skypilot-org/skypilot/tree/master/examples) directories.

## Supported Infrastructure

SkyPilot supports a wide range of infrastructure providers, including: Kubernetes, AWS, GCP, Azure, OCI, Lambda Cloud, Fluidstack, RunPod, Cudo, Digital Ocean, Paperspace, Cloudflare, Samsung, IBM, Vast.ai, VMware vSphere, and Nebius.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-dark.png">
    <img alt="SkyPilot" src="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-light.png" width=85%>
  </picture>
</p>

## Installation

Install SkyPilot using pip:

```bash
pip install -U "skypilot[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```

To access the latest features and fixes, use the nightly build or install from source:

```bash
pip install "skypilot-nightly[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```

## Learn More

*   [SkyPilot Overview](https://docs.skypilot.co/en/latest/overview.html)
*   [SkyPilot Documentation](https://docs.skypilot.co/en/latest/)
*   [SkyPilot Blog](https://blog.skypilot.co/)

**Case Studies & Integrations:** [Community Spotlights](https://blog.skypilot.co/community/)

**Stay Updated:**

*   [Slack](http://slack.skypilot.co)
*   [X / Twitter](https://twitter.com/skypilot_org)
*   [LinkedIn](https://www.linkedin.com/company/skypilot-oss/)
*   [SkyPilot Blog](https://blog.skypilot.co/)
    ([Introductory blog post](https://blog.skypilot.co/introducing-skypilot/))

**Read the Research:**

*   [SkyPilot paper](https://www.usenix.org/system/files/nsdi23-yang-zongheng.pdf) and [talk](https://www.usenix.org/conference/nsdi23/presentation/yang-zongheng) (NSDI 2023)
*   [Sky Computing whitepaper](https://arxiv.org/abs/2205.07147)
*   [Sky Computing vision paper](https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s02-stoica.pdf) (HotOS 2021)
*   [SkyServe: AI serving across regions and clouds](https://arxiv.org/pdf/2411.01438) (EuroSys 2025)
*   [Managed jobs spot instance policy](https://www.usenix.org/conference/nsdi24/presentation/wu-zhanghao) (NSDI 2024)

SkyPilot was initially developed at the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley. Learn about the project's origins in [Concept: Sky Computing](https://docs.skypilot.co/en/latest/sky-computing.html).

## Get Involved

We value your feedback and contributions!

*   **Issues & Feature Requests:** [Open a GitHub issue](https://github.com/skypilot-org/skypilot/issues/new).
*   **Questions:** Use [GitHub Discussions](https://github.com/skypilot-org/skypilot/discussions).
*   **General Discussions:** Join us on the [SkyPilot Slack](http://slack.skypilot.co).

**Contributing:** See [CONTRIBUTING](CONTRIBUTING.md) for details on how to contribute.

**[Back to Top](#skypilot-run-ai-workloads-on-any-infrastructure-simply-and-cost-effectively)**
```
Key improvements and optimizations:

*   **SEO-Friendly Title:**  Added a strong, keyword-rich H1 to immediately grab attention and improve search ranking.
*   **Concise Hook:**  A one-sentence summary at the beginning emphasizes the core benefit (running AI projects easily and affordably).
*   **Clear Headings & Structure:**  Organized information into clear, scannable sections with headings (Getting Started, Key Features, etc.).
*   **Bulleted Key Features:**  Used bullet points to highlight the main advantages of using SkyPilot, making it easy to digest.
*   **Keyword Optimization:**  Incorporated relevant keywords naturally throughout the content (e.g., "AI workloads," "infrastructure," "cost optimization," "spot instances,"  "Kubernetes").
*   **Stronger Call to Action:** Encouraged exploration of the documentation and examples.
*   **Updated News & Examples:** Kept the "Latest News" section, making it easier for users to quickly find relevant information.
*   **Clearer "1-Minute" Example:** Improved the code example presentation with formatted code and clearer explanations.
*   **Back to Top Link** added to the end for easier navigation.
*   **Concise Summaries:**  Condensed the text while retaining essential information.
*   **Improved Visuals:**  Made sure the images and shields are presented well.
*   **Complete Coverage:** Included all information from the original README while improving readability and SEO.