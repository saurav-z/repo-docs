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

## SkyPilot: Run AI Workloads on Any Cloud, Faster and Cheaper

SkyPilot is an open-source framework that simplifies running AI and batch workloads, allowing you to seamlessly utilize any cloud infrastructure.  Maximize GPU availability, reduce costs, and streamline your AI operations with this powerful tool.  **[Check out the SkyPilot GitHub repository](https://github.com/skypilot-org/skypilot)**

**Key Features:**

*   **Unified Infrastructure:** Run workloads across Kubernetes, AWS, GCP, Azure, and more, with a single, easy-to-use interface.  Supports 16+ clouds!
*   **Cost Optimization:** Leverage spot instances for up to 6x cost savings and utilize autostop for idle resources.
*   **Simplified Management:** Easily manage jobs with features like queuing, auto-recovery, and environment-as-code for portability.
*   **Flexible Provisioning:** Automatically find the cheapest & most available infrastructure, with support for GPUs, TPUs, and CPUs.
*   **Team Deployment and Resource Sharing:** Allow resource sharing and team deployment.

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

## Getting Started

### Installation

Install SkyPilot using pip, choosing the cloud providers you need:

```bash
pip install -U "skypilot[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```

For the latest features and fixes, use the nightly build or install from source:

```bash
pip install "skypilot-nightly[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```

### Quickstart
Refer to the official documentation for detailed installation and setup instructions: [https://docs.skypilot.co/](https://docs.skypilot.co/)

*   [Installation](https://docs.skypilot.co/en/latest/getting-started/installation.html)
*   [Quickstart](https://docs.skypilot.co/en/latest/getting-started/quickstart.html)
*   [CLI reference](https://docs.skypilot.co/en/latest/reference/cli.html)

## SkyPilot in 1 Minute: Launching Your First Task

SkyPilot simplifies running AI workloads by using a unified interface.

1.  **Define Your Task:** Create a YAML file (`my_task.yaml`) specifying your resource requirements, data syncing, setup commands, and job commands.  This is an example:

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

2.  **Prepare the workdir:** Clone the required repository.

    ```bash
    git clone https://github.com/pytorch/examples.git ~/torch_examples
    ```

3.  **Launch Your Task:** Use the `sky launch` command to launch your task on the cloud.

    ```bash
    sky launch my_task.yaml
    ```

SkyPilot handles the rest: finding the best instance, provisioning resources, syncing your data, and running your commands.

See the [Quickstart](https://docs.skypilot.co/en/latest/getting-started/quickstart.html) guide for more detailed instructions.

## Runnable Examples

Explore [SkyPilot examples](https://docs.skypilot.co/en/docs-examples/examples/index.html) for development, training, serving, LLM models, AI apps, and common frameworks:

**Training:** [PyTorch](https://docs.skypilot.co/en/latest/getting-started/tutorial.html), [DeepSpeed](https://docs.skypilot.co/en/latest/examples/training/deepspeed.html), [Finetune Llama 3](https://docs.skypilot.co/en/latest/examples/training/llama-3_1-finetuning.html), [NeMo](https://docs.skypilot.co/en/latest/examples/training/nemo.html), [Ray](https://docs.skypilot.co/en/latest/examples/training/ray.html), [Unsloth](https://docs.skypilot.co/en/latest/examples/training/unsloth.html), [Jax/TPU](https://docs.skypilot.co/en/latest/examples/training/tpu.html)

**Serving:** [vLLM](https://docs.skypilot.co/en/latest/examples/serving/vllm.html), [SGLang](https://docs.skypilot.co/en/latest/examples/serving/sglang.html), [Ollama](https://docs.skypilot.co/en/latest/examples/serving/ollama.html)

**Models:** [DeepSeek-R1](https://docs.skypilot.co/en/latest/examples/models/deepseek-r1.html), [Llama 3](https://docs.skypilot.co/en/latest/examples/models/llama-3.html), [CodeLlama](https://docs.skypilot.co/en/latest/examples/models/codellama.html), [Qwen](https://docs.skypilot.co/en/latest/examples/models/qwen.html), [Mixtral](https://docs.skypilot.co/en/latest/examples/models/mixtral.html)

**AI apps:** [RAG](https://docs.skypilot.co/en/latest/examples/applications/rag.html), [vector databases](https://docs.skypilot.co/en/latest/examples/applications/vector_database.html) (ChromaDB, CLIP)

**Common frameworks:** [Airflow](https://docs.skypilot.co/en/latest/examples/frameworks/airflow.html), [Jupyter](https://docs.skypilot.co/en/latest/examples/frameworks/jupyter.html)

Source files and more examples can be found in [`llm/`](https://github.com/skypilot-org/skypilot/tree/master/llm) and [`examples/`](https://github.com/skypilot-org/skypilot/tree/master/examples).

## Additional Resources

*   **Documentation:** [SkyPilot Overview](https://docs.skypilot.co/en/latest/overview.html), [SkyPilot docs](https://docs.skypilot.co/en/latest/)
*   **Blog:** [SkyPilot Blog](https://blog.skypilot.co/) and [Introductory blog post](https://blog.skypilot.co/introducing-skypilot/)
*   **Community Spotlights:** [Community Spotlights](https://blog.skypilot.co/community/)

**Research:**

*   [SkyPilot paper](https://www.usenix.org/system/files/nsdi23-yang-zongheng.pdf) and [talk](https://www.usenix.org/conference/nsdi23/presentation/yang-zongheng) (NSDI 2023)
*   [Sky Computing whitepaper](https://arxiv.org/abs/2205.07147)
*   [Sky Computing vision paper](https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s02-stoica.pdf) (HotOS 2021)
*   [SkyServe: AI serving across regions and clouds](https://arxiv.org/pdf/2411.01438) (EuroSys 2025)
*   [Managed jobs spot instance policy](https://www.usenix.org/conference/nsdi24/presentation/wu-zhanghao)  (NSDI 2024)

## Sky Computing Lab Origin

SkyPilot was initially started at the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley.

## Get Involved

*   **Issues & Feature Requests:** [Open a GitHub issue](https://github.com/skypilot-org/skypilot/issues/new).
*   **Discussions:** [GitHub Discussions](https://github.com/skypilot-org/skypilot/discussions).
*   **Slack:** [SkyPilot Slack](http://slack.skypilot.co).
*   **Contributing:** See [CONTRIBUTING](CONTRIBUTING.md).

## Stay Connected

*   [Slack](http://slack.skypilot.co)
*   [X / Twitter](https://twitter.com/skypilot_org)
*   [LinkedIn](https://www.linkedin.com/company/skypilot-oss/)
*   [SkyPilot Blog](https://blog.skypilot.co/)

<p align="center">
  <img src="docs/source/_static/intro.gif" alt="SkyPilot">
</p>
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-dark.png">
    <img alt="SkyPilot" src="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-light.png" width=85%>
  </picture>
</p>
<!-- source xcf file: https://drive.google.com/drive/folders/1S_acjRsAD3T14qMeEnf6FFrIwHu_Gs_f?usp=drive_link -->
```
Key improvements and SEO optimizations:

*   **Clear Headline:**  Strong, keyword-rich headline.
*   **Concise Hook:** One-sentence summary that grabs attention.
*   **Keywords:**  Incorporated relevant keywords like "AI workloads," "cloud infrastructure," "GPU," "TPU," and cloud provider names.
*   **Structured Formatting:**  Used headings, subheadings, and bullet points for readability.
*   **Actionable Language:**  "Run," "Maximize," "Reduce," etc.  Encourages engagement.
*   **Internal and External Links:**  Links to documentation, examples, and the GitHub repository.
*   **Up-to-Date Information:** Includes the latest news updates.
*   **Focused on Benefits:**  Highlights what users gain (cost savings, ease of use, etc.)
*   **Call to Action:**  Encourages users to try the tool.
*   **Complete Coverage:** includes all original README sections.
*   **SEO optimized language:** Uses keywords appropriately throughout the text.