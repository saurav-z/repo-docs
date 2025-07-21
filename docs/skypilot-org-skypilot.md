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

<h2 align="center">SkyPilot: Run AI Workloads on Any Infrastructure, Faster and Cheaper</h2>

---

### **Key Features**

*   🚀 **Unified Infrastructure:** Run your AI workloads across multiple clouds (AWS, GCP, Azure, and more) and on-premise resources with a single interface.
*   💰 **Cost Optimization:** Leverage spot instances for significant cost savings (3-6x) and automatic resource scaling.
*   🛠️ **Simplified Deployment:** Easily define your environment and jobs as code, simplifying portability and management.
*   🌐 **Global Availability:** Deploy and serve AI models in any region and cloud.
*   🤖 **Automated Management:** Benefit from auto-retry, intelligent scheduling, and automated resource cleanup.
*   🔄 **Flexible Resource Provisioning:** SkyPilot supports GPUs, TPUs, and CPUs, with auto-retry and flexible configuration.

---

### **What's New**

*   [Apr 2025] Run **Qwen3** on your cluster/cloud: [**example**](./llm/qwen/)
*   [Mar 2025] Run and serve **Google Gemma 3** using SkyPilot [**example**](./llm/gemma3/)
*   [Feb 2025] Prepare and serve **Retrieval Augmented Generation (RAG) with DeepSeek-R1**: [**blog post**](https://blog.skypilot.co/deepseek-rag), [**example**](./llm/rag/)
*   [Feb 2025] Run and serve **DeepSeek-R1 671B** using SkyPilot and SGLang with high throughput: [**example**](./llm/deepseek-r1/)
*   [Feb 2025] Prepare and serve large-scale image search with **vector databases**: [**blog post**](https://blog.skypilot.co/large-scale-vector-database/), [**example**](./examples/vector_database/)
*   [Jan 2025] Launch and serve distilled models from **[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)** and **[Janus](https://github.com/deepseek-ai/DeepSeek-Janus)** on Kubernetes or any cloud: [**R1 example**](./llm/deepseek-r1-distilled/) and [**Janus example**](./llm/deepseek-janus/)
*   [Oct 2024] :tada: **SkyPilot crossed 1M+ downloads** :tada:: Thank you to our community! [**Twitter/X**](https://x.com/skypilot_org/status/1844770841718067638)
*   [Sep 2024] Point, launch and serve **Llama 3.2** on Kubernetes or any cloud: [**example**](./llm/llama-3_2/)

**LLM Finetuning Cookbooks**: Finetuning Llama 2 / Llama 3.1 in your own cloud environment, privately: Llama 2 [**example**](./llm/vicuna-llama-2/) and [**blog**](https://blog.skypilot.co/finetuning-llama2-operational-guide/); Llama 3.1 [**example**](./llm/llama-3_1-finetuning/) and [**blog**](https://blog.skypilot.co/finetune-llama-3_1-on-your-infra/)

---

### **About SkyPilot**

SkyPilot is an open-source framework designed to simplify and optimize the execution of AI and batch workloads on diverse infrastructures. Whether you're training large language models, running AI applications, or managing complex batch jobs, SkyPilot provides a unified, cost-effective, and user-friendly solution.

SkyPilot helps you:

*   **Accelerate AI Development:** Quickly spin up compute resources on your preferred infrastructure.
*   **Increase Efficiency:** Automate resource management, including auto-scaling and automatic cleanup.
*   **Reduce Costs:** Optimize cloud spending through spot instance utilization and intelligent scheduling.
*   **Simplify Deployment:** Define your jobs and environments as code for improved portability and manageability.

### **Supported Infrastructure**

SkyPilot supports a wide range of infrastructure providers, including:

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
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-dark.png">
    <img alt="SkyPilot" src="https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/cloud-logos-light.png" width=85%>
  </picture>
</p>

### **Getting Started**

**Installation:**

```bash
# Choose your clouds:
pip install -U "skypilot[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```

For the latest features and fixes, use the nightly build:

```bash
# Choose your clouds:
pip install "skypilot-nightly[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```

**Documentation:** Comprehensive documentation is available [here](https://docs.skypilot.co/).

*   [Installation](https://docs.skypilot.co/en/latest/getting-started/installation.html)
*   [Quickstart](https://docs.skypilot.co/en/latest/getting-started/quickstart.html)
*   [CLI Reference](https://docs.skypilot.co/en/latest/reference/cli.html)

### **SkyPilot in 1 Minute: Example**

SkyPilot allows you to define your AI tasks using a simple YAML or Python API.  This example demonstrates how to launch a PyTorch training job on an A100 GPU:

**1. Create a task definition file (e.g., `my_task.yaml`):**

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

**2. Prepare your working directory:**

```bash
git clone https://github.com/pytorch/examples.git ~/torch_examples
```

**3. Launch the task using the `sky launch` command:**

```bash
sky launch my_task.yaml
```

SkyPilot will then:

1.  Find the most cost-effective VM instance.
2.  Provision the VM.
3.  Sync your `workdir`.
4.  Run the setup and run commands.

See the [Quickstart](https://docs.skypilot.co/en/latest/getting-started/quickstart.html) for a detailed tutorial.

### **Runnable Examples**

Explore our comprehensive collection of [SkyPilot examples](https://docs.skypilot.co/en/docs-examples/examples/index.html) for development, training, serving, LLM models, AI apps, and framework integration.

Featured examples:

| Task          | Examples                                                                                                                                                                                                                                                                     |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Training      | [PyTorch](https://docs.skypilot.co/en/latest/getting-started/tutorial.html), [DeepSpeed](https://docs.skypilot.co/en/latest/examples/training/deepspeed.html), [Finetune Llama 3](https://docs.skypilot.co/en/latest/examples/training/llama-3_1-finetuning.html), [NeMo](https://docs.skypilot.co/en/latest/examples/training/nemo.html), [Ray](https://docs.skypilot.co/en/latest/examples/training/ray.html), [Unsloth](https://docs.skypilot.co/en/latest/examples/training/unsloth.html), [Jax/TPU](https://docs.skypilot.co/en/latest/examples/training/tpu.html) |
| Serving       | [vLLM](https://docs.skypilot.co/en/latest/examples/serving/vllm.html), [SGLang](https://docs.skypilot.co/en/latest/examples/serving/sglang.html), [Ollama](https://docs.skypilot.co/en/latest/examples/serving/ollama.html) |
| Models        | [DeepSeek-R1](https://docs.skypilot.co/en/latest/examples/models/deepseek-r1.html), [Llama 3](https://docs.skypilot.co/en/latest/examples/models/llama-3.html), [CodeLlama](https://docs.skypilot.co/en/latest/examples/models/codellama.html), [Qwen](https://docs.skypilot.co/en/latest/examples/models/qwen.html), [Mixtral](https://docs.skypilot.co/en/latest/examples/models/mixtral.html) |
| AI apps       | [RAG](https://docs.skypilot.co/en/latest/examples/applications/rag.html), [vector databases](https://docs.skypilot.co/en/latest/examples/applications/vector_database.html) (ChromaDB, CLIP)                                                                  |
| Common frameworks | [Airflow](https://docs.skypilot.co/en/latest/examples/frameworks/airflow.html), [Jupyter](https://docs.skypilot.co/en/latest/examples/frameworks/jupyter.html)                                                                                                   |

Source files and more examples can be found in [`llm/`](https://github.com/skypilot-org/skypilot/tree/master/llm) and [`examples/`](https://github.com/skypilot-org/skypilot/tree/master/examples).

### **More Information**

*   [SkyPilot Overview](https://docs.skypilot.co/en/latest/overview.html)
*   [SkyPilot Documentation](https://docs.skypilot.co/en/latest/)
*   [SkyPilot Blog](https://blog.skypilot.co/) ([Introductory blog post](https://blog.skypilot.co/introducing-skypilot/))
*   [Concept: Sky Computing](https://docs.skypilot.co/en/latest/sky-computing.html)

**Case studies and integrations:** [Community Spotlights](https://blog.skypilot.co/community/)

**Follow updates:**

*   [Slack](http://slack.skypilot.co)
*   [X / Twitter](https://twitter.com/skypilot_org)
*   [LinkedIn](https://www.linkedin.com/company/skypilot-oss/)
*   [SkyPilot Blog](https://blog.skypilot.co/)

**Read the research:**

*   [SkyPilot paper](https://www.usenix.org/system/files/nsdi23-yang-zongheng.pdf) and [talk](https://www.usenix.org/conference/nsdi23/presentation/yang-zongheng) (NSDI 2023)
*   [Sky Computing whitepaper](https://arxiv.org/abs/2205.07147)
*   [Sky Computing vision paper](https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s02-stoica.pdf) (HotOS 2021)
*   [SkyServe: AI serving across regions and clouds](https://arxiv.org/pdf/2411.01438) (EuroSys 2025)
*   [Managed jobs spot instance policy](https://www.usenix.org/conference/nsdi24/presentation/wu-zhanghao)  (NSDI 2024)

### **Community & Contributions**

SkyPilot was initially started at the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley. We welcome contributions from the community!  See [CONTRIBUTING](CONTRIBUTING.md) for guidelines on how to get involved.

### **Get in Touch**

*   For issues and feature requests, please [open a GitHub issue](https://github.com/skypilot-org/skypilot/issues/new).
*   For questions, please use [GitHub Discussions](https://github.com/skypilot-org/skypilot/discussions).

Join our community:

*   [SkyPilot Slack](http://slack.skypilot.co)

---

**[Back to Top](#top)**