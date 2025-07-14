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

<h1 align="center">SkyPilot: Run AI on Any Infrastructure, Easily and Affordably</h1>

SkyPilot is your open-source key to unlocking seamless AI workload execution across any infrastructure, including multiple clouds, Kubernetes clusters, and on-premise resources, all while optimizing costs and maximizing availability.  [**Check out the SkyPilot repo for more details**](https://github.com/skypilot-org/skypilot).

---

## Key Features

*   **Unified Infrastructure Management:** Run your AI workloads (training, serving, batch jobs) on **16+ clouds**, Kubernetes, and on-premise resources through a single interface.
*   **Cost Optimization:** Automatically find the cheapest and most available infrastructure using spot instances for **3-6x cost savings** and intelligent scheduling.
*   **Simplified Deployment:**  Define your jobs with environment and job-as-code, which makes your workloads portable and easy to manage.
*   **Auto-Scaling and Reliability:** SkyPilot automatically manages cloud resources, with auto-retry for failed provisioning.
*   **Flexible Resource Provisioning:** Supports GPUs, TPUs, and CPUs, with automatic retry and failover.
*   **Team Collaboration:** Enables resource sharing and team deployment.

---

### What's New?
Stay up-to-date with the latest advancements in SkyPilot:

*   **Run and serve Large Language Models (LLMs):**
    *   [Qwen3](llm/qwen/)
    *   [Google Gemma 3](llm/gemma3/)
    *   [DeepSeek-R1](llm/deepseek-r1/)
    *   [DeepSeek-R1 671B](llm/deepseek-r1/)
    *   [Retrieval Augmented Generation (RAG) with DeepSeek-R1](llm/rag/)
    *   [Janus](llm/deepseek-janus/)
*   **Enhanced Vector Database Support:**
    *   Large-scale image search with vector databases
*   **Milestone:** SkyPilot has surpassed **1M+ downloads**!
*   **Latest Models:**
    *   Llama 3.2
*   **LLM Finetuning Cookbooks:** Finetuning Llama 2 / Llama 3.1 in your own cloud environment, privately.

    *   Llama 2 [**example**](./llm/vicuna-llama-2/) and [**blog**](https://blog.skypilot.co/finetuning-llama2-operational-guide/)
    *   Llama 3.1 [**example**](./llm/llama-3_1-finetuning/) and [**blog**](https://blog.skypilot.co/finetune-llama-3_1-on-your-infra/)

---

## Get Started

SkyPilot simplifies running your AI workloads. Here's a quick example:

1.  **Install:**

    ```bash
    pip install -U "skypilot[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
    ```
    Or, for the latest features:
    ```bash
    pip install "skypilot-nightly[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
    ```

2.  **Define Your Task (YAML):**

    ```yaml
    resources:
      accelerators: A100:8  # 8x NVIDIA A100 GPU

    num_nodes: 1  # Number of VMs to launch

    # Working directory containing the project codebase.
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

3.  **Prepare the workdir by cloning:**

    ```bash
    git clone https://github.com/pytorch/examples.git ~/torch_examples
    ```

4.  **Launch:**

    ```bash
    sky launch my_task.yaml
    ```

SkyPilot handles the rest: finding the best resources, provisioning, syncing data, and running your job.

---

## Examples & Documentation

*   **Comprehensive Examples:** Explore [**SkyPilot examples**](https://docs.skypilot.co/en/docs-examples/examples/index.html) for training, serving, and AI applications.
*   **Detailed Documentation:**  Refer to the [SkyPilot Documentation](https://docs.skypilot.co/en/latest/) for in-depth guides, installation, and API references.

---

## Learn More

*   **SkyPilot Overview:** [SkyPilot Overview](https://docs.skypilot.co/en/latest/overview.html)
*   **SkyPilot Docs:** [SkyPilot docs](https://docs.skypilot.co/en/latest/)
*   **SkyPilot Blog:** [SkyPilot blog](https://blog.skypilot.co/)

**Stay Connected:**

*   [Slack](http://slack.skypilot.co)
*   [X / Twitter](https://twitter.com/skypilot_org)
*   [LinkedIn](https://www.linkedin.com/company/skypilot-oss/)
*   [SkyPilot Blog](https://blog.skypilot.co/) ([Introductory blog post](https://blog.skypilot.co/introducing-skypilot/))
*   **Research Papers:**  Access related research papers on Sky Computing and SkyPilot:

    *   [SkyPilot paper](https://www.usenix.org/system/files/nsdi23-yang-zongheng.pdf) and [talk](https://www.usenix.org/conference/nsdi23/presentation/yang-zongheng) (NSDI 2023)
    *   [Sky Computing whitepaper](https://arxiv.org/abs/2205.07147)
    *   [Sky Computing vision paper](https://sigops.org/s/conferences/hotos/2021/papers/hotos21-stoica.pdf) (HotOS 2021)
    *   [SkyServe: AI serving across regions and clouds](https://arxiv.org/pdf/2411.01438) (EuroSys 2025)
    *   [Managed jobs spot instance policy](https://www.usenix.org/conference/nsdi24/presentation/wu-zhanghao)  (NSDI 2024)

---

## Community & Contribution

*   **Join the Community:**  [SkyPilot Slack](http://slack.skypilot.co) for discussions.
*   **Contribute:** Review the [CONTRIBUTING](CONTRIBUTING.md) guide.
*   **Report Issues:** [Open a GitHub issue](https://github.com/skypilot-org/skypilot/issues/new) for bug reports and feature requests.
*   **Ask Questions:** Use [GitHub Discussions](https://github.com/skypilot-org/skypilot/discussions) for general questions.

---

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
```
Key improvements and SEO considerations:

*   **Concise Hook:**  Starts with a strong one-sentence hook to grab attention and clearly state SkyPilot's value proposition.
*   **Clear Headings:** Uses H1 and H2 headings for better organization and readability.
*   **Keyword Optimization:**  Includes relevant keywords like "AI," "infrastructure," "cloud," "Kubernetes," "cost optimization," and names of models and frameworks that SkyPilot supports.
*   **Bulleted Lists:**  Employs bullet points for key features, making information easy to scan.
*   **Actionable Call to Action:** Encourages users to explore the documentation and examples.
*   **Emphasis on Benefits:** Highlights the benefits of using SkyPilot (cost savings, ease of use, flexibility).
*   **Up-to-date Information:**  Includes the latest news and examples, keeping the README current.
*   **Clear Installation Instructions:**  Provides concise installation instructions.
*   **Links:** Includes links to documentation, examples, the SkyPilot Slack, and the original GitHub repository.
*   **Concise Summary:**  Summarizes the key features and benefits of SkyPilot in an easily digestible format.
*   **Multiple Call-to-Actions:**  Directs users towards different sections such as examples, docs, community, etc., to increase engagement.
*   **Visual Appeal:**  Retains the original images and maintains visual consistency.
*   **Added "What's New" Section** To keep users updated about the latest features.