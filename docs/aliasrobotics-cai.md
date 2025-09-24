# Cybersecurity AI (`CAI`): Revolutionizing Security with AI

**Cybersecurity AI (CAI) is an open-source framework empowering security professionals to build and deploy AI-powered offensive and defensive automation, transforming the landscape of cybersecurity.**  Access the original repo [here](https://github.com/aliasrobotics/cai).

[![version](https://badge.fury.io/py/cai-framework.svg)](https://badge.fury.io/py/cai-framework)
[![downloads](https://static.pepy.tech/badge/cai-framework)](https://pepy.tech/projects/cai-framework)
[![Linux](https://img.shields.io/badge/Linux-Supported-brightgreen?logo=linux&logoColor=white)](https://github.com/aliasrobotics/cai)
[![OS X](https://img.shields.io/badge/OS%20X-Supported-brightgreen?logo=apple&logoColor=white)](https://github.com/aliasrobotics/cai)
[![Windows](https://img.shields.io/badge/Windows-Supported-brightgreen?logo=windows&logoColor=white)](https://github.com/aliasrobotics/cai)
[![Android](https://img.shields.io/badge/Android-Supported-brightgreen?logo=android&logoColor=white)](https://github.com/aliasrobotics/cai)
[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/fnUFcTaQAC)
[![arXiv](https://img.shields.io/badge/arXiv-2504.06017-b31b1b.svg)](https://arxiv.org/pdf/2504.06017)
[![arXiv](https://img.shields.io/badge/arXiv-2506.23592-b31b1b.svg)](https://arxiv.org/abs/2506.23592)
[![arXiv](https://img.shields.io/badge/arXiv-2508.13588-b31b1b.svg)](https://arxiv.org/abs/2508.13588)
[![arXiv](https://img.shields.io/badge/arXiv-2508.21669-b31b1b.svg)](https://arxiv.org/abs/2508.21669)
[![arXiv](https://img.shields.io/badge/arXiv-2509.14096-b31b1b.svg)](https://arxiv.org/abs/2509.14096) 
[![arXiv](https://img.shields.io/badge/arXiv-2509.14139-b31b1b.svg)](https://arxiv.org/abs/2509.14139)

<div align="center">
  <p>
    <a align="center" href="" target="https://github.com/aliasrobotics/CAI">
      <img
        width="100%"
        src="https://github.com/aliasrobotics/cai/raw/main/media/cai.png"
      >
    </a>
  </p>
</div>

CAI is the *de facto* framework for AI Security, used by thousands of individual users and hundreds of organizations. Whether you're a security researcher, ethical hacker, or IT professional, CAI provides the building blocks to create specialized AI agents for:

*   Mitigation
*   Vulnerability Discovery
*   Exploitation
*   Security Assessment

**Key Features:**

*   🤖 **300+ AI Models:** Broad support for OpenAI, Anthropic, DeepSeek, Ollama, and more, powered by LiteLLM.
*   🔧 **Built-in Security Tools:** Ready-to-use tools for reconnaissance, exploitation, and privilege escalation.
*   🏆 **Battle-tested:** Proven in HackTheBox CTFs, bug bounties, and real-world security [case studies](https://aliasrobotics.com/case-studies-robot-cybersecurity.php).
*   🎯 **Agent-based Architecture:** A modular framework for building specialized agents tailored for various security tasks.
*   🛡️ **Guardrails Protection:** Includes input and output defenses to mitigate prompt injection and dangerous command execution.
*   📚 **Research-oriented:** Research foundation democratizing cybersecurity AI for the community.

> [!NOTE]
> Read the technical report: [CAI: An Open, Bug Bounty-Ready Cybersecurity AI](https://arxiv.org/pdf/2504.06017)
>
> For further readings, refer to our [impact](#-impact) and [CAI citation](#citation) sections.

**Case Studies:**

[Case studies](https://aliasrobotics.com/case-studies-robot-cybersecurity.php) showcase CAI's effectiveness.

| [`OT` - CAI and alias0 on: Ecoforest Heat Pumps](https://aliasrobotics.com/case-study-ecoforest.php) | [`Robotics` - CAI and alias0 on: Mobile Industrial Robots (MiR)](https://aliasrobotics.com/case-study-cai-mir.php) |
|------------------------------------------------|---------------------------------|
| CAI discovers critical vulnerability in Ecoforest heat pumps allowing unauthorized remote access and potential catastrophic failures. AI-powered security testing reveals exposed credentials and DES encryption weaknesses affecting all of their deployed units across Europe.  | CAI-powered security testing of MiR (Mobile Industrial Robot) platform through automated ROS message injection attacks. This study demonstrates how AI-driven vulnerability discovery can expose unauthorized access to robot control systems and alarm triggers.  |
| [![](https://aliasrobotics.com/img/case-study-portada-ecoforest.png)](https://aliasrobotics.com/case-study-ecoforest.php) | [![](https://aliasrobotics.com/img/case-study-portada-mir-cai.png)](https://aliasrobotics.com/case-study-cai-mir.php) |

| [`IT` (Web) - CAI and alias0 on: Mercado Libre's e-commerce](https://aliasrobotics.com/case-study-mercado-libre.php) | [`OT` - CAI and alias0 on: MQTT broker](https://aliasrobotics.com/case-study-cai-mqtt-broker.php) |
|------------------------------------------------|---------------------------------|
|  CAI-powered API vulnerability discovery at Mercado Libre through automated enumeration attacks. This study demonstrates how AI-driven security testing can expose user data exposure risks in e-commerce platforms at scale.  |  CAI-powered testing exposed critical flaws in an MQTT broker within a Dockerized OT network. Without authentication, CAI subscribed to temperature and humidity topics and injected false values, corrupting data shown in Grafana dashboards. |
| [![](https://aliasrobotics.com/img/case-study-portada-mercado-libre.png)](https://aliasrobotics.com/case-study-mercado-libre.php) | [![](https://aliasrobotics.com/img/case-study-portada-mqtt-broker-cai.png)](https://aliasrobotics.com/case-study-cai-mqtt-broker.php) |

> [!WARNING]
> :warning: CAI is in active development; contribute by raising an issue or [sending a PR](https://github.com/aliasrobotics/cai/pulls).
>
> Access to this library and the use of information, materials (or portions thereof), is **<u>not intended</u>, and is <u>prohibited</u>, where such access or use violates applicable laws or regulations**. By no means the authors encourage or promote the unauthorized tampering with running systems. This can cause serious human harm and material damages.
>
> *By no means the authors of CAI encourage or promote the unauthorized tampering with compute systems. Please don't use the source code in here for cybercrime. <u>Pentest for good instead</u>*. By downloading, using, or modifying this source code, you agree to the terms of the [`LICENSE`](LICENSE) and the limitations outlined in the [`DISCLAIMER`](DISCLAIMER) file.

## :bookmark: Table of Contents

-   [Cybersecurity AI (`CAI`)](#cybersecurity-ai-cai)
    -   [:bookmark: Table of Contents](#bookmark-table-of-contents)
    -   [🎯 Impact](#-impact)
        -   [🏆 Competitions and challenges](#-competitions-and-challenges)
        -   [📊 Research Impact](#-research-impact)
        -   [📚 Research products: `Cybersecurity AI`](#-research-products-cybersecurity-ai)
    -   [PoCs](#pocs)
    -   [Motivation](#motivation)
        -   [:bust\_in\_silhouette: Why CAI?](#bust_in_silhouette-why-cai)
        -   [Ethical principles behind CAI](#ethical-principles-behind-cai)
        -   [Closed-source alternatives](#closed-source-alternatives)
    -   [Learn - `CAI` Fluency](#learn---cai-fluency)
    -   [:nut\_and\_bolt: Install](#nut_and_bolt-install)
        -   [OS X](#os-x)
        -   [Ubuntu 24.04](#ubuntu-2404)
        -   [Ubuntu 20.04](#ubuntu-2004)
        -   [Windows WSL](#windows-wsl)
        -   [Android](#android)
        -   [:nut\_and\_bolt: Setup `.env` file](#nut_and_bolt-setup-env-file)
        -   [🔹 Custom OpenAI Base URL Support](#-custom-openai-base-url-support)
    -   [:triangular\_ruler: Architecture:](#triangular_ruler-architecture)
        -   [🔹 Agent](#-agent)
        -   [🔹 Tools](#-tools)
        -   [🔹 Handoffs](#-handoffs)
        -   [🔹 Patterns](#-patterns)
        -   [🔹 Turns and Interactions](#-turns-and-interactions)
        -   [🔹 Tracing](#-tracing)
        -   [🔹 Guardrails](#-guardrails)
        -   [🔹 Human-In-The-Loop (HITL)](#-human-in-the-loop-hitl)
    -   [:rocket: Quickstart](#rocket-quickstart)
        -   [Environment Variables](#environment-variables)
        -   [OpenRouter Integration](#openrouter-integration)
        -   [MCP](#mcp)
    -   [Development](#development)
        -   [Contributions](#contributions)
        -   [Optional Requirements: caiextensions](#optional-requirements-caiextensions)
        -   [:information\_source: Usage Data Collection](#information_source-usage-data-collection)
        -   [Reproduce CI-Setup locally](#reproduce-ci-setup-locally)
    -   [FAQ](#faq)
    -   [Citation](#citation)
    -   [Acknowledgements](#acknowledgements)
        -   [Academic Collaborations](#academic-collaborations)

## 🎯 Impact

### 🏆 Competitions and challenges

See badges for rankings, etc.

[![](https://img.shields.io/badge/HTB_ranking-top_90_Spain_(5_days)-red.svg)](https://app.hackthebox.com/users/2268644)
[![](https://img.shields.io/badge/HTB_ranking-top_50_Spain_(6_days)-red.svg)](https://app.hackthebox.com/users/2268644)
[![](https://img.shields.io/badge/HTB_ranking-top_30_Spain_(7_days)-red.svg)](https://app.hackthebox.com/users/2268644)
[![](https://img.shields.io/badge/HTB_ranking-top_500_World_(7_days)-red.svg)](https://app.hackthebox.com/users/2268644)
[![](https://img.shields.io/badge/HTB_"Human_vs_AI"_CTF-top_1_(AIs)_world-red.svg)](https://ctf.hackthebox.com/event/2000/scoreboard)
[![](https://img.shields.io/badge/HTB_"Human_vs_AI"_CTF-top_1_Spain-red.svg)](https://ctf.hackthebox.com/event/2000/scoreboard)
[![](https://img.shields.io/badge/HTB_"Human_vs_AI"_CTF-top_20_World-red.svg)](https://ctf.hackthebox.com/event/2000/scoreboard)
[![](https://img.shields.io/badge/HTB_"Human_vs_AI"_CTF-750_$-yellow.svg)](https://ctf.hackthebox.com/event/2000/scoreboard)
[![](https://img.shields.io/badge/Mistral_AI_Robotics_Hackathon-2500_$-yellow.svg)](https://lu.ma/roboticshack?tk=RuryKF)

### 📊 Research Impact

*   Pioneered LLM-powered AI Security with PentestGPT, establishing the foundation for the `Cybersecurity AI` research domain [![arXiv](https://img.shields.io/badge/arXiv-2308.06782-4a9b8e.svg)](https://arxiv.org/pdf/2308.06782)
*   Established the `Cybersecurity AI` research line with **6 papers and technical reports**, with active research collaborations [![arXiv](https://img.shields.io/badge/arXiv-2504.06017-63bfab.svg)](https://arxiv.org/pdf/2504.06017) [![arXiv](https://img.shields.io/badge/arXiv-2506.23592-7dd3c0.svg)](https://arxiv.org/abs/2506.23592) [![arXiv](https://img.shields.io/badge/arXiv-2508.13588-52a896.svg)](https://arxiv.org/abs/2508.13588) [![arXiv](https://img.shields.io/badge/arXiv-2508.21669-85e0d1.svg)](https://arxiv.org/abs/2508.21669) [![arXiv](https://img.shields.io/badge/arXiv-2509.14096-3e8b7a.svg)](https://arxiv.org/pdf/2509.14096) [![arXiv](https://img.shields.io/badge/arXiv-2509.14139-6bc7b5.svg)](https://arxiv.org/pdf/2509.14139)

*   Demonstrated **3,600× performance improvement** over human penetration testers in standardized CTF benchmark evaluations [![arXiv](https://img.shields.io/badge/arXiv-2504.06017-63bfab.svg)](https://arxiv.org/pdf/2504.06017)
*   Identified **CVSS 4.3-7.5 severity vulnerabilities** in production systems through automated security assessment [![arXiv](https://img.shields.io/badge/arXiv-2504.06017-63bfab.svg)](https://arxiv.org/pdf/2504.06017)
*   **Democratization of AI-empowered vulnerability research**: CAI enables both non-security domain experts and experienced researchers to conduct more efficient vulnerability discovery, expanding the security research community while empowering small and medium enterprises to conduct autonomous security assessments [![arXiv](https://img.shields.io/badge/arXiv-2504.06017-63bfab.svg)](https://arxiv.org/pdf/2504.06017)
*   **Systematic evaluation of large language models** across both proprietary and open-weight architectures, revealing <u>substantial gaps</u> between vendor-reported capabilities and empirical cybersecurity performance metrics [![arXiv](https://img.shields.io/badge/arXiv-2504.06017-63bfab.svg)](https://arxiv.org/pdf/2504.06017)
*   Established the **autonomy levels in cybersecurity** and argued about autonomy vs automation in the field [![arXiv](https://img.shields.io/badge/arXiv-2506.23592-7dd3c0.svg)](https://arxiv.org/abs/2506.23592)
*   **Collaborative research initiatives** with international academic institutions focused on developing cybersecurity education curricula and training methodologies [![arXiv](https://img.shields.io/badge/arXiv-2508.13588-52a896.svg)](https://arxiv.org/abs/2508.13588)
*   **Contributed a comprehensive defense framework against prompt injection in AI security agents**: developed and empirically validated a multi-layered defense system that addresses the identified prompt injection issues [![arXiv](https://img.shields.io/badge/arXiv-2508.21669-85e0d1.svg)](https://arxiv.org/abs/2508.21669)
*   Explored the Cybersecurity of Humanoid Robots with CAI and identified new attack vectors showing how it `(a)` operates simultaneously as a covert surveillance node and `(b)` can be purposed as an active cyber operations platform [![arXiv](https://img.shields.io/badge/arXiv-2509.14096-3e8b7a.svg)](https://arxiv.org/pdf/2509.14096) [![arXiv](https://img.shields.io/badge/arXiv-2509.14139-6bc7b5.svg)](https://arxiv.org/pdf/2509.14139)

### 📚 Research products: `Cybersecurity AI`

| CAI, An Open, Bug Bounty-Ready Cybersecurity AI [![arXiv](https://img.shields.io/badge/arXiv-2504.06017-63bfab.svg)](https://arxiv.org/pdf/2504.06017) | The Dangerous Gap Between Automation and Autonomy [![arXiv](https://img.shields.io/badge/arXiv-2506.23592-7dd3c0.svg)](https://arxiv.org/abs/2506.23592) | CAI Fluency, A Framework for Cybersecurity AI Fluency [![arXiv](https://img.shields.io/badge/arXiv-2508.13588-52a896.svg)](https://arxiv.org/abs/2508.13588) |
| --- | --- | --- |
| [<img src="https://aliasrobotics.com/img/paper-cai.png" width="350">](https://arxiv.org/pdf/2504.06017) | [<img src="https://aliasrobotics.com/img/cai_automation_vs_autonomy.png" width="350">](https://www.arxiv.org/pdf/2506.23592) | [<img src="https://aliasrobotics.com/img/cai_fluency_cover.png" width="350">](https://arxiv.org/pdf/2508.13588) |

| Hacking the AI Hackers via Prompt Injection [![arXiv](https://img.shields.io/badge/arXiv-2508.21669-85e0d1.svg)](https://arxiv.org/abs/2508.21669) | Humanoid Robots as Attack Vectors [![arXiv](https://img.shields.io/badge/arXiv-2509.14139-6bc7b5.svg)](https://arxiv.org/pdf/2509.14139) | The Cybersecurity of a Humanoid Robot [![arXiv](https://img.shields.io/badge/arXiv-2509.14096-3e8b7a.svg)](https://arxiv.org/pdf/2509.14096) |
| --- | --- | --- |
| [<img src="https://aliasrobotics.com/img/aihackers.jpeg" width="350">](https://arxiv.org/pdf/2508.21669) | [<img src="https://aliasrobotics.com/img/humanoids-cover.png" width="350">](https://arxiv.org/pdf/2509.14139) | [<img src="https://aliasrobotics.com/img/humanoid.png" width="350">](https://arxiv.org/pdf/2509.14096) |

## PoCs

[PoCs](https://aliasrobotics.com/case-studies-robot-cybersecurity.php)

| CAI with `alias0` on ROS message injection attacks in MiR-100 robot | CAI with `alias0` on API vulnerability discovery at Mercado Libre |
|-----------------------------------------------|---------------------------------|
| [![asciicast](https://asciinema.org/a/dNv705hZel2Rzrw0cju9HBGPh.svg)](https://asciinema.org/a/dNv705hZel2Rzrw0cju9HBGPh) | [![asciicast](https://asciinema.org/a/9Hc9z1uFcdNjqP3bY5y7wO1Ww.svg)](https://asciinema.org/a/9Hc9z1uFcdNjqP3bY5y7wO1Ww) |

| CAI on JWT@PortSwigger CTF — Cybersecurity AI | CAI on HackableII Boot2Root CTF — Cybersecurity AI |
|-----------------------------------------------|---------------------------------|
| [![asciicast](https://asciinema.org/a/713487.svg)](https://asciinema.org/a/713487) | [![asciicast](https://asciinema.org/a/713485.svg)](https://asciinema.org/a/713485) |

## Motivation

### :bust_in_silhouette: Why CAI?

*   **AI-Powered Security's Growing Importance**: AI is becoming essential to address complex security vulnerabilities and stay ahead of sophisticated threats.
*   **Democratizing Cybersecurity AI**: CAI is open-source to empower security researchers and organizations.
*   **Ethical Principles**: We make these tools available so that security professionals can level the playing field.
*   **Bug Bounty Enhancement**: CAI is designed to enhance bug bounty programs to find vulnerabilities.

### Ethical principles behind CAI

*   **Democratizing Cybersecurity AI**: Providing tools to the entire security community.
*   **Transparency in AI Security Capabilities**: Benchmarking AI capabilities in cybersecurity.

CAI is built on the following core principles:

*   **Cybersecurity oriented AI framework**: CAI is specifically designed for cybersecurity use cases, aiming at semi- and fully-automating offensive and defensive security tasks.
*   **Open source, free for research**: CAI is open source and free for research purposes. We aim at democratizing access to AI and Cybersecurity. For professional or commercial use, including on-premise deployments, dedicated technical support and custom extensions [reach out](mailto:research@aliasrobotics.com) to obtain a license.
*   **Lightweight**: CAI is designed to be fast, and easy to use.
*   **Modular and agent-centric design**: CAI operates on the basis of agents and agentic patterns, which allows flexibility and scalability. You can easily add the most suitable agents and pattern for your cybersecuritytarget case.
*   **Tool-integration**: CAI integrates already built-in tools, and allows the user to integrate their own tools with their own logic easily.
*   **Logging and tracing integrated**: using [`phoenix`](https://github.com/Arize-ai/phoenix), the open source tracing and logging tool for LLMs. This provides the user with a detailed traceability of the agents and their execution.
*   **Multi-Model Support**: more than 300 supported and empowered by [LiteLLM](https://github.com/BerriAI/litellm). The most popular providers:
  - **Anthropic**: `Claude 3.7`, `Claude 3.5`, `Claude 3`, `Claude 3 Opus`
  - **OpenAI**: `O1`, `O1 Mini`, `O3 Mini`, `GPT-4o`, `GPT-4.5 Preview`
  - **DeepSeek**: `DeepSeek V3`, `DeepSeek R1`
  - **Ollama**: `Qwen2.5 72B`, `Qwen2.5 14B`, etc

### Closed-source alternatives

A list of closed source alternatives.

## Learn - `CAI` Fluency

[CAI Fluency](https://github.com/aliasrobotics/cai/blob/main/media/caiedu.PNG)

>   [!NOTE]
>
>   CAI Fluency technical report ([arXiv:2508.13588](https://arxiv.org/pdf/2508.13588)) establishes formal educational frameworks for cybersecurity AI literacy.

| **Episode** | Description                                                                                                                                             | English                                                                                                                                                           | Spanish                                                                                                                                                           |
| :---------- | :------------------------------------------------------------------------------------------------------------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Episode 0**: What is CAI? | Cybersecurity AI (`CAI`) explained  |  [![Watch the video](https://img.youtube.com/vi/nBdTxbKM4oo/0.jpg)](https://www.youtube.com/watch?v=nBdTxbKM4oo) | [![Watch the video](https://img.youtube.com/vi/FaUL9HXrQ5k/0.jpg)](https://www.youtube.com/watch?v=FaUL9HXrQ5k) |
| **Episode 1**: The `CAI` Framework | Vision & Ethics - Explore the core motivation behind CAI and delve into the crucial ethical principles guiding its development. Understand the motivation behind CAI and how you can actively contribute to the future of cybersecurity and the CAI framework. | [![Watch the video](https://img.youtube.com/vi/QEiGdsMf29M/0.jpg)](https://www.youtube.com/watch?v=QEiGdsMf29M&list=PLLc16OUiZWd4RuFdN5_Wx9xwjCVVbopzr&index=3) |  |
| **Episode 2**: From Zero to Cyber Hero | Breaking into Cybersecurity with AI - A comprehensive guide for complete beginners to become cybersecurity practitioners using CAI and AI tools. Learn how to leverage artificial intelligence to accelerate your cybersecurity learning journey, from understanding basic security concepts to performing real-world security assessments, all without requiring prior cybersecurity experience. | [![Watch the video](https://img.youtube.com/vi/hSTLHOOcQoY/0.jpg)](https://www.youtube.com/watch?v=hSTLHOOcQoY&list=PLLc16OUiZWd4RuFdN5_Wx9xwjCVVbopzr&index=14) |  |
| **Episode 3**: Vibe-Hacking Tutorial | "My first Hack" - A Vibe-Hacking guide for newbies. We demonstrate a simple web security hack using a default agent and show how to leverage tools and interpret CIA output with the help of the CAI Python API. You'll also learn to compare different LLM models to find the best fit for your hacking endeavors. | [![Watch the video](https://img.youtube.com/vi/9vZ_Iyex7uI/0.jpg)](https://www.youtube.com/watch?v=9vZ_Iyex7uI&list=PLLc16OUiZWd4RuFdN5_Wx9xwjCVVbopzr&index=1) | [![Watch the video](https://img.youtube.com/vi/iAOMaI1ftiA/0.jpg)](https://www.youtube.com/watch?v=iAOMaI1ftiA&list=PLLc16OUiZWd4RuFdN5_Wx9xwjCVVbopzr&index=2) |
| **Episode 4**: Intro ReAct | The Evolution of LLMs - Learn how LLMs evolved from basic language models to advanced multiagency AI systems. From basic LLMs to Chain-of-Thought and Reasoning LLMs towards ReAct and Multi-Agent Architectures. Get to know the basic terms | [![Watch the video](https://img.youtube.com/vi/tLdFO1flj_o/0.jpg)](https://www.youtube.com/watch?v=tLdFO1flj_o&list=PLLc16OUiZWd4RuFdN5_Wx9xwjCVVbopzr&index=13) | |
| **Episode 5**: CAI on CTF challenges | Dive into Capture The Flag (CTF) competitions using CAI. Learn how to leverage AI agents to solve various cybersecurity challenges including web exploitation, cryptography, reverse engineering, and forensics. Discover how to configure CAI for competitive hacking scenarios and maximize your CTF performance with intelligent automation. | [![Watch the video](https://img.youtube.com/vi/MrXTQ0e2to4/0.jpg)](https://www.youtube.com/watch?v=MrXTQ0e2to4&list=PLLc16OUiZWd4RuFdN5_Wx9xwjCVVbopzr&index=13) | [![Watch the video](https://img.youtube.com/vi/r9US_JZa9_c/0.jpg)](https://www.youtube.com/watch?v=r9US_JZa9_c&list=PLLc16OUiZWd4RuFdN5_Wx9xwjCVVbopzr&index=12) |
|  |  |  |  |
| **Annex 1**: `CAI` 0.5.x release  | Introduce version 0.5 of `CAI` including new multi-agent functionality, new commands such as `/history`, `/compact`, `/graph` or `/memory` and a case study showing how `CAI` found a critical security flaw in OT heap pumps spread around the world. |  [![Watch the video](https://img.youtube.com/vi/OPFH0ANUMMw/0.jpg)](https://www.youtube.com/watch?v=OPFH0ANUMMw) | [![Watch the video](https://img.youtube.com/vi/Q8AI4E4gH8k/0.jpg)](https://www.youtube.com/watch?v=Q8AI4E4gH8k) |
| **Annex 2**: `CAI` 0.4.x release and `alias0`  | Introducing version 0.4 of `CAI` with *streaming* and improved MCP support. We also introduce `alias0`, the Privacy-First Cybersecurity AI, a Model-of-Models Intelligence that implements a Privacy-by-Design architecture and obtains state-of-the-art results in cybersecurity benchmarks. |  [![Watch the video](https://img.youtube.com/vi/NZjzfnvAZcc/0.jpg)](https://www.youtube.com/watch?v=NZjzfnvAZcc) |  |
| **Annex 3**: Cybersecurity AI Community Meeting #1  | First Cybersecurity AI (`CAI`) community meeting, over 40 participants from academia, industry, and defense gathered to discuss the open-source scaffolding behind CAI — a project designed to build agentic AI systems for cybersecurity that are open, modular, and Bug Bounty-ready. |  [![Watch the video](https://img.youtube.com/vi/4JqaTiVlgsw/0.jpg)](https://www.youtube.com/watch?v=4JqaTiVlgsw) |  |

## :nut_and_bolt: Install

```bash
pip install cai-framework
```

Always create a new virtual environment to ensure proper dependency installation when updating CAI.

### OS X

```bash
brew update && \
    brew install git python@3.12

# Create virtual environment
python3.12 -m venv cai_env

# Install the package from the local directory
source cai_env/bin/activate && pip install cai-framework

# Generate a .env file and set up with defaults
echo -e 'OPENAI_API_KEY="sk-1234"\nANTHROPIC_API_KEY=""\nOLLAMA=""\nPROMPT_TOOLKIT_NO_CPR=1\nCAI_STREAM=false' > .env

# Launch CAI
cai  # first launch it can take up to 30 seconds
```

### Ubuntu 24.04

```bash
sudo apt-get update && \
    sudo apt-get install -y git python3-pip python3.12-venv

# Create the virtual environment
python3.12 -m venv cai_env

# Install the package from the local directory
source cai_env/bin/activate && pip install cai-framework

# Generate a .env file and set up with defaults
echo -e 'OPENAI_API_KEY="sk-1234"\nANTHROPIC_API_KEY=""\nOLLAMA=""\nPROMPT_TOOLKIT_NO_CPR=1\nCAI_STREAM=false' > .env

# Launch CAI
cai  # first launch it can take up to 30 seconds
```

### Ubuntu 20.04

```bash
sudo apt-get update && \
    sudo apt-get install -y software-properties-common

# Fetch Python 3.12
sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev -y

# Create the virtual environment
python3.12 -m venv cai_env

# Install the package from the local directory
source cai_env/bin/activate && pip install cai-framework

# Generate a .env file and set up with defaults
echo -e 'OPENAI_API_KEY="sk-1234"\nANTHROPIC_API_KEY=""\nOLLAMA=""\nPROMPT_TOOLKIT_NO_CPR=1\nCAI_STREAM=false' > .env

# Launch CAI
cai  # first launch it can take up to 30 seconds
```

### Windows WSL

```bash
# From Powershell write: w