# Cybersecurity AI (CAI): Unleash AI-Powered Security Automation 🛡️

**CAI is an open-source framework empowering security professionals to build and deploy AI-driven offensive and defensive automation, revolutionizing cybersecurity.**  [View the original repository](https://github.com/aliasrobotics/cai).

<div align="center">
  <p>
    <a align="center" href="" target="https://github.com/aliasrobotics/CAI">
      <img
        width="100%"
        src="https://github.com/aliasrobotics/cai/raw/main/media/cai.png"
      >
    </a>
  </p>


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


</div>

**Key Features:**

*   **🤖 300+ AI Models**:  Comprehensive support for leading AI models, including OpenAI, Anthropic, DeepSeek, and Ollama.
*   **🔧 Built-in Security Tools**:  Ready-to-use tools streamline reconnaissance, exploitation, and privilege escalation tasks.
*   **🏆 Battle-Tested**:  Proven effectiveness in HackTheBox CTFs, bug bounties, and real-world [case studies](https://aliasrobotics.com/case-studies-robot-cybersecurity.php).
*   **🎯 Agent-Based Architecture**:  Modular design allows for creating specialized AI agents tailored to diverse security needs.
*   **🛡️ Guardrails Protection**:  Integrated defenses against prompt injection and malicious command execution.
*   **📚 Research-Oriented**:  Foundation for democratizing cybersecurity AI, driving innovation, and empowering the community.

> [!NOTE]
> Explore the in-depth technical report: [CAI: An Open, Bug Bounty-Ready Cybersecurity AI](https://arxiv.org/pdf/2504.06017)
>
> For more resources, see the [Impact](#-impact) and [Citation](#citation) sections.

### Case Studies

| [`OT` - CAI and alias0 on: Ecoforest Heat Pumps](https://aliasrobotics.com/case-study-ecoforest.php) | [`Robotics` - CAI and alias0 on: Mobile Industrial Robots (MiR)](https://aliasrobotics.com/case-study-cai-mir.php) |
|------------------------------------------------|---------------------------------|
| CAI discovers critical vulnerability in Ecoforest heat pumps allowing unauthorized remote access and potential catastrophic failures. AI-powered security testing reveals exposed credentials and DES encryption weaknesses affecting all of their deployed units across Europe.  | CAI-powered security testing of MiR (Mobile Industrial Robot) platform through automated ROS message injection attacks. This study demonstrates how AI-driven vulnerability discovery can expose unauthorized access to robot control systems and alarm triggers.  |
| [![](https://aliasrobotics.com/img/case-study-portada-ecoforest.png)](https://aliasrobotics.com/case-study-ecoforest.php) | [![](https://aliasrobotics.com/img/case-study-portada-mir-cai.png)](https://aliasrobotics.com/case-study-cai-mir.php) |

| [`IT` (Web) - CAI and alias0 on: Mercado Libre's e-commerce](https://aliasrobotics.com/case-study-mercado-libre.php) | [`OT` - CAI and alias0 on: MQTT broker](https://aliasrobotics.com/case-study-cai-mqtt-broker.php) |
|------------------------------------------------|---------------------------------|
|  CAI-powered API vulnerability discovery at Mercado Libre through automated enumeration attacks. This study demonstrates how AI-driven security testing can expose user data exposure risks in e-commerce platforms at scale.  |  CAI-powered testing exposed critical flaws in an MQTT broker within a Dockerized OT network. Without authentication, CAI subscribed to temperature and humidity topics and injected false values, corrupting data shown in Grafana dashboards. |
| [![](https://aliasrobotics.com/img/case-study-portada-mercado-libre.png)](https://aliasrobotics.com/case-study-mercado-libre.php) | [![](https://aliasrobotics.com/img/case-study-portada-mqtt-broker-cai.png)](https://aliasrobotics.com/case-study-cai-mqtt-broker.php) |

> [!WARNING]
> :warning: CAI is under active development. Please contribute by raising issues or submitting pull requests ([PRs](https://github.com/aliasrobotics/cai/pulls)).
>
> Access to this library is not intended, and is prohibited, where such access or use violates applicable laws or regulations. The authors do not encourage or promote unauthorized tampering with running systems, which can cause serious harm.
>
> *The authors of CAI do not encourage or promote the unauthorized tampering with compute systems. Please don't use the source code in here for cybercrime. <u>Pentest for good instead</u>*. By using this source code, you agree to the terms of the [`LICENSE`](LICENSE) and the limitations outlined in the [`DISCLAIMER`](DISCLAIMER) file.

## Table of Contents

-   [Cybersecurity AI (CAI): Unleash AI-Powered Security Automation](#cybersecurity-ai-cai-unleash-ai-powered-security-automation-️)
    -   [Key Features](#key-features)
    -   [Table of Contents](#table-of-contents)
    -   [🎯 Impact](#-impact)
        -   [🏆 Competitions and Challenges](#-competitions-and-challenges)
        -   [📊 Research Impact](#-research-impact)
        -   [📚 Research Products: Cybersecurity AI](#-research-products-cybersecurity-ai)
    -   [PoCs](#pocs)
    -   [Motivation](#motivation)
        -   [:bust\_in\_silhouette: Why CAI?](#bust_in_silhouette-why-cai)
        -   [Ethical Principles Behind CAI](#ethical-principles-behind-cai)
        -   [Closed-Source Alternatives](#closed-source-alternatives)
    -   [Learn - CAI Fluency](#learn---cai-fluency)
    -   [:nut\_and\_bolt: Install](#nut_and_bolt-install)
        -   [OS X](#os-x)
        -   [Ubuntu 24.04](#ubuntu-2404)
        -   [Ubuntu 20.04](#ubuntu-2004)
        -   [Windows WSL](#windows-wsl)
        -   [Android](#android)
        -   [:nut\_and\_bolt: Setup `.env` File](#nut_and_bolt-setup-env-file)
        -   [🔹 Custom OpenAI Base URL Support](#-custom-openai-base-url-support)
    -   [:triangular\_ruler: Architecture](#triangular_ruler-architecture)
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
        -   [Reproduce CI-Setup Locally](#reproduce-ci-setup-locally)
    -   [FAQ](#faq)
    -   [Citation](#citation)
    -   [Acknowledgements](#acknowledgements)
        -   [Academic Collaborations](#academic-collaborations)

## 🎯 Impact

### 🏆 Competitions and Challenges

[![HTB Ranking Top 90 Spain](https://img.shields.io/badge/HTB_ranking-top_90_Spain_(5_days)-red.svg)](https://app.hackthebox.com/users/2268644)
[![HTB Ranking Top 50 Spain](https://img.shields.io/badge/HTB_ranking-top_50_Spain_(6_days)-red.svg)](https://app.hackthebox.com/users/2268644)
[![HTB Ranking Top 30 Spain](https://img.shields.io/badge/HTB_ranking-top_30_Spain_(7_days)-red.svg)](https://app.hackthebox.com/users/2268644)
[![HTB Ranking Top 500 World](https://img.shields.io/badge/HTB_ranking-top_500_World_(7_days)-red.svg)](https://app.hackthebox.com/users/2268644)
[![HTB Human vs AI CTF Top 1 AI World](https://img.shields.io/badge/HTB_"Human_vs_AI"_CTF-top_1_(AIs)_world-red.svg)](https://ctf.hackthebox.com/event/2000/scoreboard)
[![HTB Human vs AI CTF Top 1 Spain](https://img.shields.io/badge/HTB_"Human_vs_AI"_CTF-top_1_Spain-red.svg)](https://ctf.hackthebox.com/event/2000/scoreboard)
[![HTB Human vs AI CTF Top 20 World](https://img.shields.io/badge/HTB_"Human_vs_AI"_CTF-top_20_World-red.svg)](https://ctf.hackthebox.com/event/2000/scoreboard)
[![HTB Human vs AI CTF Reward](https://img.shields.io/badge/HTB_"Human_vs_AI"_CTF-750_$-yellow.svg)](https://ctf.hackthebox.com/event/2000/scoreboard)
[![Mistral AI Robotics Hackathon](https://img.shields.io/badge/Mistral_AI_Robotics_Hackathon-2500_$-yellow.svg)](https://lu.ma/roboticshack?tk=RuryKF)

### 📊 Research Impact

*   Pioneered LLM-powered AI Security with PentestGPT, setting the foundation for the Cybersecurity AI research domain [![arXiv](https://img.shields.io/badge/arXiv-2308.06782-4a9b8e.svg)](https://arxiv.org/pdf/2308.06782)
*   Established the Cybersecurity AI research line with **6 papers and technical reports**, actively collaborating with researchers. [![arXiv](https://img.shields.io/badge/arXiv-2504.06017-63bfab.svg)](https://arxiv.org/pdf/2504.06017) [![arXiv](https://img.shields.io/badge/arXiv-2506.23592-7dd3c0.svg)](https://arxiv.org/abs/2506.23592) [![arXiv](https://img.shields.io/badge/arXiv-2508.13588-52a896.svg)](https://arxiv.org/pdf/2508.13588) [![arXiv](https://img.shields.io/badge/arXiv-2508.21669-85e0d1.svg)](https://arxiv.org/abs/2508.21669) [![arXiv](https://img.shields.io/badge/arXiv-2509.14096-3e8b7a.svg)](https://arxiv.org/abs/2509.14096) [![arXiv](https://img.shields.io/badge/arXiv-2509.14139-6bc7b5.svg)](https://arxiv.org/abs/2509.14139)
*   Demonstrated **3,600× performance improvement** over human penetration testers in standardized CTF benchmark evaluations [![arXiv](https://img.shields.io/badge/arXiv-2504.06017-63bfab.svg)](https://arxiv.org/pdf/2504.06017)
*   Identified **CVSS 4.3-7.5 severity vulnerabilities** in production systems through automated security assessment [![arXiv](https://img.shields.io/badge/arXiv-2504.06017-63bfab.svg)](https://arxiv.org/pdf/2504.06017)
*   **Democratization of AI-empowered vulnerability research**: CAI empowers both non-security domain experts and experienced researchers to conduct more efficient vulnerability discovery, expanding the security research community while empowering small and medium enterprises to conduct autonomous security assessments [![arXiv](https://img.shields.io/badge/arXiv-2504.06017-63bfab.svg)](https://arxiv.org/pdf/2504.06017)
*   **Systematic evaluation of large language models** across both proprietary and open-weight architectures, revealing <u>substantial gaps</u> between vendor-reported capabilities and empirical cybersecurity performance metrics [![arXiv](https://img.shields.io/badge/arXiv-2504.06017-63bfab.svg)](https://arxiv.org/pdf/2504.06017)
*   Established the **autonomy levels in cybersecurity** and argued about autonomy vs automation in the field [![arXiv](https://img.shields.io/badge/arXiv-2506.23592-7dd3c0.svg)](https://arxiv.org/abs/2506.23592)
*   **Collaborative research initiatives** with international academic institutions focused on developing cybersecurity education curricula and training methodologies [![arXiv](https://img.shields.io/badge/arXiv-2508.13588-52a896.svg)](https://arxiv.org/pdf/2508.13588)
*   **Contributed a comprehensive defense framework against prompt injection in AI security agents**: developed and empirically validated a multi-layered defense system that addresses the identified prompt injection issues [![arXiv](https://img.shields.io/badge/arXiv-2508.21669-85e0d1.svg)](https://arxiv.org/abs/2508.21669)
*   Explored the Cybersecurity of Humanoid Robots with CAI and identified new attack vectors showing how it `(a)` operates simultaneously as a covert surveillance node and `(b)` can be purposed as an active cyber operations platform [![arXiv](https://img.shields.io/badge/arXiv-2509.14096-3e8b7a.svg)](https://arxiv.org/pdf/2509.14096) [![arXiv](https://img.shields.io/badge/arXiv-2509.14139-6bc7b5.svg)](https://arxiv.org/pdf/2509.14139)

### 📚 Research Products: Cybersecurity AI

| CAI, An Open, Bug Bounty-Ready Cybersecurity AI [![arXiv](https://img.shields.io/badge/arXiv-2504.06017-63bfab.svg)](https://arxiv.org/pdf/2504.06017) | The Dangerous Gap Between Automation and Autonomy [![arXiv](https://img.shields.io/badge/arXiv-2506.23592-7dd3c0.svg)](https://arxiv.org/abs/2506.23592) | CAI Fluency, A Framework for Cybersecurity AI Fluency [![arXiv](https://img.shields.io/badge/arXiv-2508.13588-52a896.svg)](https://arxiv.org/abs/2508.13588) |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                     [<img src="https://aliasrobotics.com/img/paper-cai.png" width="350">](https://arxiv.org/pdf/2504.06017)                      |                     [<img src="https://aliasrobotics.com/img/cai_automation_vs_autonomy.png" width="350">](https://www.arxiv.org/pdf/2506.23592)                     |                     [<img src="https://aliasrobotics.com/img/cai_fluency_cover.png" width="350">](https://arxiv.org/pdf/2508.13588)                     |

| Hacking the AI Hackers via Prompt Injection [![arXiv](https://img.shields.io/badge/arXiv-2508.21669-85e0d1.svg)](https://arxiv.org/abs/2508.21669) | Humanoid Robots as Attack Vectors [![arXiv](https://img.shields.io/badge/arXiv-2509.14139-6bc7b5.svg)](https://arxiv.org/pdf/2509.14139) | The Cybersecurity of a Humanoid Robot [![arXiv](https://img.shields.io/badge/arXiv-2509.14096-3e8b7a.svg)](https://arxiv.org/pdf/2509.14096) |
| :------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------: |
|                         [<img src="https://aliasrobotics.com/img/aihackers.jpeg" width="350">](https://arxiv.org/pdf/2508.21669)                          |                     [<img src="https://aliasrobotics.com/img/humanoids-cover.png" width="350">](https://arxiv.org/pdf/2509.14139)                     |                     [<img src="https://aliasrobotics.com/img/humanoid.png" width="350">](https://arxiv.org/pdf/2509.14096)                     |

## PoCs

| CAI with `alias0` on ROS message injection attacks in MiR-100 robot | CAI with `alias0` on API vulnerability discovery at Mercado Libre |
| :-----------------------------------------------------------------: | :--------------------------------------------------------------: |
|   [![asciicast](https://asciinema.org/a/dNv705hZel2Rzrw0cju9HBGPh.svg)](https://asciinema.org/a/dNv705hZel2Rzrw0cju9HBGPh)   |    [![asciicast](https://asciinema.org/a/9Hc9z1uFcdNjqP3bY5y7wO1Ww.svg)](https://asciinema.org/a/9Hc9z1uFcdNjqP3bY5y7wO1Ww)    |

| CAI on JWT@PortSwigger CTF — Cybersecurity AI | CAI on HackableII Boot2Root CTF — Cybersecurity AI |
| :----------------------------------------------------------------: | :-------------------------------------------------------------------: |
|      [![asciicast](https://asciinema.org/a/713487.svg)](https://asciinema.org/a/713487)      |       [![asciicast](https://asciinema.org/a/713485.svg)](https://asciinema.org/a/713485)       |

More case studies and PoCs are available at [https://aliasrobotics.com/case-studies-robot-cybersecurity.php](https://aliasrobotics.com/case-studies-robot-cybersecurity.php).

## Motivation

### :bust_in_silhouette: Why CAI?

The cybersecurity landscape is undergoing a dramatic transformation as AI becomes increasingly integrated into security operations. **We predict that by 2028, AI-powered security testing tools will outnumber human pentesters**. This shift represents a fundamental change in how we approach cybersecurity challenges. *AI is not just another tool - it's becoming essential for addressing complex security vulnerabilities and staying ahead of sophisticated threats. As organizations face more advanced cyber attacks, AI-enhanced security testing will be crucial for maintaining robust defenses.*

This work builds upon prior efforts and we believe that democratizing access to advanced cybersecurity AI tools is vital for the entire security community. That's why we're releasing Cybersecurity AI (CAI) as an open source framework. Our goal is to empower security researchers, ethical hackers, and organizations to build and deploy powerful AI-driven security tools. By making these capabilities openly available, we aim to level the playing field and ensure that cutting-edge security AI technology isn't limited to well-funded private companies or state actors.

Bug Bounty programs have become a cornerstone of modern cybersecurity, providing a crucial mechanism for organizations to identify and fix vulnerabilities in their systems before they can be exploited. These programs have proven highly effective at securing both public and private infrastructure, with researchers discovering critical vulnerabilities that might have otherwise gone unnoticed. CAI is specifically designed to enhance these efforts by providing a lightweight, ergonomic framework for building specialized AI agents that can assist in various aspects of Bug Bounty hunting - from initial reconnaissance to vulnerability validation and reporting. Our framework aims to augment human expertise with AI capabilities, helping researchers work more efficiently and thoroughly in their quest to make digital systems more secure.

### Ethical Principles Behind CAI

We believe in transparent AI security. Our decision to open-source CAI is guided by two core ethical principles:

1.  **Democratizing Cybersecurity AI:** We want to give the entire security community access to powerful AI tools, not just a few privileged entities.
2.  **Transparency in AI Security Capabilities:** Our research shows that current LLM vendors are often overstating their cybersecurity abilities. By open-sourcing CAI, we're creating a transparent benchmark for what AI can *actually* do in cybersecurity.

CAI adheres to core principles:

*   **Cybersecurity Focus**: CAI is designed for cybersecurity, aiming to automate offensive and defensive security tasks.
*   **Open Source, Free for Research**:  CAI is open source, fostering collaboration and access to AI for all. For commercial use, contact us to obtain a license.
*   **Lightweight**: CAI prioritizes speed and ease of use.
*   **Modular, Agent-Centric Design**:  CAI uses agents and agentic patterns, allowing for flexibility and scalability.
*   **Tool Integration**: CAI provides built-in tools and easy integration of custom tools.
*   **Logging & Tracing**: Powered by [`phoenix`](https://github.com/Arize-ai/phoenix) for detailed traceability.
*   **Multi-Model Support**: Compatible with numerous models from Anthropic, OpenAI, DeepSeek, and Ollama.

### Closed-Source Alternatives

While the field of Cybersecurity AI is rapidly evolving, many companies are developing closed-source solutions. We provide an incomplete list of these proprietary initiatives:

*   [Autonomous Cyber](https://www.acyber.co/)
*   [CrackenAGI](https://cracken.ai/)
*   [ETHIACK](https://ethiack.com/)
*   [Horizon3](https://horizon3.ai/)
*   [Irregular](https://www.irregular.com/)
*   [Kindo](https://www.kindo.ai/)
*   [Lakera](https://lakera.ai)
*   [Mindfort](www.mindfort.ai)
*   [Mindgard](https://mindgard.ai/)
*   [NDAY Security](https://ndaysecurity.com/)
*   [Runsybil](https://www.runsybil.com)
*   [Selfhack](https://www.selfhack.fi)
*   [Sola Security](https://sola.security/)
*   [SQUR](https://squr.ai/)
*   [Staris](https://staris.tech/)
*   [Sxipher](https://www.sxipher.com/) (seems discontinued)
*   [Terra Security](https://www.terra.security)
*   [Xint](https://xint.io/)
*   [XBOW](https://www.xbow.com)
*   [ZeroPath](https://www.zeropath.com)
*   [Zynap](https://www.zynap.com)
*   [7ai](https://7ai.com)

## Learn - CAI Fluency

<div align="center">
  <p>
    <a align="center" href="" target="https://github.com/aliasrobotics/CAI">
      <img
        width="100%"
        src="https://github.com/aliasrobotics/cai/raw/main/media/caiedu.PNG"
      >
    </a>
  </p>
</div>

> [!NOTE]
> CAI Fluency technical report ([arXiv:2508.13588](https://arxiv.org/pdf/2508.13588)) establishes formal educational frameworks for cybersecurity AI literacy.

| Episode | Description                              | English                                                                                                                               | Spanish                                                                                                          |
| :------ | :--------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------------------------- |
| **0**   | What is CAI?                             |  [![Watch the video](https://img.youtube.com/vi/nBdTxbKM4oo/0.jpg)](https://www.youtube.com/watch?v=nBdTxbKM4oo)                    | [![Watch the video](https://img.youtube.com/vi/FaUL9HXrQ5k/0.jpg)](https://www.youtube.com/watch?v=FaUL9HXrQ5k) |
| **1**   | The CAI Framework                      | Explore the core motivation behind CAI and delve into the crucial ethical principles guiding its development. Understand the motivation behind CAI and how you can actively contribute to the future of cybersecurity and the CAI framework.  |  |
| **2**   | From Zero to Cyber Hero                 | Breaking into Cybersecurity with AI - A comprehensive guide for complete beginners to become cybersecurity practitioners using CAI and AI tools. Learn how to leverage artificial intelligence to accelerate your cybersecurity learning journey, from understanding basic security concepts to performing real-world security assessments, all without requiring prior cybersecurity experience. | |
| **3**   | Vibe-Hacking Tutorial                  | "My first Hack" - A Vibe-Hacking guide for newbies. We demonstrate a simple web security hack using a default agent and show how to leverage tools and interpret CIA output with the help of the CAI Python API. You'll also learn to compare different LLM models to find the best fit for your hacking endeavors. | [![Watch the video](https://img.youtube.com/vi/iAOMaI1ftiA/0.jpg)](https://www.youtube.com/watch?v=iAOMaI1ftiA&list=PLLc16OUiZWd4RuFdN5_Wx9xwjCVVbopzr&index=2)  |
| **4**   | Intro ReAct                            | Learn how LLMs evolved from basic language models to advanced multiagency AI systems. From basic LLMs to Chain-of-Thought and Reasoning LLMs towards ReAct and Multi-Agent Architectures. Get to know the basic terms | |
| **5**   | CAI on CTF challenges | Dive into Capture The Flag (CTF) competitions using CAI. Learn how to leverage AI agents to solve various cybersecurity challenges including web exploitation, cryptography, reverse engineering, and forensics. Discover how to configure CAI for competitive hacking scenarios and maximize your CTF performance with intelligent automation. |  |
| **Annex 1**: CAI 0.5.x release  | Introduce version 0.5 of `CAI` including new multi-agent functionality, new commands such as `/history`, `/compact`, `/graph` or `/memory` and a case study showing how `CAI` found a critical security flaw in OT heap pumps spread around the world.  |  [![Watch the video](https://img.youtube.com/vi/OPFH0ANUMMw/0.jpg)](https://www.youtube.com/watch?v=OPFH0ANUMMw) | [![Watch the video](https://img.youtube.com/vi/Q8AI4E4gH8k/0.jpg)](https://www.youtube.com/watch?v=Q8AI4E4gH8k) |
| **Annex 2**: CAI 0.4.x release and alias0  | Introducing version 0.4 of `CAI` with *streaming* and improved MCP support. We also introduce `alias0`, the Privacy-First Cybersecurity AI, a Model-of-Models Intelligence that implements a Privacy-by-Design architecture and obtains state-of-the-art results in cybersecurity benchmarks. |  [![Watch the video](https://img.youtube.com/vi/NZjzfnvAZcc/0.jpg)](https://www.youtube.com/watch?v=NZjzfnvAZcc) |  |
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
echo -e 'OPENAI_API_KEY="sk-1234"\nANTHROPIC_API_KEY=""\nOLLAMA=""\nPROMPT_TOOLKIT_NO_CPR=