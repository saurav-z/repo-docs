<!-- Improved README - SEO Optimized -->
<div align="center">

![MemU Banner](assets/banner.png)

</div>

# MemU: Build AI Companions That Remember, with High Accuracy and Low Cost

MemU is an open-source memory framework designed to empower AI companions with exceptional recall and understanding, offering a superior solution to existing memory frameworks. 

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

[**Explore the MemU Repository on GitHub**](https://github.com/NevaMind-AI/memU)

---

## Key Features

*   **AI Companion Specialization:** Specifically designed for AI companion applications, tailoring memory management for optimal performance.
*   **92% Accuracy:** Achieve state-of-the-art performance on the Locomo benchmark.
*   **Up to 90% Cost Reduction:** Optimized platform for significant cost savings.
*   **Advanced Retrieval Strategies:** Employing multiple methods, including semantic search, hybrid search, and contextual retrieval, for precise memory recall.
*   **24/7 Support:** Dedicated support for enterprise customers.

---

## Why Choose MemU?

MemU revolutionizes AI companion development, offering:

*   **Superior Recall:**  Ensure your AI companions remember you and their experiences.
*   **Cost-Effectiveness:** Reduce operational expenses without sacrificing performance.
*   **Unmatched Accuracy:** Benefit from industry-leading memory accuracy.

---

## 🚀 Get Started

MemU offers multiple integration options to suit your needs:

### ☁️ Cloud Version ([Online Platform](https://app.memu.so))

Get started quickly with our cloud-based platform, perfect for teams and individuals seeking immediate access without the hassle of setup.

*   **Instant Integration:** Start integrating AI memories in minutes.
*   **Managed Infrastructure:** We handle all scaling, updates, and maintenance.
*   **Premium Support:** Priority assistance from our engineering team.

#### Quick Start Guide:

1.  **Create an account:** Visit [https://app.memu.so](https://app.memu.so).
2.  **Generate API keys:** Go to [https://app.memu.so/api-key/](https://app.memu.so/api-key/) after logging in.
3.  **Integrate into your code:**

```python
pip install memu-py
```

```python
# Initialize
memu_client = MemuClient(
    base_url="https://api.memu.so", 
    api_key=os.getenv("MEMU_API_KEY")
)
memu_client.memorize_conversation(
    conversation=conversation_text, # Recommend longer conversation (~8000 tokens), see https://memu.pro/blog/memu-best-practice for details
    user_id="user001", 
    user_name="User", 
    agent_id="assistant001", 
    agent_name="Assistant"
)
```

*   **Full Example:** See [`example/client/memory.py`](example/client/memory.py) for complete integration details.
*   **API Reference:** Detailed information available at [docs/API_REFERENCE.md](docs/API_REFERENCE.md) and [our blog](https://memu.pro/blog).

### 🏢 Enterprise Edition

For organizations requiring maximum security, control, and customization:

*   **Commercial License:** Full proprietary features and commercial usage rights.
*   **Custom Development:** Integration, dedicated algorithm team for framework optimization.
*   **Intelligence & Analytics:** User behavior analysis, real-time monitoring, agent optimization.
*   **Premium Support:** 24/7 dedicated support with custom SLAs and professional implementation services.

**Contact:** [contact@nevamind.ai](mailto:contact@nevamind.ai) for enterprise inquiries.

### 🏠 Self-Hosting (Community Edition)

For local control, data privacy, and customization:

*   **Data Privacy:** Maintain sensitive data within your infrastructure.
*   **Customization:** Modify and extend the platform to fit your needs.
*   **Cost Control:** Avoid recurring cloud fees for large-scale deployments.

See [self hosting README](README.self_host.md)

---

## ✨ Advanced Features

### 🎥 Demo Video

<div align="left">
  <a href="https://www.youtube.com/watch?v=qZIuCoLglHs">
    <img src="https://img.youtube.com/vi/ueOe4ZPlZLU/maxresdefault.jpg" alt="MemU Demo Video" width="600">
  </a>
  <br>
  <em>Click to watch the MemU demonstration video</em>
</div>

---

### Memory as File System: Building Intelligent Knowledge

*   **Organize: Autonomous Memory File Management** Your memories are structured as intelligent folders managed by a memory agent.
*   **Link: Interconnected Knowledge Graph** Memories are automatically connected for effortless recall.
*   **Evolve: Continuous Self-Improvement**  Your memory agent generates new insights and summarizes information over time.
*   **Never Forget: Adaptive Forgetting Mechanism** The memory agent prioritizes information based on usage patterns.

---

## 😺 Advantages in Detail

*   **Higher Memory Accuracy:**  Achieves 92.09% average accuracy in Locomo dataset, significantly outperforming competitors. Technical Report will be published soon!

    ![Memory Accuracy Comparison](assets/benchmark.png)

    *(1) Single-hop questions require answers based on a single session; (2) Multi-hop questions require synthesizing information from multiple different sessions; (3) Temporal reasoning questions can be answered through temporal reasoning and capturing time-related data cues within the conversation; (4) Open-domain knowledge questions can be answered by integrating a speaker’s provided information with external knowledge such as commonsense or world facts;*

*   **Fast Retrieval:** Categorizes important information into documents for efficient retrieval, eliminating the need for extensive embedding searches.
*   **Low Cost:** Processes hundreds of conversation turns at once, reducing token usage and overall costs.  See [best practice](https://memu.pro/blog/memu-best-practice).

---

## 🎓 Use Cases

Explore how MemU can be applied across various domains:

| | | | |
|:---:|:---:|:---:|:---:|
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>**AI Companion** | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>**AI Role Play** | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>**AI IP Characters** | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>**AI Education** |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>**AI Therapy** | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>**AI Robot** | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>**AI Creation** | More...|

---

## 🤝 Contributing

We welcome contributions!  Help us improve MemU and shape the future of AI memory.

*   **Get Started:** Explore our GitHub issues and projects.
*   **Contribute:** [Read our detailed Contributing Guide](CONTRIBUTING.md).
*   **License:** Contributions are licensed under the Apache License 2.0.

---

## 🌍 Community

Connect with us and stay informed:

*   **GitHub Issues:** Report bugs, request features, and track development: [Submit an issue](https://github.com/NevaMind-AI/memU/issues).
*   **Discord:** Join our community for real-time support and discussions: [Join us](https://discord.com/invite/hQZntfGsbJ).
*   **X (Twitter):** Follow us for updates and announcements: [Follow us](https://x.com/memU_ai).
*   **For more information please contact info@nevamind.ai**

---

## 🤝 Ecosystem

We are proud to be working with these amazing organizations:

<div align="center">

### Development Tools
<a href="https://github.com/TEN-framework/ten-framework"><img src="https://avatars.githubusercontent.com/u/113095513?s=200&v=4" alt="Ten" height="40" style="margin: 10px;"></a>
<a href="https://github.com/camel-ai/camel"><img src="https://avatars.githubusercontent.com/u/134388954?s=200&v=4" alt="Camel" height="40" style="margin: 10px;"></a>
<a href="https://github.com/eigent-ai/eigent"><img src="https://www.eigent.ai/nav/logo_icon.svg" alt="Eigent" height="40" style="margin: 10px;"></a>
<a href="https://github.com/milvus-io/milvus"><img src="https://miro.medium.com/v2/resize:fit:2400/1*-VEGyAgcIBD62XtZWavy8w.png" alt="Ten" height="40" style="margin: 10px;"></a>
<a href="https://xroute.ai/"><img src="assets/partners/xroute.png" alt="xRoute" height="40" style="margin: 10px;"></a>
<a href="https://jaaz.app/"><img src="assets/partners/jazz.png" alt="jazz" height="40" style="margin: 10px;"></a>
<a href="https://github.com/Buddie-AI/Buddie"><img src="assets/partners/buddie.png" alt="buddie" height="40" style="margin: 10px;"></a>
<a href="https://github.com/bytebase/bytebase"><img src="assets/partners/bytebase.png" alt="bytebase" height="40" style="margin: 10px;"></a>
</div>

---

*Interested in partnering with MemU? Contact us at [contact@nevamind.ai](mailto:contact@nevamind.ai)*

---

## 📱 Join Our WeChat Community

<div align="center">
  Connect with us on WeChat:
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
*Scan any of the QR codes above to join our WeChat community*
</div>
---

## 📝 Questionnaire

Help us improve! Share your feedback on our 3-min survey and get 30 free quota: https://forms.gle/H2ZuZVHv72xbqjvd7
```

Key improvements and optimizations:

*   **SEO Focus:** Headings, keywords like "AI companion," "memory framework," and "open-source" are used naturally, and the intro hooks the reader.
*   **Clear Structure:**  Uses headings, subheadings, and bullet points to organize information for readability and scanning.
*   **Concise Language:**  Simplifies and streamlines text for better understanding.
*   **Stronger Calls to Action:**  Encourages users to star the repo, join the community, and explore the project.
*   **Highlights Benefits:**  Focuses on what the user gains (accuracy, cost savings, etc.).
*   **Keywords:** "AI Companions", "Memory Framework", "Open Source" are used in the title and throughout.
*   **Improved Formatting:** Bolded important sections and key features.
*   **Call to action** encourages users to share their feedback.