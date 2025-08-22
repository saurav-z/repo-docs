<!-- Improved README for MemU - Optimized for SEO -->

<div align="center">

![MemU Banner](assets/banner.png)

</div>

# MemU: The Leading Memory Framework for AI Companions 🧠

**Build AI companions that remember everything with MemU, the open-source memory framework designed for high accuracy, fast retrieval, and low cost.** [Explore the original repo](https://github.com/NevaMind-AI/memU)

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

---

## Key Features of MemU

*   ✅ **AI Companion Specialization:** Tailored for AI companion applications.
*   ✅ **92% Accuracy:** Achieve state-of-the-art performance on the Locomo benchmark.
*   ✅ **Up to 90% Cost Reduction:** Optimized for cost-effective AI memory solutions.
*   ✅ **Advanced Retrieval Strategies:** Utilize semantic search, hybrid search, and contextual retrieval.
*   ✅ **24/7 Support:** Dedicated support for enterprise customers.

---

## Why Choose MemU?

MemU is designed to be the ultimate memory solution for AI companions, offering:

*   **Unmatched Accuracy:** Benefit from superior memory recall and understanding.
*   **Blazing-Fast Retrieval:** Quickly access the information your AI needs.
*   **Cost-Effective Solutions:** Optimize your AI companion's memory without breaking the bank.

## Get Started with MemU

### ☁️ Cloud Version ([Online Platform](https://app.memu.so))

The fastest way to integrate your application with memU. Perfect for teams and individuals who want immediate access without setup complexity. We host the models, APIs, and cloud storage, ensuring your application gets the best quality AI memory.

-   **Instant Access** - Start integrating AI memories in minutes
-   **Managed Infrastructure** - We handle scaling, updates, and maintenance for optimal memory quality
-   **Premium Support** - Subscribe and get priority assistance from our engineering team

**Step 1:** Create account

Create account on https://app.memu.so

Then, go to https://app.memu.so/api-key/ for generating api-keys.

**Step 2:** Add three lines to your code
```python
pip install memu-py

# Example usage
from memu import MemuClient
```

**Step 3:** Quick Start
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
Check [API reference](docs/API_REFERENCE.md) or [our blog](https://memu.pro/blog) for more details. 

📖 **See [`example/client/memory.py`](example/client/memory.py) for complete integration details**

✨ **That's it!** MemU remembers everything and helps your AI learn from past conversations.

### 🏢 Enterprise Edition

For organizations requiring maximum security, customization, control and best quality:

-   **Commercial License** - Full proprietary features, commercial usage rights, white-labeling options
-   **Custom Development** - SSO/RBAC integration, dedicated algorithm team for scenario-specific framework optimization
-   **Intelligence & Analytics** - User behavior analysis, real-time production monitoring, automated agent optimization
-   **Premium Support** - 24/7 dedicated support, custom SLAs, professional implementation services

📧 **Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 Self-Hosting (Community Edition)

For users and developers who prefer local control, data privacy, or customization:

*   **Data Privacy** - Keep sensitive data within your infrastructure
*   **Customization** - Modify and extend the platform to fit your needs
*   **Cost Control** - Avoid recurring cloud fees for large-scale deployments

See [self hosting README](README.self_host.md)

---

## ✨ Advanced Memory Features

### 🎥 Demo Video

<div align="left">
  <a href="https://www.youtube.com/watch?v=qZIuCoLglHs">
    <img src="https://img.youtube.com/vi/ueOe4ZPlZLU/maxresdefault.jpg" alt="MemU Demo Video" width="600">
  </a>
  <br>
  <em>Click to watch the MemU demonstration video</em>
</div>

---

### Memory as a File System

MemU's innovative approach to memory management allows for:

*   **Organize**: Autonomous memory file management. Memories are structured as intelligent folders managed by a memory agent.
*   **Link**: Interconnected knowledge graph. Connections between memories build a rich network of hyperlinked documents.
*   **Evolve**: Continuous self-improvement. The memory agent generates new insights by analyzing existing memories.
*   **Never Forget**: Adaptive forgetting mechanism. Prioritizes information based on usage patterns for personalized information hierarchy.

---

## Performance & Advantages

### 🥇 Superior Accuracy

MemU achieves 92.09% average accuracy in the Locomo dataset, significantly outperforming competitors.

![Memory Accuracy Comparison](assets/benchmark.png)
<em>(1) Single-hop questions require answers based on a single session; (2) Multi-hop questions require synthesizing information from multiple different sessions; (3) Temporal reasoning questions can be answered through temporal reasoning and capturing time-related data cues within the conversation; (4) Open-domain knowledge questions can be answered by integrating a speaker’s provided information with external knowledge such as commonsense or world facts; </em>

### ⚡ Fast Retrieval

MemU categorizes important information into documents, eliminating the need for extensive embedding searches.

### 💰 Cost-Effective

Process hundreds of conversation turns at once, saving on token usage and operational costs.

---

## 🚀 Use Cases for MemU

MemU is a versatile framework with applications across a variety of domains:

| | | | |
|:---:|:---:|:---:|:---:|
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>**AI Companion** | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>**AI Role Play** | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>**AI IP Characters** | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>**AI Education** |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>**AI Therapy** | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>**AI Robot** | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>**AI Creation** | More...|
---

## 🤝 Contribute to MemU

Become a part of the MemU community and help shape the future of AI companions!

📋 **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

### **📄 License**

Contributions are licensed under the **Apache License 2.0**.

---

## 🌍 Community & Support

*   **GitHub Issues:** Report bugs, request features. [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support and connect with the community. [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Follow for updates. [Follow us](https://x.com/memU_ai)
*   **For more information please contact info@nevamind.ai**

---

## 🤝 Ecosystem & Partners

MemU works with awesome organizations:

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

## 📱 Stay Connected

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
*Scan any of the QR codes above to join our WeChat community*
</div>

---

## 📝 Help Us Improve

Share your feedback in our short survey for 30 free quotas! https://forms.gle/H2ZuZVHv72xbqjvd7
```

Key improvements and SEO optimizations:

*   **Clear Headline:** Focused on "Leading Memory Framework for AI Companions."
*   **Concise Hook:** Immediately highlights the core benefit.
*   **Keyword Optimization:** Uses relevant keywords like "AI companions," "memory framework," "open-source," "high accuracy," "fast retrieval," and "low cost" throughout the document.
*   **Structured Headings:** Uses H2 and H3 tags for better readability and SEO.
*   **Bulleted Key Features:** Clearly lists the core advantages.
*   **Call to Action:** Encourages users to "Explore the original repo".
*   **Strong Emphasis on Benefits:** Focuses on the *why* of using MemU.
*   **Clear Instructions:**  The "Get Started" section is clearly defined with cloud, enterprise, and self-hosting options.
*   **Internal Links:**  Includes links back to the API reference, blog, and example code to improve user experience.
*   **Optimized Images:** Uses descriptive alt text for images.
*   **Concise Language:**  Streamlined descriptions to improve readability.
*   **Community Engagement:**  Encourages interaction and contribution.
*   **Use Cases:** Highlights different applications of the framework.
*   **Meta Description (Implied):**  The introduction and key features work as a concise summary that would be suitable as a meta description.

This improved README is more informative, easier to read, and optimized for search engines, making it more likely to attract users and contributors.