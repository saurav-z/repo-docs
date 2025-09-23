<div align="center">

![MemU Banner](assets/banner.png)

</div>

# MemU: The Ultimate Memory Framework for AI Companions

**Build AI companions that remember with MemU, the open-source memory framework engineered for high accuracy, fast retrieval, and cost-effectiveness.  [Explore the MemU Repo on GitHub](https://github.com/NevaMind-AI/memU)**

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

## Key Features & Benefits

*   ✅ **AI Companion Specialization:** Designed specifically for AI companion applications.
*   ✅ **High Accuracy:** Achieve state-of-the-art memory recall with a 92% accuracy score on the Locomo benchmark.
*   ✅ **Cost-Effective:** Reduce operational costs by up to 90% through optimized online platform integration.
*   ✅ **Advanced Retrieval Strategies:** Leverage multiple methods including semantic search, hybrid search, and contextual retrieval for optimal results.
*   ✅ **24/7 Support (Enterprise):** Dedicated support for enterprise clients ensures seamless integration and optimal performance.
*   ✅ **Intelligent Memory Management:**  Autonomous memory file management with interconnected knowledge graphs that evolve over time.
*   ✅ **Adaptive Forgetting Mechanism:**  Prioritize relevant information and optimize memory usage.

---

## Why Choose MemU?

MemU empowers you to create AI companions that offer truly personalized and engaging interactions.  Its unique features enable:

*   **Enhanced User Experience:** AI companions remember past conversations and adapt to user preferences, creating a more natural and engaging interaction.
*   **Improved Knowledge Retention:**  MemU's advanced memory management ensures that AI companions can access and utilize relevant information effectively.
*   **Cost Optimization:**  Benefit from significant cost savings with MemU's efficient design and optimized infrastructure.

---

## Get Started with MemU

MemU offers flexible integration options to suit your needs:

### ☁️ **Cloud Version (Online Platform)**

The fastest way to integrate with memU, ideal for teams and individuals.

*   **Instant Access:** Integrate AI memories in minutes through our easy-to-use platform.
*   **Managed Infrastructure:** We handle scaling, updates, and maintenance.
*   **Premium Support:** Receive priority assistance from our engineering team.

#### Quick Integration Steps

**1. Create an Account:** Sign up at [https://app.memu.so](https://app.memu.so).

**2. Generate API Keys:** Obtain your API keys at [https://app.memu.so/api-key/](https://app.memu.so/api-key/).

**3. Integrate into your code (Python Example):**

```python
pip install memu-py

# Example usage
from memu import MemuClient
import os

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

**4. Explore the API:** See [API reference](docs/API_REFERENCE.md) or [our blog](https://memu.pro/blog) for more details.

**5. Complete integration:**  See [`example/client/memory.py`](example/client/memory.py) for complete integration details.

### 🏢 **Enterprise Edition**

For organizations requiring maximum security, customization, and control:

*   **Commercial License:** Full proprietary features and commercial usage rights.
*   **Custom Development:** SSO/RBAC integration and scenario-specific optimization.
*   **Intelligence & Analytics:** User behavior analysis and automated agent optimization.
*   **Premium Support:** 24/7 dedicated support and professional implementation services.

**Contact:** [contact@nevamind.ai](mailto:contact@nevamind.ai) for Enterprise Inquiries.

### 🏠 **Self-Hosting (Community Edition)**

For developers who prefer local control and data privacy:

*   **Data Privacy:** Keep data within your infrastructure.
*   **Customization:** Modify and extend the platform to fit your needs.
*   **Cost Control:** Avoid cloud fees.

See [self hosting README](README.self_host.md) for setup instructions.

---

## ✨ Explore the MemU Advantage

### 🎥 Demo Video

<div align="left">
  <a href="https://www.youtube.com/watch?v=qZIuCoLglHs">
    <img src="https://img.youtube.com/vi/ueOe4ZPlZLU/maxresdefault.jpg" alt="MemU Demo Video" width="600">
  </a>
  <br>
  <em>Click to watch the MemU demonstration video</em>
</div>

### Memory as a File System

*   **Organize:** Memories are structured as intelligent folders managed by a memory agent.
*   **Link:** Creates connections between related memories, building a network of hyperlinked documents.
*   **Evolve:**  Generates new insights and creates summaries through self-reflection.
*   **Never Forget:** Adaptive Forgetting Mechanism: The memory agent automatically prioritizes information based on usage patterns.

---

## 😺 Advantages: Benchmarks and Performance

###  Higher Memory Accuracy

MemU achieves 92.09% average accuracy in Locomo dataset across all reasoning tasks, significantly outperforming competitors.

![Memory Accuracy Comparison](assets/benchmark.png)
*(1) Single-hop questions require answers based on a single session; (2) Multi-hop questions require synthesizing information from multiple different sessions; (3) Temporal reasoning questions can be answered through temporal reasoning and capturing time-related data cues within the conversation; (4) Open-domain knowledge questions can be answered by integrating a speaker’s provided information with external knowledge such as commonsense or world facts;*

### Fast Retrieval

We categorize important information into documents, and during retrieval, we only need to find the relevant document content, eliminating the need for extensive embedding searches for fragmented sentences.

### Low cost

We can process hundreds of conversation turns at once, eliminating the need for developers to repeatedly call memory functions, thus saving users from wasting tokens on multiple memory operations. See [best practice](https://memu.pro/blog/memu-best-practice).

---

## 🎓 Use Cases: Real-World Applications

| Use Case         | Description                                  |
| ---------------- | -------------------------------------------- |
| AI Companion     | Enhance interactions and remember users.    |
| AI Role Play     | Create immersive and personalized experiences. |
| AI IP Characters | Bring AI characters to life with memory.     |
| AI Education     | Improve learning through memory retention.   |
| AI Therapy       | Provide personalized support.               |
| AI Robot         | Enable robots to learn and adapt.           |
| AI Creation      | Foster creativity and inspiration.          |
| More...          | Explore the possibilities!                     |

<div align="center">
  <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>**AI Companion** | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>**AI Role Play** | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>**AI IP Characters** | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>**AI Education** | <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>**AI Therapy** | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>**AI Robot** | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>**AI Creation** |
</div>

---

## 🤝 Contributing & Community

Help us build the future of AI memory!

### Contributing

*   **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**
*   By contributing, you agree to license your contributions under the **Apache License 2.0**.

### Community

*   **GitHub Issues:** Report bugs, request features, and track development: [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support and chat with the community: [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Follow for updates and announcements: [Follow us](https://x.com/memU_ai)
*   **For more information:** info@nevamind.ai

---

## 🤝 Ecosystem

We're proud to work with these amazing organizations:

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

Connect with us on WeChat:

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
</div>

---

## 📝 Questionnaire

Help us improve MemU! Share your feedback on our 3-min survey and get 30 free quota：https://forms.gle/H2ZuZVHv72xbqjvd7