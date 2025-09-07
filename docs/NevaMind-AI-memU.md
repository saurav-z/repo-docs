<div align="center">

![MemU Banner](assets/banner.png)

</div>

# MemU: Revolutionizing AI Companions with Next-Gen Memory 🧠

**MemU** is an open-source memory framework designed to empower AI companions with superior recall, speed, and cost-efficiency, making them more engaging and responsive.  [Explore the original repository](https://github.com/NevaMind-AI/memU) for the code.

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

## Key Features & Benefits

*   ✅ **AI Companion Specialization:** Specifically designed for AI companion applications.
*   ✅ **Exceptional Accuracy:** Achieve state-of-the-art performance, with 92% accuracy on the Locomo benchmark.
*   ✅ **Cost-Effective:** Reduce operational costs by up to 90% through optimized online platform.
*   ✅ **Advanced Retrieval Strategies:** Leverage multiple methods like semantic, hybrid, and contextual search for optimal memory recall.
*   ✅ **Cloud and Self-Hosting Options:** Choose the best deployment strategy for your needs.
*   ✅ **Comprehensive Support:** 24/7 support available for enterprise customers.

## Get Started with MemU

### ☁️ **Cloud Version (Quickest Integration)**

The fastest way to integrate your application with memU. Ideal for teams and individuals seeking immediate access without setup complexity.  

*   **Instant Access:** Integrate AI memories in minutes.
*   **Managed Infrastructure:** MemU handles scaling, updates, and maintenance.
*   **Premium Support:** Prioritized assistance from our engineering team is available.

#### Quick Start Guide:

**Step 1:** Create an account at [https://app.memu.so](https://app.memu.so) and generate an API key at [https://app.memu.so/api-key/](https://app.memu.so/api-key/).

**Step 2:** Install the Python package:

```bash
pip install memu-py
```

**Step 3:** Use the following code as an example:

```python
# Example usage
import os
from memu import MemuClient

# Initialize
memu_client = MemuClient(
    base_url="https://api.memu.so", 
    api_key=os.getenv("MEMU_API_KEY")
)
memu_client.memorize_conversation(
    conversation=conversation_text,  # Recommend longer conversation (~8000 tokens)
    user_id="user001", 
    user_name="User", 
    agent_id="assistant001", 
    agent_name="Assistant"
)
```

**For in-depth details, check our:**

*   [API Reference](docs/API_REFERENCE.md)
*   [Blog](https://memu.pro/blog)
*   [`example/client/memory.py`](example/client/memory.py) for complete integration details

### 🏢 **Enterprise Edition**

For maximum security, customization, and control:

*   **Commercial License:** Proprietary features and white-labeling options.
*   **Custom Development:** Integration, dedicated algorithm team optimization.
*   **Intelligence & Analytics:** User behavior analysis, real-time production monitoring.
*   **Premium Support:** 24/7 dedicated support, custom SLAs, and implementation services.

📧 **Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 **Self-Hosting (Community Edition)**

For local control, data privacy, and customization:

*   **Data Privacy:** Keep sensitive data within your infrastructure.
*   **Customization:** Modify and extend the platform to fit your needs.
*   **Cost Control:** Avoid recurring cloud fees for large-scale deployments.

See [self hosting README](README.self_host.md) for instructions.

## 🧠 Memory as a File System: Unleashing AI's Potential

MemU transforms how AI interacts with memory, enhancing recall and intelligence through:

*   **Organize:** Autonomous memory file management.
*   **Link:** Interconnected Knowledge Graph.
*   **Evolve:** Continuous self-improvement.
*   **Never Forget:** Adaptive Forgetting Mechanism.

## ⭐ Key Advantages of MemU

### Superior Accuracy
MemU excels, achieving 92.09% average accuracy in the Locomo dataset across all reasoning tasks, far surpassing competitors.

![Memory Accuracy Comparison](assets/benchmark.png)
*   (1) Single-hop questions; (2) Multi-hop questions; (3) Temporal reasoning; (4) Open-domain knowledge questions

### Fast Retrieval

We classify information into documents to quickly find relevant content during retrieval.

### Low Cost

The capability to process numerous conversation turns simultaneously eliminates the need for repeated memory function calls, therefore saving tokens. See [best practice](https://memu.pro/blog/memu-best-practice).

## 🚀 Use Cases

MemU powers a variety of applications:

| AI Companion | AI Role Play | AI IP Characters | AI Education |
|:---|:---|:---|:---|
|<img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200">|<img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200">|<img src="assets/usecase/ai_ip-0000.png" width="150" height="200">|<img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200">|
| AI Therapy | AI Robot | AI Creation | More... |
|<img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200">|<img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200">|<img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200">|

## 🤝 Contributing

We welcome contributions!  Improve MemU by exploring our [GitHub issues](https://github.com/NevaMind-AI/memU/issues) and [projects](https://github.com/NevaMind-AI/memU/projects).

📋 **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

### **📄 License**

Contributions are licensed under the **Apache License 2.0**.

---

## 🌍 Community

Connect with us and stay updated:

*   **GitHub Issues:** Report bugs, request features, and track development. [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support and chat with the community. [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Follow for updates, AI insights, and announcements. [Follow us](https://x.com/memU_ai)
*   **For more information please contact info@nevamind.ai**

## 🤝 Ecosystem

We collaborate with amazing organizations:

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

Connect with us on WeChat for the latest updates, community discussions, and exclusive content:

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
</div>

*Scan either of the QR codes above to join our WeChat community*

---
## 💡 Questionnaire

Help us improve! Share your feedback on our 3-min survey and get 30 free quota：https://forms.gle/H2ZuZVHv72xbqjvd7
```
Key improvements and explanations:

*   **SEO Optimization:**
    *   Focused on relevant keywords (AI companion, memory framework, AI, open source).
    *   Used descriptive headings (H1, H2, H3) for better structure and SEO.
    *   Included a compelling introductory sentence/hook.
    *   Increased the use of bulleted lists.
    *   Added partner links
*   **Summarization and Clarity:**
    *   Condensed the text while retaining key information.
    *   Simplified language for broader understanding.
    *   Prioritized the most important features and benefits.
*   **Structure and Readability:**
    *   Organized content with clear headings and subheadings.
    *   Emphasized important points with bold text.
    *   Provided clear calls to action (e.g., "Get Started," "Join Us").
*   **Conciseness:**  Removed redundant phrases and streamlined the presentation.
*   **Call to Action:** The primary CTA is clear and provides value to the user.
*   **Markdown Formatting:** Ensured proper markdown formatting for easy readability on GitHub.
*   **Links:** Added links where appropriate.