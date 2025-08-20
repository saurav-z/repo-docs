# Omnara: Your Mobile Command Center for AI Agents 🚀

**Take control of your AI workforce with Omnara, the platform that lets you monitor, interact, and guide your AI agents from anywhere.**  ([Back to the Omnara Repository](https://github.com/omnara-ai/omnara))

[![PyPI version](https://badge.fury.io/py/omnara.svg)](https://badge.fury.io/py/omnara)
[![Downloads](https://pepy.tech/badge/omnara)](https://pepy.tech/project/omnara)
[![Python Versions](https://img.shields.io/pypi/pyversions/omnara.svg)](https://pypi.org/project/omnara/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub stars](https://img.shields.io/github/stars/omnara-ai/omnara?style=social)](https://github.com/omnara-ai/omnara)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

<div align="center">
  [📱 **Download iOS App**](https://apps.apple.com/us/app/omnara-ai-command-center/id6748426727) • [🌐 **Try Web Dashboard**](https://omnara.ai) • [🎥 **See a Demo**](https://www.loom.com/share/03d30efcf8e44035af03cbfebf840c73?sid=1c209c04-8a4c-4dd6-8c92-735c399886a6) • [⭐ **Star on GitHub**](https://github.com/omnara-ai/omnara)
</div>

---

## Key Features of Omnara

*   **Real-Time Monitoring:** Observe every action your AI agents take.
*   **Interactive Q&A:**  Provide instant feedback and guidance when your agents need it.
*   **Mobile-First Design:**  Manage your AI from your phone, tablet, or desktop.
*   **Smart Notifications:**  Receive alerts only when your input is required.
*   **Unified Dashboard:**  Monitor all your AI agents in one central interface.

### See Omnara in Action
![Mobile Notifications](./docs/assets/iNotifications-Stack.gif)
![Agent Activity Feed](./docs/assets/Mobile-app-showcase.gif)

---

## Why Choose Omnara?

Omnara solves the common pain points of working with AI agents, providing:

*   **Visibility:**  No more guessing what your AI is doing.
*   **Control:** Guide your agents in real-time.
*   **Productivity:** Reduce wasted time and maximize output.
*   **Accessibility:** Manage your AI from anywhere.

---

## Architecture Overview

Omnara offers a unified platform for monitoring and controlling your AI agents.  The system is built on a robust architecture utilizing a variety of technologies:

```mermaid
graph TB
    subgraph "Your AI Agents"
        A[🤖 AI Agents<br/>Claude Code, Cursor, etc.]
    end

    subgraph "Omnara Platform"
        API[🌐 API Server]
        DB[(📊 PostgreSQL)]
        NOTIFY[🔔 Notification Service<br/>Push/Email/SMS]
    end

    subgraph "Your Devices"
        M[📱 Mobile App]
        W[💻 Web Dashboard]
    end

    A -->|Send updates| API
    API -->|Store data| DB
    API -->|Trigger notifications| NOTIFY
    NOTIFY -->|Alert users| M
    DB -->|Real-time sync| M
    DB -->|Real-time sync| W
    M -->|User responses| API
    W -->|User responses| API
    API -->|Deliver feedback| A

    style A fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    style API fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style DB fill:#ffccbc,stroke:#d84315,stroke-width:2px
    style NOTIFY fill:#fff59d,stroke:#f57f17,stroke-width:2px
    style M fill:#f8bbd0,stroke:#c2185b,stroke-width:3px
    style W fill:#f8bbd0,stroke:#c2185b,stroke-width:3px
```

---

## How Omnara Works

1.  **Connect Your Agent:** Integrate your AI agents using the Omnara SDK or a wrapper.
2.  **Get Real-Time Updates:** Monitor agent progress in real-time.
3.  **Respond Instantly:** Provide input and guidance from any location.

---

## Two Ways to Use Omnara

| Mode                     | Setup                                   | How It Works                                                                 |
| ------------------------ | --------------------------------------- | ---------------------------------------------------------------------------- |
| **Real-Time Monitoring** | `omnara` or `uv run omnara`             | Monitor your AI session, forwards updates to Omnara.                         |
| **Remote Launch**        | `omnara serve` or `uv run omnara serve` | Launch and control agents remotely from your mobile device via MCP. |

---

## Technical Stack

*   **Backend:** FastAPI with read/write servers
*   **Frontend:** React (Web) + React Native (Mobile)
*   **Protocol:** Model Context Protocol (MCP) & REST API
*   **Database:** PostgreSQL with SQLAlchemy ORM
*   **Auth:** Dual JWT system (Supabase for users, custom for agents)

---

## Quick Start

### Option 1: Monitor Your AI Sessions

1.  **Install Omnara:**
    ```bash
    # Using pip
    pip install omnara

    # Using uv (faster)
    uv tool install omnara
    ```
2.  **Start Monitoring:**
    ```bash
    # If installed with pip
    omnara

    # If installed with uv
    omnara
    ```
3.  **Authenticate:** Open the automatically launched browser window.
4.  **View:** Access the Omnara dashboard to monitor your agent's activity.

### Option 2: Launch Agents Remotely

1.  **Start the server** on your computer:
    ```bash
    # Using pip
    pip install omnara
    omnara serve

    # Using uv (faster)
    uv tool install omnara
    omnara serve
    ```
2.  **Configure Agent:** Set up your agent in the mobile app with the provided webhook URL.
3.  **Launch:** Initiate and manage agents remotely from anywhere.

---

### For Developers

<details>
<summary><b>🛠️ Development Setup</b></summary>

**Prerequisites:** Docker, Python 3.10+, Node.js

**Quick Start:**
```bash
git clone https://github.com/omnara-ai/omnara
cd omnara
cp .env.example .env
python scripts/generate_jwt_keys.py
./dev-start.sh  # Starts everything automatically
```

**Stop services:** `./dev-stop.sh`

For detailed setup instructions, manual configuration, and contribution guidelines, see our [Contributing Guide](CONTRIBUTING.md).

</details>

---

## Advanced Usage

> **Note:** Most users should use the `omnara` or `omnara serve` commands. The following are for advanced use cases.

### Method 1: Direct Wrapper Script

Run the monitoring wrapper directly (what `omnara` does under the hood):

```bash
# Basic usage
python -m integrations.cli_wrappers.claude_code.claude_wrapper_v3 --api-key YOUR_API_KEY

# With git diff tracking
python -m integrations.cli_wrappers.claude_code.claude_wrapper_v3 --api-key YOUR_API_KEY --git-diff

# Custom API endpoint (for self-hosted)
python -m integrations.cli_wrappers.claude_code.claude_wrapper_v3 --api-key YOUR_API_KEY --base-url https://your-server.com
```

### Method 2: Manual MCP Configuration

For custom MCP setups, you can configure manually:

```json
{
  "mcpServers": {
    "omnara": {
      "command": "pipx",
      "args": ["run", "--no-cache", "omnara", "mcp", "--api-key", "YOUR_API_KEY"]
    }
  }
}
```

### Method 3: Python SDK
```python
from omnara import OmnaraClient
import uuid

client = OmnaraClient(api_key="your-api-key")
instance_id = str(uuid.uuid4())

# Log progress and check for user feedback
response = client.send_message(
    agent_type="claude-code",
    content="Analyzing codebase structure",
    agent_instance_id=instance_id,
    requires_user_input=False
)

# Ask for user input when needed
answer = client.send_message(
    content="Should I refactor this legacy module?",
    agent_instance_id=instance_id,
    requires_user_input=True
)
```

### Method 4: REST API
```bash
curl -X POST https://api.omnara.ai/api/v1/messages/agent \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content": "Starting deployment process", "agent_type": "claude-code", "requires_user_input": false}'
```

---

## Pricing

| Plan          | Price    | Features                                   |
| ------------- | -------- | ------------------------------------------ |
| **Free**      | $0/mo    | 10 agents/month, Core features              |
| **Pro**       | $9/mo    | Unlimited agents, Priority support          |
| **Enterprise** | [Contact Us](https://cal.com/ishaan-sehgal-8kc22w/omnara-demo) | Teams, SSO, Custom integrations |

---

## Contributing

We welcome contributions!  See our [Contributing Guide](CONTRIBUTING.md) for more details.

### Development Commands
```bash
make lint       # Run code quality checks
make format     # Auto-format code
make test       # Run test suite
./dev-start.sh  # Start development servers
```

---

## Support

*   💬 [GitHub Discussions](https://github.com/omnara-ai/omnara/discussions)
*   🐛 [Report Issues](https://github.com/omnara-ai/omnara/issues)
*   📧 [Email Support](mailto:ishaan@omnara.com)
*   📖 Documentation (coming soon)

---

## License

Omnara is licensed under the [Apache 2.0 License](LICENSE).

---

<div align="center">
  **Built with ❤️ by the Omnara team**
  <br>
  [Website](https://omnara.ai) • [Twitter](https://twitter.com/omnaraai) • [LinkedIn](https://linkedin.com/company/omnara)
</div>