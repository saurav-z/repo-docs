# SmokeDetector: Real-time Spam Detection for Stack Exchange

**Keep your Stack Exchange communities clean with SmokeDetector, a headless chatbot that instantly identifies and reports spam in real-time.** Find the original project on GitHub: [Charcoal-SE/SmokeDetector](https://github.com/Charcoal-SE/SmokeDetector)

SmokeDetector analyzes questions from Stack Exchange's realtime tab, using ChatExchange and the Stack Exchange API to quickly flag and post suspected spam to chatrooms.

**Key Features:**

*   **Real-time Spam Detection:** Identifies and reports spam as it happens.
*   **Automated Chat Reporting:** Posts spam reports to designated chatrooms for moderator review.
*   **Stack Exchange Integration:**  Leverages the Stack Exchange API for comprehensive analysis.
*   **Flexible Deployment:**  Supports setup via git clone, virtual environments, and Docker containers for various operating systems and configurations.
*   **Customizable:** Easily configure the bot with a `config` file.

**Getting Started:**

Comprehensive documentation, including setup and running instructions, can be found in the [wiki](https://charcoal-se.org/smokey). Detailed instructions are available for setting up SmokeDetector using these methods:

*   **Basic Setup:** Install dependencies using pip, configure the `config` file, and run with `python3 nocrash.py` (recommended for continuous operation).
*   **Virtual Environment Setup:** Use a virtual environment to isolate dependencies.
*   **Docker Setup:** Run SmokeDetector in a Docker container for enhanced isolation and portability. Instructions are included for creating a container. Automate deployment with Docker Compose.

**Requirements:**

*   Python versions within the [supported phase of the Python life cycle](https://devguide.python.org/versions/)
*   Stack Exchange account
*   Git 1.8 or higher (Git 2.11+ recommended) for committing blacklist/watchlist changes.

**Other Information:**

*   **Blacklist Removal:**  See "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" for the process for removing a website from the blacklist.
*   **License:**  SmokeDetector is licensed under the Apache License 2.0 and MIT licenses.