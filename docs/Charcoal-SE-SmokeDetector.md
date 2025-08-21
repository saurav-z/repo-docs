# SmokeDetector: The Headless Chatbot Combating Spam on Stack Exchange

This powerful chatbot, built for the Stack Exchange community, automatically identifies and reports spam, keeping chatrooms clean and informative.

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

SmokeDetector is a headless chatbot designed to monitor the Stack Exchange network for spam and other unwanted content. It leverages the Stack Exchange API and chat features to identify and report potentially harmful content in real-time.

**Key Features:**

*   **Real-time Spam Detection:** Monitors the Stack Exchange real-time feed for suspicious content.
*   **Automated Reporting:** Posts detected spam to chatrooms for review and action.
*   **Uses ChatExchange:** Integrates with ChatExchange for seamless chat interaction.
*   **Stack Exchange API Integration:** Utilizes the Stack Exchange API to access and analyze content.
*   **Multiple Setup Options:** Supports setup via shell, virtual environment, and Docker.

**How It Works:**

SmokeDetector utilizes [ChatExchange](https://github.com/Manishearth/ChatExchange) to connect to Stack Exchange chatrooms. It monitors the Stack Exchange real-time feed, examines questions, and accesses answers via the Stack Exchange API. Once potential spam is detected, the bot automatically posts it to designated chatrooms, providing community members with an easy way to address spam and ensure site quality.

**Getting Started:**

Comprehensive documentation, including detailed setup and usage instructions, is available in the [wiki](https://charcoal-se.org/smokey).

*   [Setting Up and Running SmokeDetector](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector)

**Setup Options:**

*   **Basic Setup:** Provides a simple command-line setup guide.
*   **Virtual Environment Setup:** Instructions for setting up SmokeDetector within a Python virtual environment to isolate dependencies.
*   **Docker Setup:** Guides users through containerizing SmokeDetector using Docker for enhanced isolation and portability.

**Requirements:**

*   Python versions that are in the [supported phase of the Python life cycle](https://devguide.python.org/versions/)
*   Git 1.8 or higher is recommended.

**Blacklist Removal:**

If you represent a website listed on the blacklist and would like to request its removal, please refer to the [Process for Blacklist Removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal) for detailed instructions.

**License:**

SmokeDetector is licensed under the terms of either the [Apache License, Version 2.0](LICENSE-APACHE) or the [MIT License](LICENSE-MIT).

**Contribute:**

Find out more about this project on its [GitHub repository](https://github.com/Charcoal-SE/SmokeDetector).