<div align="center">
    <img src="./media/logo_small.webp" alt="Spec Kit Logo"/>
    <h1>🌱 Spec Kit: Revolutionizing Software Development</h1>
    <h3><em>Build high-quality software faster with AI-powered Spec-Driven Development.</em></h3>
</div>

<p align="center">
    **Focus on *what* to build, not *how* to code.  Spec Kit leverages AI to transform specifications into working software.**
    <br>
    <a href="https://github.com/github/spec-kit">
        <img src="https://github.com/github/spec-kit/actions/workflows/release.yml/badge.svg" alt="Release Status">
    </a>
    <br>
    <a href="https://github.com/github/spec-kit">Go to the Spec Kit Repository</a>
</p>

---

## 🚀 Key Features of Spec Kit

*   ✅ **Intent-Driven Development:** Define *what* you want to build first, allowing the system to figure out the *how*.
*   ✅ **Executable Specifications:** Turn specifications into direct, working implementations.
*   ✅ **AI-Powered Automation:** Leverage the power of AI coding agents for faster development.
*   ✅ **Multi-Step Refinement:** Iterate and refine specifications for optimal results.
*   ✅ **Tech-Stack Agnostic:** Build applications using diverse technologies and frameworks.
*   ✅ **Flexible Development Phases:** Supports Greenfield, Creative Exploration, and Brownfield projects.

---

## ⚡️ Get Started with Spec-Driven Development

Spec Kit simplifies software development by focusing on specifications, enabling your AI coding agent to do the heavy lifting of implementation.

1.  **Install Specify:** Initialize your project with:

    ```bash
    uvx --from git+https://github.com/github/spec-kit.git specify init <PROJECT_NAME>
    ```

2.  **Create Your Spec:** Use the `/specify` command to describe your desired application functionality. Focus on the *what* and *why*.

    ```bash
    /specify Build an application that can help me organize my photos in separate photo albums. Albums are grouped by date and can be re-organized by dragging and dropping on the main page. Albums are never in other nested albums. Within each album, photos are previewed in a tile-like interface.
    ```

3.  **Define Your Implementation Plan:** Use the `/plan` command to specify the tech stack and architecture.

    ```bash
    /plan The application uses Vite with minimal number of libraries. Use vanilla HTML, CSS, and JavaScript as much as possible. Images are not uploaded anywhere and metadata is stored in a local SQLite database.
    ```

4.  **Break Down and Implement:** Use `/tasks` to generate tasks and then use your AI agent to implement features.

    For detailed instructions, consult our [comprehensive guide](./spec-driven.md).

---

## 📚 Core Philosophy of Spec-Driven Development

Spec-Driven Development is a structured approach that emphasizes:

*   **Intent-Driven Development:** Prioritizing specifications before implementation.
*   **Rich Specification Creation:** Utilizing guardrails and organizational principles.
*   **Multi-Step Refinement:** Refining specifications and implementations through iterative cycles.
*   **AI-Driven Implementation:** Utilizing advanced AI to interpret specifications and generate code.

---

## 🌟 Development Phases & Goals

| Phase                      | Focus                               | Key Activities                                                                                                                               |
| -------------------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **0-to-1 Development**      | Generate from scratch               | High-level requirements, specification generation, implementation planning, production-ready application creation.                                 |
| **Creative Exploration**   | Parallel implementations            | Exploring diverse solutions, supporting multiple technology stacks, experimenting with UX patterns.                                         |
| **Iterative Enhancement** | Brownfield modernization            | Iteratively adding features, modernizing legacy systems, adapting processes.                                                                  |

## 🎯 Experimental Goals

Spec Kit's research centers on:

*   **Technology Independence:** Building applications across varied tech stacks.
*   **Enterprise Constraints:** Developing mission-critical apps within organizational constraints.
*   **User-Centric Development:** Building apps for diverse user cohorts.
*   **Creative & Iterative Processes:** Refining iterative feature development and supporting upgrades.

---

## 🔧 Prerequisites

*   **Operating System:** Linux/macOS (or WSL2 on Windows)
*   **AI Coding Agent:** [Claude Code](https://www.anthropic.com/claude-code), [GitHub Copilot](https://code.visualstudio.com/), or [Gemini CLI](https://github.com/google-gemini/gemini-cli)
*   **Package Manager:** [uv](https://docs.astral.sh/uv/)
*   **Programming Language:** [Python 3.11+](https://www.python.org/downloads/)
*   **Version Control:** [Git](https://git-scm.com/downloads)

---

## 📖 Learn More

*   **[Comprehensive Guide to Spec-Driven Development](./spec-driven.md)** - Delve deeper into the methodology.
*   **[Detailed Step-by-Step Process](#detailed-process)** - Get hands-on with implementation.

---

## 🔍 Troubleshooting

### Git Credential Manager on Linux

If you encounter Git authentication issues on Linux:

```bash
#!/usr/bin/env bash
set -e
echo "Downloading Git Credential Manager v2.6.1..."
wget https://github.com/git-ecosystem/git-credential-manager/releases/download/v2.6.1/gcm-linux_amd64.2.6.1.deb
echo "Installing Git Credential Manager..."
sudo dpkg -i gcm-linux_amd64.2.6.1.deb
echo "Configuring Git to use GCM..."
git config --global credential.helper manager
echo "Cleaning up..."
rm gcm-linux_amd64.2.6.1.deb
```

---

## 👥 Maintainers

*   Den Delimarsky ([@localden](https://github.com/localden))
*   John Lam ([@jflam](https://github.com/jflam))

---

## 💬 Support

For assistance, please open a [GitHub issue](https://github.com/github/spec-kit/issues/new). We welcome bug reports, feature requests, and any questions regarding Spec-Driven Development.

---

## 🙏 Acknowledgements

This project builds upon the work and research of [John Lam](https://github.com/jflam).

---

## 📄 License

This project is licensed under the [MIT license](./LICENSE).