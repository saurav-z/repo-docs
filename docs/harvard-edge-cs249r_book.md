# Machine Learning Systems: Build Real-World AI Systems

**Learn how to build and deploy AI systems from edge devices to the cloud with this open-source textbook.**  

[View the original repository on GitHub](https://github.com/harvard-edge/cs249r_book)

[![Build Status](https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/validate-dev.yml?branch=dev&label=Build&logo=githubactions&cacheSeconds=300)](https://github.com/harvard-edge/cs249r_book/actions/workflows/validate-dev.yml)
[![Website](https://img.shields.io/website?url=https%3A%2F%2Fmlsysbook.ai&label=Website&logo=readthedocs)](https://mlsysbook.ai)
[![Ecosystem](https://img.shields.io/website?url=https%3A%2F%2Fmlsysbook.org&label=Ecosystem&logo=internet-explorer)](https://mlsysbook.org)
[![License](https://img.shields.io/badge/License-CC--BY--NC--SA%204.0-blue.svg)](https://github.com/harvard-edge/cs249r_book/blob/dev/LICENSE)
[![Funding](https://img.shields.io/badge/Fund%20Us-Open%20Collective-blue.svg?logo=open-collective)](https://opencollective.com/mlsysbook)
[![Powered by Netlify](https://img.shields.io/badge/Powered%20by-Netlify-00C7B7?logo=netlify&logoColor=white)](https://www.netlify.com)

**[📖 Read Online](https://mlsysbook.ai)** • **[💾 Download PDF](https://mlsysbook.ai/Machine-Learning-Systems.pdf)** • **[🌐 Explore Ecosystem](https://mlsysbook.org)**

📚 **Hardcopy edition coming 2026 via MIT Press!**

## Key Features

*   **Comprehensive Coverage:** Master the full stack of ML systems, from design to deployment.
*   **Real-World Focus:** Learn practical skills applicable to building production-ready AI solutions.
*   **Open Source & Free:** Access a complete, open-source textbook and resources.
*   **Community-Driven:** Join a global community of learners and contributors.
*   **Hands-on Labs:** Apply your knowledge through practical exercises and projects.
*   **Edge-to-Cloud:**  Understand the full spectrum of deployment, from embedded devices to cloud infrastructure.

## About the Book

"Machine Learning Systems" is an open-source textbook developed from Harvard University's CS249r course, designed to empower students and professionals with the knowledge to build and deploy real-world AI systems. Originally created by Professor Vijay Janapa Reddi, this book has become a valuable resource for universities and individuals worldwide.  The book focuses on the engineering aspects of AI, addressing the critical need for skilled engineers to build scalable and efficient AI systems.

## What You'll Learn

This book goes beyond model training, teaching you the critical skills needed to build, deploy, and maintain ML systems:

*   **System Design:** Architect scalable and maintainable ML systems.
*   **Data Engineering:** Build robust data pipelines for collection, processing, and labeling.
*   **Model Deployment:** Deploy models into production environments.
*   **MLOps & Monitoring:** Implement reliable and continuously operating systems.
*   **Edge AI:** Optimize and deploy AI models on resource-constrained devices (mobile, embedded, IoT).

## Support the Project

Help us democratize AI education and expand access to essential resources!

<div align="center">

### Show Your Support

**Star this repository** to show your support for open AI education!

[![Stars](https://img.shields.io/github/stars/harvard-edge/cs249r_book?style=for-the-badge&logo=github&color=gold)](https://github.com/harvard-edge/cs249r_book/stargazers)

**Goal:** 10,000 stars = $100,000 in additional education funding

[**⭐ Star Now**](https://github.com/harvard-edge/cs249r_book) — *takes 2 seconds!*

### Fund the Mission (New!)

We've graduated this project from Harvard to enable global access and expand AI systems education worldwide. Please help us support educators globally, especially in the Global South, by providing TinyML kits for students, funding workshops, and sustaining our open-source infrastructure.

[![Open Collective](https://img.shields.io/badge/💝%20Support%20AI%20Education-Open%20Collective-blue.svg?style=for-the-badge)](https://opencollective.com/mlsysbook)

*From $15/month to sponsor a learner to $250 for workshops — every contribution democratizes AI education.*

</div>

## Community and Resources

Explore the complete ecosystem for learning and contributing:

| Resource | Description |
|---|---|
| [📚 **Main Site**](https://mlsysbook.org) | Complete learning platform |
| [🔥 **TinyTorch**](https://mlsysbook.org/tinytorch) | Educational ML framework |
| [💬 **Discussions**](https://github.com/harvard-edge/cs249r_book/discussions) | Ask questions, share insights |
| [👥 **Community**](https://mlsysbook.org/community) | Join our global learning community |

## Who This Book Is For

Find the resources that best suit your needs:

### 🎓 Students

*   [📖 Read online](https://mlsysbook.ai)
*   [📄 Download PDF](https://mlsysbook.ai/Machine-Learning-Systems.pdf)
*   [🧪 Try hands-on labs](https://mlsysbook.org)

### 👩‍🏫 Educators

*   [📋 Course materials](https://mlsysbook.org)
*   [🎯 Instructor resources](https://mlsysbook.org)
*   [💡 Teaching guides](https://mlsysbook.org)

### 🛠️ Contributors

*   [🤝 Contribution guide](docs/contribute.md)
*   [⚡ Development setup](#development)
*   [💬 Join discussions](https://github.com/harvard-edge/cs249r_book/discussions)

## Quick Start

### For Readers

```bash
# Read online (continuously updated)
open https://mlsysbook.ai

# Or download PDF for offline access
curl -O https://mlsysbook.ai/Machine-Learning-Systems.pdf
```

### For Contributors

```bash
git clone https://github.com/harvard-edge/cs249r_book.git
cd cs249r_book

# Quick setup (recommended)
./binder setup      # Setup environment and dependencies
./binder doctor     # Check system health

# Fast development workflow
./binder preview intro    # Fast chapter development
./binder build intro      # Build specific chapter
./binder build            # Build complete book (HTML)
./binder help            # See all commands
```

## Contributing

We welcome contributions from the global community! Here's how you can help:

### Ways to Contribute

*   📝 **Content:** Suggest edits, improvements, or new examples.
*   🛠️ **Tools:** Enhance development scripts and automation.
*   🎨 **Design:** Improve figures, diagrams, and visual elements.
*   🌍 **Localization:** Translate content for global accessibility.
*   🔧 **Infrastructure:** Help with build systems and deployment.

### Quality Standards

All contributions benefit from automated quality assurance:

*   ✅ **Pre-commit validation:** Automatic cleanup and checks.
*   📋 **Content review:** Formatting and style validation.
*   🧪 **Testing:** Build and link verification.
*   👥 **Peer review:** Community feedback.

[**Start Contributing →**](docs/contribute.md)

## Development

### Book Binder CLI (Recommended)

Use the Book Binder CLI for a streamlined development and building process.

```bash
# Chapter development (fast iteration)
./binder preview intro                # Build and preview single chapter
./binder preview intro,ml_systems     # Build and preview multiple chapters

# Complete book building
./binder build                        # Build complete website (HTML)
./binder pdf                          # Build complete PDF
./binder epub                         # Build complete EPUB

# Management
./binder clean                        # Clean artifacts
./binder status                       # Show current status
./binder doctor                       # Run health check
./binder help                         # Show all commands
```

### Development Commands

```bash
# Book Binder CLI (Recommended)
./binder setup            # First-time setup
./binder build            # Build complete HTML book
./binder pdf              # Build complete PDF book
./binder epub             # Build complete EPUB book
./binder preview intro    # Preview chapter development

# Traditional setup (if needed)
python3 -m venv .venv
source .venv/bin/activate
pip install -r tools/dependencies/requirements.txt
pre-commit install
```

### Project Structure

```
MLSysBook/
├── binder                   # ⚡ Fast development CLI (recommended)
├── quarto/                  # Main book content (Quarto)
│   ├── contents/            # Chapter content
│   │   ├── core/            # Core chapters
│   │   ├── labs/            # Hands-on labs
│   │   ├── frontmatter/     # Preface, acknowledgments
│   │   ├── backmatter/      # References and resources
│   │   └── parts/           # Book parts and sections
│   ├── _extensions/         # Quarto extensions
│   ├── config/              # Build configurations
│   │   ├── _quarto-html.yml # Website build configuration
│   │   └── _quarto-pdf.yml  # PDF build configuration
│   ├── data/                # Cross-reference and metadata files
│   ├── assets/              # Images, styles, media
│   ├── filters/             # Lua filters
│   ├── scripts/             # Build scripts
│   └── _quarto.yml          # Active config (symlink)
├── tools/                   # Development automation
│   ├── scripts/             # Organized development scripts
│   │   ├── content/         # Content management tools
│   │   ├── cross_refs/      # Cross-reference management
│   │   ├── genai/           # AI-assisted content tools
│   │   ├── maintenance/     # System maintenance scripts
│   │   ├── testing/         # Test and validation scripts
│   │   └── utilities/       # General utility scripts
│   ├── dependencies/        # Package requirements
│   └── setup/               # Setup and configuration
├── config/                  # Project configuration
│   ├── dev/                 # Development configurations
│   ├── linting/             # Code quality configurations
│   └── quarto/              # Quarto publishing settings
├── docs/                    # Documentation
│   ├── BINDER.md            # Binder CLI guide
│   ├── BUILD.md             # Build instructions
│   ├── DEVELOPMENT.md       # Development guide
│   └── contribute.md        # Contribution guidelines
├── CHANGELOG.md             # Project changelog
├── CITATION.bib             # Citation information
├── pyproject.toml           # Python project configuration
└── README.md                # This file
```

### Documentation

*   [⚡ Binder CLI Guide](docs/BINDER.md) — Fast development with the Book Binder
*   [📋 Development Guide](docs/DEVELOPMENT.md) — Comprehensive setup and workflow
*   [🛠️ Maintenance Guide](docs/MAINTENANCE_GUIDE.md) — Daily tasks and troubleshooting
*   [🔨 Build Instructions](docs/BUILD.md) — Detailed build process
*   [🤝 Contribution Guidelines](docs/contribute.md) — How to contribute effectively

### Publishing

```bash
# Interactive publishing (recommended)
./binder publish

# Command-line publishing
./binder publish "Description" COMMIT_HASH

# Manual workflow (if needed)
./binder build html && ./binder build pdf
# Then use GitHub Actions to deploy
```

**Publishing Options:**

*   **`./binder publish`** — Unified command with interactive and command-line modes
*   **GitHub Actions** — Automated deployment via workflows

### Getting Started

```bash
# First time setup
./binder setup

# Check system health
./binder doctor

# Quick preview
./binder preview intro
```

## Citation & License

### Citation

```bibtex
@inproceedings{reddi2024mlsysbook,
  title        = {MLSysBook.AI: Principles and Practices of Machine Learning Systems Engineering},
  author       = {Reddi, Vijay Janapa},
  booktitle    = {2024 International Conference on Hardware/Software Codesign and System Synthesis (CODES+ ISSS)},
  pages        = {41--42},
  year         = {2024},
  organization = {IEEE},
  url          = {https://mlsysbook.org}
}
```

### License

This work is licensed under **Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International** (CC BY-NC-SA 4.0). You may share and adapt the material for non-commercial purposes with appropriate credit.

---

<div align="center">

**Made with ❤️ for AI learners worldwide**

Our goal is to educate 1 million AI systems engineers for the future at the edge of AI.
</div>