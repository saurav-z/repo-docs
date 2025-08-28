# Machine Learning Systems: Build Real-World AI (Your Complete Guide)

**Master the principles and practices of engineering intelligent systems with this open-source book, originally developed at Harvard University.** [Explore the original repository](https://github.com/harvard-edge/cs249r_book).

[![Build Status](https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/validate-dev.yml?branch=dev&label=Build&logo=githubactions&cacheSeconds=300)](https://github.com/harvard-edge/cs249r_book/actions/workflows/validate-dev.yml)
[![Website](https://img.shields.io/website?url=https%3A%2F%2Fmlsysbook.ai&label=Website&logo=readthedocs)](https://mlsysbook.ai)
[![Ecosystem](https://img.shields.io/website?url=https%3A%2F%2Fmlsysbook.org&label=Ecosystem&logo=internet-explorer)](https://mlsysbook.org)
[![License](https://img.shields.io/badge/License-CC--BY--NC--SA%204.0-blue.svg)](https://github.com/harvard-edge/cs249r_book/blob/dev/LICENSE)
[![Funding](https://img.shields.io/badge/Fund%20Us-Open%20Collective-blue.svg?logo=open-collective)](https://opencollective.com/mlsysbook)
[![Powered by Netlify](https://img.shields.io/badge/Powered%20by-Netlify-00C7B7?logo=netlify&logoColor=white)](https://www.netlify.com)

**[📖 Read Online](https://mlsysbook.ai) | [💾 Download PDF](https://mlsysbook.ai/Machine-Learning-Systems.pdf) | [🌐 Explore Ecosystem](https://mlsysbook.org)**

📚 **Hardcopy edition coming 2026 via MIT Press!**

---

## About the Book: Your Path to AI Systems Engineering

This **open-source textbook**, born from Harvard's CS249r course, guides you through building real-world AI systems, from edge devices to cloud deployment.  It's used by universities and students globally.

> **Our Mission:** To democratize AI systems education, empowering learners through accessible resources.

### Why This Book Matters

Learn to build efficient, scalable, and sustainable AI systems.  As AI capabilities grow, the key is the engineering that builds them.

---

## Key Features: What You'll Learn

Go beyond model training and master the **full stack** of ML systems.

*   **System Design:** Architect scalable and maintainable ML systems.
*   **Data Engineering:** Build robust pipelines for data collection, labeling, and processing.
*   **Model Deployment:** Deploy production-ready systems from prototypes.
*   **MLOps & Monitoring:** Create reliable, continuously operating systems.
*   **Edge AI:** Optimize for resource-constrained environments (mobile, embedded, IoT).

---

## Support the Mission: Help Us Educate the Next Generation of AI Engineers

<div align="center">

### Show Your Support with a Star!
**Star this repository** to help secure funding for AI education.

[![Stars](https://img.shields.io/github/stars/harvard-edge/cs249r_book?style=for-the-badge&logo=github&color=gold)](https://github.com/harvard-edge/cs249r_book/stargazers)

**Goal:** 10,000 stars = $100,000 in additional education funding

[**⭐ Star Now**](https://github.com/harvard-edge/cs249r_book) — *takes 2 seconds!*

### Fund AI Education (New!)
Support educators globally by donating to our Open Collective.  Contributions provide TinyML kits, fund workshops, and sustain our infrastructure.

[![Open Collective](https://img.shields.io/badge/💝%20Support%20AI%20Education-Open%20Collective-blue.svg?style=for-the-badge)](https://opencollective.com/mlsysbook)

*From $15/month to sponsor a learner to $250 for workshops — every contribution democratizes AI education.*

</div>

---

## Community & Resources: Dive Deeper

| Resource | Description |
|----------|-------------|
| [📚 **Main Site**](https://mlsysbook.org) | Complete learning platform |
| [🔥 **TinyTorch**](https://mlsysbook.org/tinytorch) | Educational ML framework |
| [💬 **Discussions**](https://github.com/harvard-edge/cs249r_book/discussions) | Ask questions, share insights |
| [👥 **Community**](https://mlsysbook.org/community) | Join our global learning community |

---

## Targeted Audiences

### 🎓 Students
-   [📖 Read online](https://mlsysbook.ai)
-   [📄 Download PDF](https://mlsysbook.ai/Machine-Learning-Systems.pdf)
-   [🧪 Try hands-on labs](https://mlsysbook.org)

### 👩‍🏫 Educators
-   [📋 Course materials](https://mlsysbook.org)
-   [🎯 Instructor resources](https://mlsysbook.org)
-   [💡 Teaching guides](https://mlsysbook.org)

### 🛠️ Contributors
-   [🤝 Contribution guide](docs/contribute.md)
-   [⚡ Development setup](#development)
-   [💬 Join discussions](https://github.com/harvard-edge/cs249r_book/discussions)

---

## Quick Start: Get Started Now

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

---

## Contributing: Join Our Open Source Community

We welcome contributions!

### Ways to Contribute
*   **📝 Content:** Suggest edits, improvements, or new examples.
*   **🛠️ Tools:** Enhance development scripts and automation.
*   **🎨 Design:** Improve figures, diagrams, and visual elements.
*   **🌍 Localization:** Translate content for global accessibility.
*   **🔧 Infrastructure:** Help with build systems and deployment.

### Quality Standards
All contributions benefit from automated quality assurance:
*   ✅ **Pre-commit validation** — Automatic cleanup and checks
*   📋 **Content review** — Formatting and style validation
*   🧪 **Testing** — Build and link verification
*   👥 **Peer review** — Community feedback

[**Start Contributing →**](docs/contribute.md)

---

## Development: Building the Book

### Book Binder CLI (Recommended)

The **Book Binder** CLI streamlines your development workflow:

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
-   [⚡ Binder CLI Guide](docs/BINDER.md) — Fast development with the Book Binder
-   [📋 Development Guide](docs/DEVELOPMENT.md) — Comprehensive setup and workflow
-   [🛠️ Maintenance Guide](docs/MAINTENANCE_GUIDE.md) — Daily tasks and troubleshooting
-   [🔨 Build Instructions](docs/BUILD.md) — Detailed build process
-   [🤝 Contribution Guidelines](docs/contribute.md) — How to contribute effectively

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
-   **`./binder publish`** — Unified command with interactive and command-line modes
-   **GitHub Actions** — Automated deployment via workflows

### Getting Started
```bash
# First time setup
./binder setup

# Check system health
./binder doctor

# Quick preview
./binder preview intro
```

---

## Citation & License: Cite and Share

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