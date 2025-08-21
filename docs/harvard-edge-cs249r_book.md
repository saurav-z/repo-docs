# Machine Learning Systems: Build Real-World AI (Open-Source Textbook)

**Master the art of building and deploying cutting-edge AI systems with this comprehensive, open-source textbook.** [Explore the original repository](https://github.com/harvard-edge/cs249r_book).

---

## Key Features

*   📚 **Comprehensive Coverage:** Learn the full stack of ML systems, from design to deployment.
*   🛠️ **Hands-on Labs:** Build practical skills with real-world examples and projects.
*   🌐 **Open Source & Accessible:** Free to read online, download, and contribute to.
*   🎓 **For Students & Educators:** Course materials, instructor resources, and teaching guides are available.
*   🤝 **Community-Driven:** Join a global community of learners and contributors.

---

## About This Book

Developed from Harvard University's CS249r course, this open-source textbook provides a practical guide to building and deploying Machine Learning Systems. It covers a wide range of topics, including system design, data engineering, model deployment, MLOps, and edge AI. The book aims to expand access to AI systems education worldwide, empowering learners to build efficient, scalable, and sustainable AI systems. Hardcopy edition coming 2026 via MIT Press!

---

## What You'll Learn

Go beyond model training and become proficient in the complete ML systems lifecycle:

*   **System Design:** Architect scalable and maintainable ML systems.
*   **Data Engineering:** Build robust data pipelines for collection, labeling, and processing.
*   **Model Deployment:** Take models from prototype to production-ready systems.
*   **MLOps & Monitoring:** Ensure reliable, continuously operating ML systems.
*   **Edge AI:** Deploy resource-efficient systems on mobile, embedded, and IoT devices.

---

## ⭐ Support This Project

Help us democratize AI education!

*   **Star the Repository:** Show your support and help us secure funding.
    [![Stars](https://img.shields.io/github/stars/harvard-edge/cs249r_book?style=for-the-badge&logo=github&color=gold)](https://github.com/harvard-edge/cs249r_book/stargazers)
    [**⭐ Star Now**](https://github.com/harvard-edge/cs249r_book)
    *Goal: 10,000 stars = $100,000 in additional education funding*
*   **Fund the Mission:** Support educators globally and provide access to TinyML kits and workshops.
    [![Open Collective](https://img.shields.io/badge/💝%20Support%20AI%20Education-Open%20Collective-blue.svg?style=for-the-badge)](https://opencollective.com/mlsysbook)

---

## 🌐 Community & Resources

| Resource                                   | Description                                      |
| :----------------------------------------- | :----------------------------------------------- |
| [📚 **Main Site**](https://mlsysbook.org)      | Complete learning platform                       |
| [🔥 **TinyTorch**](https://mlsysbook.org/tinytorch) | Educational ML framework                        |
| [💬 **Discussions**](https://github.com/harvard-edge/cs249r_book/discussions) | Ask questions, share insights                    |
| [👥 **Community**](https://mlsysbook.org/community)   | Join our global learning community            |

---

## 🎯 For Different Audiences

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

---

## 🚀 Quick Start

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
./binder hello      # Welcome and overview

# Fast development workflow
./binder preview intro    # Fast chapter development
./binder build html       # Build complete book
./binder help            # See all commands
```

---

## 🤝 Contributing

We welcome contributions from the global community!  See the [Contribution Guide](docs/contribute.md)

### Ways to Contribute

*   **📝 Content** — Suggest edits, improvements, or new examples
*   **🛠️ Tools** — Enhance development scripts and automation
*   **🎨 Design** — Improve figures, diagrams, and visual elements
*   **🌍 Localization** — Translate content for global accessibility
*   **🔧 Infrastructure** — Help with build systems and deployment

### Quality Standards

All contributions benefit from automated quality assurance:

*   ✅ **Pre-commit validation** — Automatic cleanup and checks
*   📋 **Content review** — Formatting and style validation
*   🧪 **Testing** — Build and link verification
*   👥 **Peer review** — Community feedback

---

## 🛠️ Development

### Book Binder CLI (Recommended)

The **Book Binder** is our lightning-fast development CLI for streamlined building and iteration:

```bash
# Chapter development (fast iteration)
./binder preview intro                # Build and preview single chapter
./binder preview intro,ml_systems     # Build and preview multiple chapters

# Complete book building
./binder build html                   # Build complete website
./binder build pdf                    # Build complete PDF
./binder build epub                   # Build complete EPUB

# Management
./binder clean                    # Clean artifacts
./binder status                   # Show current status
./binder help                     # Show all commands
```

### Development Commands

```bash
# Book Binder CLI (Recommended)
./binder setup            # First-time setup
./binder build html       # Build complete HTML book
./binder build pdf        # Build complete PDF book
./binder preview intro    # Preview chapter development
./binder publish          # Publish to production

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
# Welcome and overview
./binder hello

# First time setup
./binder setup

# Quick preview
./binder preview intro
```

---

## 📋 Citation & License

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