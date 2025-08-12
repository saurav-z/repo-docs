# Machine Learning Systems: Build Real-World AI from Edge to Cloud

**Unlock the secrets of building and deploying AI systems with this comprehensive, open-source textbook, originally developed at Harvard University.** ([Original Repository](https://github.com/harvard-edge/cs249r_book))

---

## Key Features

*   **Comprehensive Coverage:** Learn system design, data engineering, model deployment, MLOps, and edge AI.
*   **Hands-on Approach:** Go beyond theory with practical examples and labs.
*   **Open-Source & Accessible:** Free online access, PDF downloads, and a global community.
*   **Community-Driven:** Contribute to the project and connect with fellow learners.
*   **Built by Experts:** Developed from Harvard University's CS249r course.

---

## About the Book

This open-source textbook, **"Machine Learning Systems,"** is your guide to building real-world AI systems, covering everything from edge devices to cloud deployment. Originally developed as Harvard University's CS249r course by [Prof. Vijay Janapa Reddi](https://github.com/profvjreddi/homepage), it is now used by universities and students worldwide.

> *"This grew out of a concern that while students could train AI models, few understood how to build the systems that actually make them work. As AI becomes more capable and autonomous, the critical bottleneck won't be the algorithms - it will be the engineers who can build efficient, scalable, and sustainable systems that safely harness that intelligence."* - **Vijay Janapa Reddi**

*   **Hardcopy Edition coming 2026 via MIT Press!**

### Why This Book Exists

The book focuses on the skills needed to build and deploy effective AI systems, which are often overlooked in traditional machine learning education.

---

## 📚 What You'll Learn

Master the **full stack** of real-world ML systems and gain a competitive edge:

*   **System Design:** Build scalable and maintainable ML architectures.
*   **Data Engineering:** Develop robust pipelines for data collection, labeling, and processing.
*   **Model Deployment:** Deploy production-ready systems from prototypes.
*   **MLOps & Monitoring:** Create reliable and continuously operating systems.
*   **Edge AI:** Deploy resource-efficient systems on mobile, embedded, and IoT devices.

---

## ⭐ Support the Project

Help us expand AI education worldwide!

<div align="center">

### Show Your Support
**Star this repository** to help us demonstrate the value of open AI education to funders and institutions.

[![Stars](https://img.shields.io/github/stars/harvard-edge/cs249r_book?style=for-the-badge&logo=github&color=gold)](https://github.com/harvard-edge/cs249r_book/stargazers)

**Goal:** 10,000 stars = $100,000 in additional education funding

[**⭐ Star Now**](https://github.com/harvard-edge/cs249r_book) — *takes 2 seconds!*

### Fund the Mission (New!)
We've graduated this project from Harvard to enable global access and expand AI systems education worldwide. Please help us support educators globally, especially in the Global South, by providing TinyML kits for students, funding workshops, and sustaining our open-source infrastructure.

[![Open Collective](https://img.shields.io/badge/💝%20Support%20AI%20Education-Open%20Collective-blue.svg?style=for-the-badge)](https://opencollective.com/mlsysbook)

*From $15/month to sponsor a learner to $250 for workshops — every contribution democratizes AI education.*

</div>

---

## 🌐 Community & Resources

*   **[Main Site](https://mlsysbook.org):** Complete learning platform.
*   **[TinyTorch](https://mlsysbook.org/tinytorch):** Educational ML framework.
*   **[Discussions](https://github.com/harvard-edge/cs249r_book/discussions):** Ask questions and share insights.
*   **[Community](https://mlsysbook.org/community):** Join our global learning community.

---

## 🎯 For Different Audiences

### 🎓 Students

*   [Read online](https://mlsysbook.ai)
*   [Download PDF](https://mlsysbook.ai/Machine-Learning-Systems.pdf)
*   [Try hands-on labs](https://mlsysbook.org)

### 👩‍🏫 Educators

*   [Course materials](https://mlsysbook.org)
*   [Instructor resources](https://mlsysbook.org)
*   [Teaching guides](https://mlsysbook.org)

### 🛠️ Contributors

*   [Contribution guide](docs/contribute.md)
*   [Development setup](#development)
*   [Join discussions](https://github.com/harvard-edge/cs249r_book/discussions)

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
make setup-hooks  # Setup automated quality controls
make install      # Install dependencies

# Recommended: Use the binder for development
./binder preview intro    # Fast chapter development
./binder help            # See all commands

# Or use traditional Make commands
make preview        # Start development server
```

---

## 🤝 Contributing

We welcome contributions from the global community!

### Ways to Contribute

*   **📝 Content:** Suggest edits, improvements, or new examples.
*   **🛠️ Tools:** Enhance development scripts and automation.
*   **🎨 Design:** Improve figures, diagrams, and visual elements.
*   **🌍 Localization:** Translate content for global accessibility.
*   **🔧 Infrastructure:** Help with build systems and deployment.

### Quality Standards

All contributions benefit from automated quality assurance:

*   ✅ **Pre-commit validation:** Automatic cleanup and checks.
*   📋 **Content review:** Formatting and style validation.
*   🧪 **Testing:** Build and link verification.
*   👥 **Peer review:** Community feedback.

[**Start Contributing →**](docs/contribute.md)

---

## 🛠️ Development

### Book Binder CLI (Recommended)

The **Book Binder** streamlines building and iteration.

```bash
# Fast chapter development
./binder build intro html             # Build single chapter
./binder build intro,ml_systems html  # Build multiple chapters together
./binder preview intro                # Build and preview chapter

# Full book building
./binder build * html                 # Build complete website
./binder build * pdf                  # Build complete PDF

# Management
./binder clean                    # Clean artifacts
./binder status                   # Show current status
./binder help                     # Show all commands
```

### Make Commands (Traditional)

```bash
# Building
make build          # Build HTML version
make build-pdf      # Build PDF version
make preview        # Start development server

# Quality Control
make clean          # Clean build artifacts
make test           # Run validation tests
make lint           # Check for issues

# Get help
make help           # Show all commands
```

### Project Structure

```
MLSysBook/
├── binder                   # ⚡ Fast development CLI (recommended)
├── book/                    # Main book content (Quarto)
│   ├── contents/            # Chapter content
│   │   ├── core/            # Core chapters
│   │   ├── labs/            # Hands-on labs
│   │   ├── frontmatter/     # Preface, acknowledgments
│   │   ├── backmatter/      # References and resources
│   │   └── parts/           # Book parts and sections
│   ├── _extensions/         # Quarto extensions
│   ├── data/                # Cross-reference and metadata files
│   ├── _quarto-html.yml     # Website build configuration
│   ├── _quarto-pdf.yml      # PDF build configuration

│   ├── _quarto.yml          # Active config (symlink)
│   ├── index.qmd            # Main entry point
│   └── assets/              # Images, styles, media
├── build/                   # Build artifacts (git-ignored)
│   ├── html/                # HTML website output
│   ├── pdf/                 # PDF book output
│   └── dist/                # Distribution files
├── scripts/                 # Root-level development scripts
│   ├── content/             # Content management tools
│   ├── cross_refs/          # Cross-reference management
│   ├── genai/               # AI-assisted content tools
│   ├── maintenance/         # Maintenance scripts
│   ├── testing/             # Test scripts
│   └── utilities/           # General utilities
├── tools/                   # Development automation
│   ├── scripts/             # Organized development scripts
│   │   ├── build/           # Build and development tools
│   │   ├── content/         # Content management tools
│   │   ├── maintenance/     # System maintenance scripts
│   │   ├── testing/         # Test and validation scripts
│   │   ├── utilities/       # General utility scripts
│   │   └── docs/            # Script documentation
│   ├── dependencies/        # Package requirements
│   └── setup/               # Setup and configuration
├── config/                  # Build configuration
│   ├── dev/                 # Development configurations
│   ├── linting/             # Code quality configurations
│   ├── quarto/              # Quarto publishing settings
│   ├── lua/                 # Lua filters and scripts
│   └── tex/                 # LaTeX templates
├── locals/                  # Local development files
├── assets/                  # Global assets (covers, icons)
├── docs/                    # Documentation
│   ├── BINDER.md            # Binder CLI guide
│   ├── DEVELOPMENT.md       # Development guide
│   ├── MAINTENANCE_GUIDE.md # Daily workflow guide
│   ├── BUILD.md             # Build instructions
│   └── contribute.md        # Contribution guidelines
├── CHANGELOG.md             # Project changelog
├── CITATION.bib             # Citation information
└── Makefile                 # Traditional development commands
```

### Documentation

*   [⚡ Binder CLI Guide](docs/BINDER.md) — Fast development with the Book Binder
*   [📋 Development Guide](docs/DEVELOPMENT.md) — Comprehensive setup and workflow
*   [🛠️ Maintenance Guide](docs/MAINTENANCE_GUIDE.md) — Daily tasks and troubleshooting
*   [🔨 Build Instructions](docs/BUILD.md) — Detailed build process
*   [🤝 Contribution Guidelines](docs/contribute.md) — How to contribute effectively

### Publishing

```bash
# Command-line trigger (recommended)
./binder publish "Description" COMMIT_HASH

# Interactive wizard
./binder publish

# Manual steps
./binder build - html && ./binder build - pdf
# Then copy PDF to assets and push to main
```

**Publishing Options:**

*   `./binder publish` — Unified command with interactive and command-line modes
*   Web Interface — Manual trigger via GitHub Actions UI

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