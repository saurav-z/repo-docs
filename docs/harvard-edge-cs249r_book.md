# Machine Learning Systems: Build Real-World AI 

**Master the art of building AI systems with this comprehensive, open-source textbook.**

[📚 Read the Book](https://mlsysbook.ai) | [💾 Download PDF](https://mlsysbook.ai/Machine-Learning-Systems.pdf) | [🌐 Explore the Ecosystem](https://mlsysbook.org) | [⭐ Star on GitHub](https://github.com/harvard-edge/cs249r_book)

[![Build Status](https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/controller.yml?branch=dev&label=Build)](https://github.com/harvard-edge/cs249r_book/actions/workflows/controller.yml?query=branch%3Adev)
[![Website](https://img.shields.io/website?url=https://mlsysbook.ai&label=Website)](https://mlsysbook.ai)
[![Ecosystem](https://img.shields.io/website?url=https://mlsysbook.org&label=Ecosystem)](https://mlsysbook.org)
[![License](https://img.shields.io/badge/license-CC--BY--NC--SA%204.0-blue)](https://github.com/harvard-edge/cs249r_book/blob/dev/LICENSE)
[![Open Collective](https://img.shields.io/badge/fund%20us-Open%20Collective-blue.svg)](https://opencollective.com/mlsysbook)

---

## About the Book

This open-source textbook, originating from Harvard University's CS249r course, provides a comprehensive guide to building real-world AI systems, from edge devices to cloud deployment. It's used by universities and students worldwide. This is a project by [Prof. Vijay Janapa Reddi](https://github.com/profvjreddi/homepage).

**Key Features:**

*   **Comprehensive Coverage:** Learn to build the complete AI systems stack.
*   **Open-Source:** Freely available for everyone to read, use, and contribute to.
*   **Practical Focus:**  Hands-on labs and real-world examples.
*   **Community-Driven:** Join a global community of learners and contributors.
*   **Continuous Updates:** Content is constantly updated.

📚 **Hardcopy edition coming 2026 via MIT Press!**

---

## What You'll Learn

Go beyond training models and build production-ready AI systems.

*   **System Design:** Architect scalable and maintainable ML systems.
*   **Data Engineering:** Create robust data pipelines for ML.
*   **Model Deployment:** Deploy your models into production.
*   **MLOps & Monitoring:** Implement reliable and observable systems.
*   **Edge AI:** Deploy resource-efficient AI on edge devices.

---

## Support the Project

Help us expand AI education worldwide.

**⭐ Star this repository** to show your support!

[![Stars](https://img.shields.io/github/stars/harvard-edge/cs249r_book?style=for-the-badge&logo=github&color=gold)](https://github.com/harvard-edge/cs249r_book/stargazers)

**Goal:** 10,000 stars = $100,000 in additional education funding

[**⭐ Star Now**](https://github.com/harvard-edge/cs249r_book) — *takes 2 seconds!*

**Fund the Mission**

Support our mission to expand AI education by funding TinyML kits, workshops, and infrastructure.

[![Open Collective](https://img.shields.io/badge/💝%20Support%20AI%20Education-Open%20Collective-blue.svg?style=for-the-badge)](https://opencollective.com/mlsysbook)

---

## Community & Resources

Find everything you need to learn and contribute.

*   [📚 **Main Site**](https://mlsysbook.org): Complete learning platform
*   [🔥 **TinyTorch**](https://mlsysbook.org/tinytorch): Educational ML framework
*   [💬 **Discussions**](https://github.com/harvard-edge/cs249r_book/discussions): Ask questions and share insights
*   [👥 **Community**](https://mlsysbook.org/community): Join our global learning community

---

## For Different Audiences

**Students:**

*   [📖 Read online](https://mlsysbook.ai)
*   [📄 Download PDF](https://mlsysbook.ai/Machine-Learning-Systems.pdf)
*   [🧪 Try hands-on labs](https://mlsysbook.org)

**Educators:**

*   [📋 Course materials](https://mlsysbook.org)
*   [🎯 Instructor resources](https://mlsysbook.org)
*   [💡 Teaching guides](https://mlsysbook.org)

**Contributors:**

*   [🤝 Contribution guide](docs/contribute.md)
*   [⚡ Development setup](#development)
*   [💬 Join discussions](https://github.com/harvard-edge/cs249r_book/discussions)

---

## Quick Start

**For Readers:**

```bash
# Read online (continuously updated)
open https://mlsysbook.ai

# Or download PDF for offline access
curl -O https://mlsysbook.ai/Machine-Learning-Systems.pdf
```

**For Contributors:**

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

## Contributing

We welcome contributions to improve the book.

**Ways to Contribute:**

*   📝 Content (edits, improvements, examples)
*   🛠️ Tools (development scripts and automation)
*   🎨 Design (figures, diagrams, visual elements)
*   🌍 Localization (translations)
*   🔧 Infrastructure (build systems and deployment)

**Quality Standards:**

*   ✅ Pre-commit validation
*   📋 Content review
*   🧪 Testing
*   👥 Peer review

[**Start Contributing →**](docs/contribute.md)

---

## Development

**Book Binder CLI (Recommended)**

The Book Binder offers a streamlined development experience.

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

**Make Commands (Traditional)**

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

**Project Structure**
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

**Documentation**

*   [⚡ Binder CLI Guide](docs/BINDER.md)
*   [📋 Development Guide](docs/DEVELOPMENT.md)
*   [🛠️ Maintenance Guide](docs/MAINTENANCE_GUIDE.md)
*   [🔨 Build Instructions](docs/BUILD.md)
*   [🤝 Contribution Guidelines](docs/contribute.md)

---

## Citation & License

**Citation**

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

**License**

This work is licensed under **Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International** (CC BY-NC-SA 4.0).

---

<div align="center">

**Made with ❤️ for AI learners worldwide**
</div>