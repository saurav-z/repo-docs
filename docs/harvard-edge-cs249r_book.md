# Machine Learning Systems: Build Real-World AI (Open Source)

**Learn to build and deploy production-ready AI systems with this comprehensive, open-source textbook.**

[View the original repository on GitHub](https://github.com/harvard-edge/cs249r_book)

---

## Key Features

*   **Comprehensive Coverage:** Master the full stack of Machine Learning Systems, from design to deployment.
*   **Hands-on Labs:** Practice with real-world examples and build production-ready systems.
*   **Open-Source & Free:** Access the book, code, and resources without cost.
*   **Community-Driven:** Join a global community of learners, educators, and contributors.
*   **Edge to Cloud:** Learn to deploy AI on edge devices, in the cloud, and everywhere in between.

---

## What You'll Learn

This textbook covers the critical aspects of building and deploying real-world ML systems:

*   **System Design:** Architect scalable and maintainable ML systems.
*   **Data Engineering:** Build robust data pipelines for training and inference.
*   **Model Deployment:** Deploy models into production environments.
*   **MLOps & Monitoring:** Implement reliable and continuously operating systems.
*   **Edge AI:** Deploy resource-efficient systems on mobile, embedded, and IoT devices.

---

## Resources & Community

*   **[📖 Read Online](https://mlsysbook.ai)**: Access the continuously updated textbook.
*   **[💾 Download PDF](https://mlsysbook.ai/Machine-Learning-Systems.pdf)**: Download the PDF for offline reading.
*   **[🌐 Explore Ecosystem](https://mlsysbook.org)**: Discover the complete learning platform, including labs and community.
*   **[💬 Discussions](https://github.com/harvard-edge/cs249r_book/discussions)**: Ask questions and share insights.
*   **[🔥 TinyTorch](https://mlsysbook.org/tinytorch)**: Educational ML framework for hands-on learning.
*   **[👥 Community](https://mlsysbook.org/community)**: Join our global learning community.

---

## 🚀 Get Started

### For Readers

*   **Read Online:** [https://mlsysbook.ai](https://mlsysbook.ai)
*   **Download PDF:** [https://mlsysbook.ai/Machine-Learning-Systems.pdf](https://mlsysbook.ai/Machine-Learning-Systems.pdf)

### For Educators

*   **Course Materials:** [https://mlsysbook.org](https://mlsysbook.org)
*   **Instructor Resources:** [https://mlsysbook.org](https://mlsysbook.org)
*   **Teaching Guides:** [https://mlsysbook.org](https://mlsysbook.org)

### For Contributors

*   **Contribution Guide:** [docs/contribute.md](docs/contribute.md)
*   **Development Setup:** See below
*   **Join Discussions:** [https://github.com/harvard-edge/cs249r_book/discussions](https://github.com/harvard-edge/cs249r_book/discussions)

---

## 🤝 Contributing

We welcome contributions from the global community to improve and expand this educational resource. You can contribute by:

*   **Content:** Suggest edits, improvements, and new examples.
*   **Tools:** Enhance development scripts and automation.
*   **Design:** Improve figures, diagrams, and visual elements.
*   **Localization:** Translate content for global accessibility.
*   **Infrastructure:** Help with build systems and deployment.

[**Start Contributing →**](docs/contribute.md)

---

## 🛠️ Development

### Book Binder CLI (Recommended)

The Book Binder CLI offers a streamlined development workflow for fast building and iteration:

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
- [⚡ Binder CLI Guide](docs/BINDER.md) — Fast development with the Book Binder
- [📋 Development Guide](docs/DEVELOPMENT.md) — Comprehensive setup and workflow
- [🛠️ Maintenance Guide](docs/MAINTENANCE_GUIDE.md) — Daily tasks and troubleshooting  
- [🔨 Build Instructions](docs/BUILD.md) — Detailed build process
- [🤝 Contribution Guidelines](docs/contribute.md) — How to contribute effectively

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
- **`./binder publish`** — Unified command with interactive and command-line modes
- **GitHub Actions** — Automated deployment via workflows

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

## ⭐ Support & Funding

*   **Star the Repository:**  Help us demonstrate the value of open AI education to funders and institutions by starring this repository.
*   **Fund the Mission:** Support educators globally by contributing to our Open Collective. Every contribution democratizes AI education.

[![Stars](https://img.shields.io/github/stars/harvard-edge/cs249r_book?style=for-the-badge&logo=github&color=gold)](https://github.com/harvard-edge/cs249r_book/stargazers)

[![Open Collective](https://img.shields.io/badge/💝%20Support%20AI%20Education-Open%20Collective-blue.svg?style=for-the-badge)](https://opencollective.com/mlsysbook)

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
This work is licensed under **Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International** (CC BY-NC-SA 4.0).