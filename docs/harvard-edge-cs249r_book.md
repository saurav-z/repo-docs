# Machine Learning Systems: The Open-Source Textbook for Building Real-World AI Systems

This comprehensive open-source textbook teaches you the principles and practices of building and deploying cutting-edge machine learning systems. ([Original Repo](https://github.com/harvard-edge/cs249r_book))

<div align="center">

[![Build Status](https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/controller.yml?branch=dev&label=Build)](https://github.com/harvard-edge/cs249r_book/actions/workflows/controller.yml?query=branch%3Adev)
[![Website](https://img.shields.io/website?url=https://mlsysbook.ai&label=Website)](https://mlsysbook.ai)
[![Ecosystem](https://img.shields.io/website?url=https://mlsysbook.org&label=Ecosystem)](https://mlsysbook.org)
[![License](https://img.shields.io/badge/license-CC--BY--NC--SA%204.0-blue)](https://github.com/harvard-edge/cs249r_book/blob/dev/LICENSE)
[![Open Collective](https://img.shields.io/badge/fund%20us-Open%20Collective-blue.svg)](https://opencollective.com/mlsysbook)

**[📖 Read Online](https://mlsysbook.ai)** • **[💾 Download PDF](https://mlsysbook.ai/Machine-Learning-Systems.pdf)** • **[🌐 Explore Ecosystem](https://mlsysbook.org)**

📚 **Hardcopy edition coming 2026 via MIT Press!**

</div>

---

## Key Features

*   **Comprehensive Coverage:** Learn to build complete ML systems, from edge devices to cloud deployment.
*   **Open-Source & Accessible:**  Expand access to AI systems education worldwide.
*   **Hands-on Learning:**  Go beyond training models and master real-world ML system engineering.
*   **Community-Driven:** Contribute to a global learning community and help shape the future of AI education.
*   **Practical Focus:** Build scalable, efficient, and sustainable AI systems.

## What You'll Learn

Master the full stack of real-world ML systems with this open-source textbook.

*   **System Design:** Design scalable and maintainable ML architectures.
*   **Data Engineering:** Build robust pipelines for data collection, labeling, and processing.
*   **Model Deployment:** Deploy production-ready systems from prototypes.
*   **MLOps & Monitoring:**  Create reliable and continuously operating systems.
*   **Edge AI:** Deploy resource-efficient ML models on mobile, embedded, and IoT devices.

---

## Support This Work

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

*   [📚 **Main Site**](https://mlsysbook.org): Complete learning platform.
*   [🔥 **TinyTorch**](https://mlsysbook.org/tinytorch): Educational ML framework.
*   [💬 **Discussions**](https://github.com/harvard-edge/cs249r_book/discussions): Ask questions, share insights.
*   [👥 **Community**](https://mlsysbook.org/community): Join our global learning community.

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

*   📝 **Content:** Suggest edits, improvements, or new examples.
*   🛠️ **Tools:** Enhance development scripts and automation.
*   🎨 **Design:** Improve figures, diagrams, and visual elements.
*   🌍 **Localization:** Translate content for global accessibility.
*   🔧 **Infrastructure:** Help with build systems and deployment.

### Quality Standards

All contributions benefit from automated quality assurance:

*   ✅ **Pre-commit validation** — Automatic cleanup and checks.
*   📋 **Content review** — Formatting and style validation.
*   🧪 **Testing** — Build and link verification.
*   👥 **Peer review** — Community feedback.

[**Start Contributing →**](docs/contribute.md)

---

## 🛠️ Development

### Book Binder CLI (Recommended)

The **Book Binder** is our lightning-fast development CLI for streamlined building and iteration:

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
│   │   └── parts/           # Book parts and sections
│   ├── _quarto-html.yml     # Website build configuration
│   ├── _quarto-pdf.yml      # PDF build configuration
│   ├── _quarto.yml          # Active config (symlink)
│   ├── index.qmd            # Main entry point
│   └── assets/              # Images, styles, media
├── build/                   # Build artifacts (git-ignored)
│   ├── html/                # HTML website output
│   ├── pdf/                 # PDF book output
│   └── dist/                # Distribution files
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
│   ├── _extensions/         # Quarto extensions
│   ├── lua/                 # Lua filters and scripts
│   └── tex/                 # LaTeX templates
├── assets/                  # Global assets (covers, icons)
├── docs/                    # Documentation
│   ├── BINDER.md            # Binder CLI guide
│   ├── DEVELOPMENT.md       # Development guide
│   ├── MAINTENANCE_GUIDE.md # Daily workflow guide
│   ├── BUILD.md             # Build instructions
│   └── contribute.md        # Contribution guidelines
└── Makefile                 # Traditional development commands
```

### Documentation

*   [⚡ Binder CLI Guide](docs/BINDER.md): Fast development with the Book Binder.
*   [📋 Development Guide](docs/DEVELOPMENT.md): Comprehensive setup and workflow.
*   [🛠️ Maintenance Guide](docs/MAINTENANCE_GUIDE.md): Daily tasks and troubleshooting.
*   [🔨 Build Instructions](docs/BUILD.md): Detailed build process.
*   [🤝 Contribution Guidelines](docs/contribute.md): How to contribute effectively.

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

**Made for the global AI education community with ❤️**

*Empowering the next generation of AI systems engineers*

</div>