# Machine Learning Systems: Principles and Practices

**Master the art of building real-world AI systems with this open-source textbook.**

[![Build](https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/validate-dev.yml?branch=dev&label=Build&logo=githubactions&cacheSeconds=300)](https://github.com/harvard-edge/cs249r_book/actions/workflows/validate-dev.yml)
[![Last Commit](https://img.shields.io/github/last-commit/harvard-edge/cs249r_book/dev?label=Last%20Commit&logo=git&cacheSeconds=300)](https://github.com/harvard-edge/cs249r_book/commits/dev)
[![Website](https://img.shields.io/website?url=https%3A%2F%2Fmlsysbook.ai&label=Website&logo=readthedocs)](https://mlsysbook.ai)
[![Ecosystem](https://img.shields.io/website?url=https%3A%2F%2Fmlsysbook.org&label=Ecosystem&logo=internet-explorer)](https://mlsysbook.org)
[![Paper](https://img.shields.io/badge/Paper-MLSysBook.AI%20Overview-blue?logo=academia)](LINK_TO_PAPER)
[![Funding](https://img.shields.io/badge/Fund%20Us-Open%20Collective-blue.svg?logo=open-collective)](https://opencollective.com/mlsysbook)
[![License](https://img.shields.io/badge/License-CC--BY--NC--SA%204.0-blue.svg)](https://github.com/harvard-edge/cs249r_book/blob/dev/LICENSE)
[![Powered by Netlify](https://img.shields.io/badge/Powered%20by-Netlify-00C7B7?logo=netlify&logoColor=white)](https://www.netlify.com)

**[📖 Read Online](https://mlsysbook.ai)** •
**[💾 Download PDF](https://mlsysbook.ai/pdf)** •
**[💾 Download ePub](https://mlsysbook.ai/epub)** •
**[🌐 Explore Ecosystem](https://mlsysbook.org)**

📚 **Hardcopy edition coming 2026 via MIT Press!**

---

## About Machine Learning Systems

This is an **open-source textbook** designed to empower you with the knowledge and skills to build and deploy AI systems, from edge devices to cloud infrastructure. Originally developed as Harvard University's CS249r course by [Prof. Vijay Janapa Reddi](https://github.com/profvjreddi/homepage), this resource is now used by universities and learners globally.

### Core Principles

*   **Practical Focus:** Learn by doing with hands-on labs and real-world examples.
*   **Comprehensive Coverage:**  Master the entire ML systems lifecycle, not just model training.
*   **Open & Accessible:** Free to use, modify, and share under a Creative Commons license.

### Key Benefits

*   **Demystify AI Systems:** Understand the complexities of building production-ready AI.
*   **Career Advancement:** Equip yourself with in-demand skills for the future of AI.
*   **Community-Driven:** Join a global network of learners and contributors.

---

## What You'll Learn: Building AI Systems

This book guides you beyond model training to master the full spectrum of Machine Learning Systems Engineering.

*   **System Design:** Learn how to architect scalable, maintainable ML systems.
*   **Data Engineering:**  Build robust pipelines for data collection, labeling, and processing.
*   **Model Deployment:** Get production-ready by deploying models from prototype to product.
*   **MLOps & Monitoring:**  Build reliable and continuously operating systems.
*   **Edge AI:** Deploy ML models efficiently on resource-constrained devices (mobile, embedded, IoT).

---

## Support and Contribute

### Show Your Support

Help us to demonstrate the value of open AI education to funders and institutions.

[![Stars](https://img.shields.io/github/stars/harvard-edge/cs249r_book?style=for-the-badge&logo=github&color=gold)](https://github.com/harvard-edge/cs249r_book/stargazers)

**Goal:** 10,000 stars = $100,000 in additional education funding

[**⭐ Star Now**](https://github.com/harvard-edge/cs249r_book) — *takes 2 seconds!*

### Fund the Mission

Help us support educators globally, especially in the Global South, by providing TinyML kits for students, funding workshops, and sustaining our open-source infrastructure.

[![Open Collective](https://img.shields.io/badge/💝%20Support%20AI%20Education-Open%20Collective-blue.svg?style=for-the-badge)](https://opencollective.com/mlsysbook)

*From $15/month to sponsor a learner to $250 for workshops — every contribution democratizes AI education.*

---

## 🌐 Resources & Community

*   📚 **Main Site:** Complete learning platform at [mlsysbook.org](https://mlsysbook.org)
*   🔥 **TinyTorch:** Educational ML framework at [mlsysbook.org/tinytorch](https://mlsysbook.org/tinytorch)
*   💬 **Discussions:** Ask questions and share insights on [GitHub Discussions](https://github.com/harvard-edge/cs249r_book/discussions)
*   👥 **Community:** Join our global learning community on [mlsysbook.org/community](https://mlsysbook.org/community)

---

## 🎯  Target Audiences

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
*   💬 [Join discussions](https://github.com/harvard-edge/cs249r_book/discussions)

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
./binder doctor     # Check system health

# Fast development workflow
./binder preview intro    # Fast chapter development
./binder build intro      # Build specific chapter
./binder build            # Build complete book (HTML)
./binder help            # See all commands
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

The **Book Binder** is our lightning-fast development CLI for streamlined building and iteration:

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

## 🙏 Contributors

Thank you to these contributors for making this resource better for everyone.

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?s=100" width="100px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/hzeljko"><img src="https://avatars.githubusercontent.com/hzeljko?s=100" width="100px;" alt="Zeljko Hrcek"/><br /><sub><b>Zeljko Hrcek</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/Mjrovai"><img src="https://avatars.githubusercontent.com/Mjrovai?s=100" width="100px;" alt="Marcelo Rovai"/><br /><sub><b>Marcelo Rovai</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/jasonjabbour"><img src="https://avatars.githubusercontent.com/jasonjabbour?s=100" width="100px;" alt="jasonjabbour"/><br /><sub><b>jasonjabbour</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/uchendui"><img src="https://avatars.githubusercontent.com/uchendui?s=100" width="100px;" alt="Ikechukwu Uchendu"/><br /><sub><b>Ikechukwu Uchendu</b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/kai4avaya"><img src="https://avatars.githubusercontent.com/kai4avaya?s=100" width="100px;" alt="Kai Kleinbard"/><br /><sub><b>Kai Kleinbard</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/Naeemkh"><img src="https://avatars.githubusercontent.com/Naeemkh?s=100" width="100px;" alt="Naeem Khoshnevis"/><br /><sub><b>Naeem Khoshnevis</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/Sara-Khosravi"><img src="https://avatars.githubusercontent.com/Sara-Khosravi?s=100" width="100px;" alt="Sara Khosravi"/><br /><sub><b>Sara Khosravi</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/V0XNIHILI"><img src="https://avatars.githubusercontent.com/V0XNIHILI?s=100" width="100px;" alt="Douwe den Blanken"/><br /><sub><b>Douwe den Blanken</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/18jeffreyma"><img src="https://avatars.githubusercontent.com/18jeffreyma?s=100" width="100px;" alt="Jeffrey Ma"/><br /><sub><b>Jeffrey Ma</b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/shanzehbatool"><img src="https://avatars.githubusercontent.com/shanzehbatool?s=100" width="100px;" alt="shanzehbatool"/><br /><sub><b>shanzehbatool</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/eliasab16"><img src="https://avatars.githubusercontent.com/eliasab16?s=100" width="100px;" alt="Elias"/><br /><sub><b>Elias</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/JaredP94"><img src="https://avatars.githubusercontent.com/JaredP94?s=100" width="100px;" alt="Jared Ping"/><br /><sub><b>Jared Ping</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/ishapira1"><img src="https://avatars.githubusercontent.com/ishapira1?s=100" width="100px;" alt="Itai Shapira"/><br /><sub><b>Itai Shapira</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/8863743b4f26c1a20e730fcf7ebc3bc0?d=identicon&s=100?s=100" width="100px;" alt="Maximilian Lam"/><br /><sub><b>Maximilian Lam</b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/jaysonzlin"><img src="https://avatars.githubusercontent.com/jaysonzlin?s=100" width="100px;" alt="Jayson Lin"/><br /><sub><b>Jayson Lin</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/andreamurillomtz"><img src="https://avatars.githubusercontent.com/andreamurillomtz?s=100" width="100px;" alt="Andrea"/><br /><sub><b>Andrea</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/sophiacho1"><img src="https://avatars.githubusercontent.com/sophiacho1?s=100" width="100px;" alt="Sophia Cho"/><br /><sub><b>Sophia Cho</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/alxrod"><img src="https://avatars.githubusercontent.com/alxrod?s=100" width="100px;" alt="Alex Rodriguez"/><br /><sub><b>Alex Rodriguez</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/korneelf1"><img src="https://avatars.githubusercontent.com/korneelf1?s=100" width="100px;" alt="Korneel Van den Berghe"/><br /><sub><b>Korneel Van den Berghe</b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/zishenwan"><img src="https://avatars.githubusercontent.com/zishenwan?s=100" width="100px;" alt="Zishen Wan"/><br /><sub><b>Zishen Wan</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/colbybanbury"><img src="https://avatars.githubusercontent.com/colbybanbury?s=100" width="100px;" alt="Colby Banbury"/><br /><sub><b>Colby Banbury</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/mmaz"><img src="https://avatars.githubusercontent.com/mmaz?s=100" width="100px;" alt="Mark Mazumder"/><br /><sub><b>Mark Mazumder</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/DivyaAmirtharaj"><img src="https://avatars.githubusercontent.com/DivyaAmirtharaj?s=100" width="100px;" alt="Divya Amirtharaj"/><br /><sub><b>Divya Amirtharaj</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/srivatsankrishnan"><img src="https://avatars.githubusercontent.com/srivatsankrishnan?s=100" width="100px;" alt="Srivatsan Krishnan"/><br /><sub><b>Srivatsan Krishnan</b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/ma3mool"><img src="https://avatars.githubusercontent.com/ma3mool?s=100" width="100px;" alt="Abdulrahman Mahmoud"/><br /><sub><b>Abdulrahman Mahmoud</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/aptl26"><img src="https://avatars.githubusercontent.com/aptl26?s=100" width="100px;" alt="Aghyad Deeb"/><br /><sub><b>Aghyad Deeb</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/arnaumarin"><img src="https://avatars.githubusercontent.com/arnaumarin?s=100" width="100px;" alt="marin-llobet"/><br /><sub><b>marin-llobet</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/James-QiuHaoran"><img src="https://avatars.githubusercontent.com/James-QiuHaoran?s=100" width="100px;" alt="Haoran Qiu"/><br /><sub><b>Haoran Qiu</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/oishib"><img src="https://avatars.githubusercontent.com/oishib?s=100" width="100px;" alt="oishib"/><br /><sub><b>oishib</b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/jared-ni"><img src="https://avatars.githubusercontent.com/jared-ni?s=100" width="100px;" alt="Jared Ni"/><br /><sub><b>Jared Ni</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/ELSuitorHarvard"><img src="https://avatars.githubusercontent.com/ELSuitorHarvard?s=100" width="100px;" alt="ELSuitorHarvard"/><br /><sub><b>ELSuitorHarvard</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/Ekhao"><img src="https://avatars.githubusercontent.com/Ekhao?s=100" width="100px;" alt="Emil Njor"/><br /><sub><b>Emil Njor</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/MichaelSchnebly"><img src="https://avatars.githubusercontent.com/MichaelSchnebly?s=100" width="100px;" alt="Michael Schnebly"/><br /><sub><b>Michael Schnebly</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/AditiR-42"><img src="https://avatars.githubusercontent.com/AditiR-42?s=100" width="100px;" alt="Aditi Raju"/><br /><sub><b>Aditi Raju</b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/jaywonchung"><img src="https://avatars.githubusercontent.com/jaywonchung?s=100" width="100px;" alt="Jae-Won Chung"/><br /><sub><b>Jae-Won Chung</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/leo47007"><img src="https://avatars.githubusercontent.com/leo47007?s=100" width="100px;" alt="Yu-Shun Hsiao"/><br /><sub><b>Yu-Shun Hsiao</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/BaeHenryS"><img src="https://avatars.githubusercontent.com/BaeHenryS?s=100" width="100px;" alt="Henry Bae"/><br /><sub><b>Henry Bae</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/ShvetankPrakash"><img src="https://avatars.githubusercontent.com/ShvetankPrakash?s=100" width="100px;" alt="Shvetank Prakash"/><br /><sub><b>Shvetank Prakash</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/af39c27c6090c50a1921a9b6366e81cc?d=identicon&s=100?s=100" width="100px;" alt="Emeka Ezike"/><br /><sub><b>Emeka Ezike</b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/arbass22"><img src="https://avatars.githubusercontent.com/arbass22?s=100" width="100px;" alt="Andrew Bass"/><br /><sub><b>Andrew Bass</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/jzhou1318"><img src="https://avatars.githubusercontent.com/jzhou1318?s=100" width="100px;" alt="Jennifer Zhou"/><br /><sub><b>Jennifer Zhou</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/aryatschand"><img src="https://avatars.githubusercontent.com/aryatschand?s=100" width="100px;" alt="Arya Tschand"/><br /><sub><b>Arya Tschand</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/pongtr"><img src="https://avatars.githubusercontent.com/pongtr?s=100" width="100px;" alt="Pong Trairatvorakul"/><br /><sub><b>Pong Trairatvorakul</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/0c931fcfd03cd548d44c90602dd773ba?d=identicon&s=100?s=100" width="100px;" alt="Matthew Stewart"/><br /><sub><b>Matthew Stewart</b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/marcozennaro"><img src="https://avatars.githubusercontent.com/marcozennaro?s=100" width="100px;" alt="Marco Zennaro"/><br /><sub><b>Marco Zennaro</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/euranofshin"><img src="https://avatars.githubusercontent.com/euranofshin?s=100" width="100px;" alt="Eura Nofshin"/><br /><sub><b>Eura Nofshin</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/BrunoScaglione"><img src="https://avatars.githubusercontent.com/BrunoScaglione?s=100" width="100px;" alt="Bruno Scaglione"/><br /><sub><b>Bruno Scaglione</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/taunoe"><img src="https://avatars.githubusercontent.com/taunoe?s=100" width="100px;" alt="Tauno Erik"/><br /><sub><b>Tauno Erik</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/alex-oesterling"><img src="https://avatars.githubusercontent.com/alex-oesterling?s=100" width="100px;" alt="Alex Oesterling"/><br /><sub><b>Alex Oesterling</b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/gnodipac886"><img src="https://avatars.githubusercontent.com/gnodipac886?s=100" width="100px;" alt="gnodipac886"/><br /><sub><b>gnodipac886</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/FinAminToastCrunch"><img src="https://avatars.githubusercontent.com/FinAminToastCrunch?s=100" width="100px;" alt="Fin Amin"/><br /><sub><b>Fin Amin</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/Allen-Kuang"><img src="https://avatars.githubusercontent.com/Allen-Kuang?s=100" width="100px;" alt="Allen-Kuang"/><br /><sub><b>Allen-Kuang</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/TheHiddenLayer"><img src="https://avatars.githubusercontent.com/TheHiddenLayer?s=100" width="100px;" alt="TheHiddenLayer"/><br /><sub><b>TheHiddenLayer</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/Gjain234"><img src="https://avatars.githubusercontent.com/Gjain234?s=100" width="100px;" alt="Gauri Jain"/><br /><sub><b>Gauri Jain</b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/468ef35acc69f3266efd700992daa369?d=identicon&s=100?s=100" width="100px;" alt="Fatima Shah"/><br /><sub><b>Fatima Shah</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/serco425"><img src="https://avatars.github