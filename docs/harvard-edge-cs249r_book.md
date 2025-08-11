# Machine Learning Systems: Build Real-World AI Systems

**Learn how to build and deploy real-world AI systems, from edge devices to the cloud, with this open-source textbook!** ([Original Repo](https://github.com/harvard-edge/cs249r_book))

[![Build](https://github.com/harvard-edge/cs249r_book/actions/workflows/validate-dev.yml/badge.svg?label=Build)](https://github.com/harvard-edge/cs249r_book/actions/workflows/validate-dev.yml)
[![Website](https://img.shields.io/website?url=https%3A%2F%2Fmlsysbook.ai&label=Website&logo=readthedocs)](https://mlsysbook.ai)
[![Ecosystem](https://img.shields.io/website?url=https%3A%2F%2Fmlsysbook.org&label=Ecosystem&logo=internet-explorer)](https://mlsysbook.org)
[![License](https://img.shields.io/badge/License-CC--BY--NC--SA%204.0-blue.svg)](https://github.com/harvard-edge/cs249r_book/blob/dev/LICENSE)
[![Funding](https://img.shields.io/badge/Fund%20Us-Open%20Collective-blue.svg?logo=open-collective)](https://opencollective.com/mlsysbook)

**[📖 Read Online](https://mlsysbook.ai)** • **[💾 Download PDF](https://mlsysbook.ai/Machine-Learning-Systems.pdf)** • **[🌐 Explore Ecosystem](https://mlsysbook.org)**

📚 **Hardcopy edition coming 2026 via MIT Press!**

---

## About the Book

Developed from Harvard University's CS249r course, this open-source textbook equips you with the knowledge and skills to build production-ready AI systems. It was originally created by [Prof. Vijay Janapa Reddi](https://github.com/profvjreddi/homepage).

### Key Features:

*   **Comprehensive Coverage:** Master the full stack of ML systems, from edge devices to cloud deployment.
*   **Hands-on Learning:** Go beyond training models and build real-world AI systems.
*   **Open-Source & Accessible:** Learn from a community-driven resource, available to everyone.
*   **Community Focused**: Join a global community focused on AI education.

> *"This grew out of a concern that while students could train AI models, few understood how to build the systems that actually make them work. As AI becomes more capable and autonomous, the critical bottleneck won't be the algorithms - it will be the engineers who can build efficient, scalable, and sustainable systems that safely harness that intelligence."* - **Vijay Janapa Reddi**

---

## What You'll Learn

This book teaches you how to build and deploy real-world AI systems, covering:

*   **System Design:** Architect scalable and maintainable ML systems.
*   **Data Engineering:** Build robust pipelines for data collection, processing, and labeling.
*   **Model Deployment:** Deploy your models into production-ready systems.
*   **MLOps & Monitoring:** Implement reliable and continuously operating systems.
*   **Edge AI:** Optimize for resource-efficient deployment on mobile, embedded, and IoT devices.

---

## Support the Mission

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

## Resources

| Resource | Description |
|----------|-------------|
| [📚 **Main Site**](https://mlsysbook.org) | Complete learning platform |
| [🔥 **TinyTorch**](https://mlsysbook.org/tinytorch) | Educational ML framework |
| [💬 **Discussions**](https://github.com/harvard-edge/cs249r_book/discussions) | Ask questions, share insights |
| [👥 **Community**](https://mlsysbook.org/community) | Join our global learning community |

---

## Who is this for?

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

## Getting Started

### For Readers

```bash
# Read online
open https://mlsysbook.ai

# Download PDF
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

## Contributing

We welcome contributions!

### How to Contribute

*   **📝 Content:** Suggest edits, improvements, or new examples.
*   **🛠️ Tools:** Enhance development scripts and automation.
*   **🎨 Design:** Improve figures, diagrams, and visual elements.
*   **🌍 Localization:** Translate content.
*   **🔧 Infrastructure:** Help with build systems and deployment.

### Quality Standards

*   ✅ **Pre-commit validation**
*   📋 **Content review**
*   🧪 **Testing**
*   👥 **Peer review**

[**Start Contributing →**](docs/contribute.md)

---

## Development

### Book Binder CLI (Recommended)

```bash
./binder build intro html
./binder build intro,ml_systems html
./binder preview intro
./binder build * html
./binder build * pdf
./binder clean
./binder status
./binder help
```

### Make Commands

```bash
make build
make build-pdf
make preview
make clean
make test
make lint
make help
```

### Project Structure

```
MLSysBook/
├── binder
├── book/
│   ├── contents/
│   ├── _extensions/
│   ├── data/
│   ├── _quarto-html.yml
│   ├── _quarto-pdf.yml
│   ├── _quarto.yml
│   ├── index.qmd
│   └── assets/
├── build/
│   ├── html/
│   ├── pdf/
│   └── dist/
├── scripts/
│   ├── content/
│   ├── cross_refs/
│   ├── genai/
│   ├── maintenance/
│   ├── testing/
│   └── utilities/
├── tools/
│   ├── scripts/
│   ├── dependencies/
│   └── setup/
├── config/
│   ├── dev/
│   ├── linting/
│   ├── quarto/
│   ├── lua/
│   └── tex/
├── locals/
├── assets/
├── docs/
│   ├── BINDER.md
│   ├── DEVELOPMENT.md
│   ├── MAINTENANCE_GUIDE.md
│   ├── BUILD.md
│   └── contribute.md
├── CHANGELOG.md
├── CITATION.bib
└── Makefile
```

### Documentation

*   [⚡ Binder CLI Guide](docs/BINDER.md)
*   [📋 Development Guide](docs/DEVELOPMENT.md)
*   [🛠️ Maintenance Guide](docs/MAINTENANCE_GUIDE.md)
*   [🔨 Build Instructions](docs/BUILD.md)
*   [🤝 Contribution Guidelines](docs/contribute.md)

### Publishing

```bash
./binder publish "Description" COMMIT_HASH
./binder publish
./binder build - html && ./binder build - pdf
```

**Publishing Options:**

*   `./binder publish` - Unified command with interactive and command-line modes
*   Web Interface - Manual trigger via GitHub Actions UI

### Getting Started

```bash
./binder hello
./binder setup
./binder preview intro
```

---

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

This work is licensed under **Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International** (CC BY-NC-SA 4.0).

---

<div align="center">

**Made with ❤️ for AI learners worldwide**
</div>