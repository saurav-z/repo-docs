# Machine Learning Systems: Principles and Practices

**Learn how to build real-world AI systems, from edge devices to cloud deployment, with this comprehensive, open-source textbook.**

[![Build](https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/validate-dev.yml?branch=dev&label=Build&logo=githubactions&cacheSeconds=300)](https://github.com/harvard-edge/cs249r_book/actions/workflows/validate-dev.yml)
[![Website](https://img.shields.io/website?url=https%3A%2F%2Fmlsysbook.ai&label=Website&logo=readthedocs)](https://mlsysbook.ai)
[![Ecosystem](https://img.shields.io/website?url=https%3A%2F%2Fmlsysbook.org&label=Ecosystem&logo=internet-explorer)](https://mlsysbook.org)
[![License](https://img.shields.io/badge/License-CC--BY--NC--SA%204.0-blue.svg)](https://github.com/harvard-edge/cs249r_book/blob/dev/LICENSE)
[![Stars](https://img.shields.io/github/stars/harvard-edge/cs249r_book?style=social&label=Star&logo=github)](https://github.com/harvard-edge/cs249r_book)

[**View the source code on GitHub**](https://github.com/harvard-edge/cs249r_book)

---

## Key Features

*   **Open-Source:** Access a free, community-driven resource.
*   **Comprehensive Coverage:** Master the full ML systems stack.
*   **Hands-on Learning:** Build practical AI systems.
*   **Edge to Cloud:** Learn deployment across diverse environments.
*   **Community-Driven:** Join a global community of learners and contributors.

---

## What You'll Learn

This book goes beyond model training, equipping you with the skills to build and deploy complete, production-ready ML systems.

*   **System Design:** Architect scalable and maintainable ML systems.
*   **Data Engineering:** Build robust data pipelines for your AI.
*   **Model Deployment:** Deploy your models into production.
*   **MLOps & Monitoring:** Ensure reliable, continuously operating systems.
*   **Edge AI:** Optimize ML for resource-constrained devices (mobile, embedded, IoT).

---

## Support the Mission

This project is dedicated to expanding access to AI education. Your support directly helps learners worldwide:

*   **Star the Repository:** Show your support and help us reach our funding goals.
*   **Fund the Mission:**  Donate via [Open Collective](https://opencollective.com/mlsysbook) to support educators and provide TinyML kits.

---

## Community and Resources

*   [**Main Site**](https://mlsysbook.org): Complete learning platform.
*   [**TinyTorch**](https://mlsysbook.org/tinytorch): Educational ML framework.
*   [**Discussions**](https://github.com/harvard-edge/cs249r_book/discussions): Ask questions, share insights.
*   [**Community**](https://mlsysbook.org/community): Join the global learning community.

---

## For Different Audiences

### 🎓 Students

*   [📖 Read Online](https://mlsysbook.ai)
*   [📄 Download PDF](https://mlsysbook.ai/Machine-Learning-Systems.pdf)
*   [🧪 Try Hands-on Labs](https://mlsysbook.org)

### 👩‍🏫 Educators

*   [📋 Course Materials](https://mlsysbook.org)
*   [🎯 Instructor Resources](https://mlsysbook.org)
*   [💡 Teaching Guides](https://mlsysbook.org)

### 🛠️ Contributors

*   [🤝 Contribution Guide](docs/contribute.md)
*   [⚡ Development Setup](#development)
*   [💬 Join Discussions](https://github.com/harvard-edge/cs249r_book/discussions)

---

## Quick Start

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

# Quick Setup (Recommended)
./binder setup
./binder doctor

# Fast Development Workflow
./binder preview intro
./binder build intro
./binder build
./binder help
```

---

## Contributing

We welcome contributions from the global community!  See [docs/contribute.md](docs/contribute.md) for details.

---

## Development

### Book Binder CLI (Recommended)

Use the Book Binder CLI for a streamlined development workflow.

```bash
./binder preview intro  # Preview a chapter
./binder build           # Build the complete website
./binder pdf             # Build the PDF
./binder epub            # Build the EPUB
./binder help            # View all commands
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

*   [⚡ Binder CLI Guide](docs/BINDER.md)
*   [📋 Development Guide](docs/DEVELOPMENT.md)
*   [🛠️ Build Instructions](docs/BUILD.md)
*   [🤝 Contribution Guidelines](docs/contribute.md)

### Publishing

```bash
./binder publish
```

### Getting Started

```bash
./binder setup
./binder doctor
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

## Contributors

[See the full list of contributors](https://github.com/harvard-edge/cs249r_book/graphs/contributors).

---

<div align="center">
**Made with ❤️ for AI learners worldwide.**
</div>