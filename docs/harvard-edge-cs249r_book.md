# Machine Learning Systems: Your Guide to Building Real-World AI 

**Master the art of building AI systems, from edge devices to cloud deployment, with this comprehensive, open-source textbook.** Developed from Harvard University's CS249r course, this project is available on [GitHub](https://github.com/harvard-edge/cs249r_book).

---

## Key Features:

*   **Comprehensive Coverage:** Learn system design, data engineering, model deployment, MLOps, Edge AI, and more.
*   **Hands-on Learning:** Build real-world AI systems, going beyond just model training.
*   **Open-Source & Accessible:** Free online resources and downloadable PDFs, empowering learners worldwide.
*   **Community Driven:** Join a global community and contribute to the project.
*   **Future-Proof Skills:** Develop expertise in the critical skills needed to build efficient, scalable, and sustainable AI systems.

---

## What You'll Learn:

This book equips you with the skills to build production-ready ML systems, covering the full stack:

*   **System Design:** Architect scalable and maintainable ML systems.
*   **Data Engineering:** Build robust pipelines for data collection, processing, and labeling.
*   **Model Deployment:** Deploy models into production environments from prototype.
*   **MLOps & Monitoring:** Implement reliable, continuously operating AI systems.
*   **Edge AI:** Deploy and optimize AI models on resource-constrained devices (mobile, IoT).

---

## Support the Project:

### Star the Repository:

Help demonstrate the value of open AI education!

[![Stars](https://img.shields.io/github/stars/harvard-edge/cs249r_book?style=for-the-badge&logo=github&color=gold)](https://github.com/harvard-edge/cs249r_book/stargazers)

**[⭐ Star Now](https://github.com/harvard-edge/cs249r_book)**

### Fund the Mission:

Support AI education globally and empower the next generation of AI systems engineers:

[![Open Collective](https://img.shields.io/badge/💝%20Support%20AI%20Education-Open%20Collective-blue.svg?style=for-the-badge)](https://opencollective.com/mlsysbook)

---

## Resources:

*   **Main Site:** [https://mlsysbook.org](https://mlsysbook.org)
*   **TinyTorch (Educational ML Framework):** [https://mlsysbook.org/tinytorch](https://mlsysbook.org/tinytorch)
*   **Discussions:** [https://github.com/harvard-edge/cs249r_book/discussions](https://github.com/harvard-edge/cs249r_book/discussions)
*   **Community:** [https://mlsysbook.org/community](https://mlsysbook.org/community)

---

##  For Different Audiences:

### Students:

*   📖 **Read online:** [https://mlsysbook.ai](https://mlsysbook.ai)
*   📄 **Download PDF:** [https://mlsysbook.ai/Machine-Learning-Systems.pdf](https://mlsysbook.ai/Machine-Learning-Systems.pdf)
*   🧪 **Try hands-on labs:** [https://mlsysbook.org](https://mlsysbook.org)

### Educators:

*   📋 **Course materials:** [https://mlsysbook.org](https://mlsysbook.org)
*   🎯 **Instructor resources:** [https://mlsysbook.org](https://mlsysbook.org)
*   💡 **Teaching guides:** [https://mlsysbook.org](https://mlsysbook.org)

### Contributors:

*   🤝 **Contribution guide:** `docs/contribute.md`
*   ⚡ **Development setup:**  See "Development" Section
*   💬 **Join discussions:** [https://github.com/harvard-edge/cs249r_book/discussions](https://github.com/harvard-edge/cs249r_book/discussions)

---

## Quick Start

### For Readers:

```bash
# Read online
open https://mlsysbook.ai

# Download PDF
curl -O https://mlsysbook.ai/Machine-Learning-Systems.pdf
```

### For Contributors:

```bash
git clone https://github.com/harvard-edge/cs249r_book.git
cd cs249r_book
make setup-hooks  # Setup automated quality controls
make install      # Install dependencies
./binder preview intro    # Fast chapter development
./binder help            # See all commands
```

---

## Contributing

**We welcome contributions!**

### Ways to Contribute:

*   📝 Content: Suggest edits, improvements, new examples.
*   🛠️ Tools: Enhance development scripts and automation.
*   🎨 Design: Improve figures, diagrams, and visuals.
*   🌍 Localization: Translate content for global accessibility.
*   🔧 Infrastructure: Assist with build and deployment.

### Quality Standards:

*   ✅ Pre-commit validation
*   📋 Content review
*   🧪 Testing
*   👥 Peer review

[**Start Contributing →**](docs/contribute.md)

---

## Development

### Book Binder CLI (Recommended)

```bash
./binder build intro html             # Build single chapter
./binder preview intro                # Build and preview chapter
./binder build * html                 # Build complete website
./binder build * pdf                  # Build complete PDF
```

### Make Commands (Traditional)

```bash
make build
make build-pdf
make preview
```

### Project Structure

```
MLSysBook/
├── binder                   # ⚡ Fast development CLI (recommended)
├── book/                    # Main book content (Quarto)
│   ├── contents/            # Chapter content
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
│   ├── utilities/           # General utilities
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

*   [⚡ Binder CLI Guide](docs/BINDER.md)
*   [📋 Development Guide](docs/DEVELOPMENT.md)
*   [🛠️ Maintenance Guide](docs/MAINTENANCE_GUIDE.md)
*   [🔨 Build Instructions](docs/BUILD.md)
*   [🤝 Contribution Guidelines](docs/contribute.md)

### Publishing

```bash
./binder publish "Description" COMMIT_HASH
```

### Getting Started

```bash
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

This work is licensed under **Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International (CC BY-NC-SA 4.0)**.

---

<div align="center">

**Made with ❤️ for AI learners worldwide**

</div>