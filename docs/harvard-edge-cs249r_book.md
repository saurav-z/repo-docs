# Machine Learning Systems: Build Real-World AI Systems (Open Source)

**Master the art of building and deploying AI systems with this comprehensive, open-source textbook.** Originally developed at Harvard University, this resource provides a complete guide to machine learning systems engineering.  [Explore the original repository](https://github.com/harvard-edge/cs249r_book).

---

## Key Features

*   **Comprehensive Coverage:** Learn system design, data engineering, model deployment, MLOps, and edge AI.
*   **Hands-on Labs:** Build practical AI systems, from edge devices to cloud deployment.
*   **Open Source & Community-Driven:**  Access the textbook, code, and join a global community of learners.
*   **For Students and Educators:**  Course materials, instructor resources, and teaching guides are available.
*   **Constantly Updated:** Benefit from the latest advancements in AI systems engineering.

---

## What You'll Learn

This book goes beyond training models, teaching you the full stack of real-world ML systems.

*   **System Design:** Build scalable, maintainable ML architectures.
*   **Data Engineering:** Create robust pipelines for data collection, labeling, and processing.
*   **Model Deployment:** Deploy production-ready systems from prototypes.
*   **MLOps & Monitoring:** Build reliable, continuously operating systems.
*   **Edge AI:** Deploy resource-efficient AI on mobile, embedded, and IoT devices.

---

##  Access & Resources

*   **📖 Read Online:** [mlsysbook.ai](https://mlsysbook.ai)
*   **💾 Download PDF:** [mlsysbook.ai/Machine-Learning-Systems.pdf](https://mlsysbook.ai/Machine-Learning-Systems.pdf)
*   **🌐 Explore Ecosystem:** [mlsysbook.org](https://mlsysbook.org)
*   **📚 Hardcopy Edition:** Coming 2026 via MIT Press!

---

## Support the Project

<div align="center">

###  Show Your Support

**Star this repository** to help demonstrate the value of open AI education.

[![Stars](https://img.shields.io/github/stars/harvard-edge/cs249r_book?style=for-the-badge&logo=github&color=gold)](https://github.com/harvard-edge/cs249r_book/stargazers)

**Goal:** 10,000 stars = $100,000 in additional education funding

[**⭐ Star Now**](https://github.com/harvard-edge/cs249r_book)

### Fund the Mission

Help democratize AI education by supporting educators globally, providing TinyML kits, funding workshops, and sustaining our open-source infrastructure.

[![Open Collective](https://img.shields.io/badge/💝%20Support%20AI%20Education-Open%20Collective-blue.svg?style=for-the-badge)](https://opencollective.com/mlsysbook)

_Every contribution, from $15/month to sponsor a learner to $250 for workshops, makes a difference._

</div>

---

## Community & Resources

*   [📚 **Main Site**](https://mlsysbook.org): Complete learning platform
*   [🔥 **TinyTorch**](https://mlsysbook.org/tinytorch): Educational ML framework
*   [💬 **Discussions**](https://github.com/harvard-edge/cs249r_book/discussions): Ask questions, share insights
*   [👥 **Community**](https://mlsysbook.org/community): Join our global learning community

---

## Contributing

We welcome contributions from the global community!

### Ways to Contribute
*   **📝 Content**: Suggest edits, improvements, or new examples
*   **🛠️ Tools**: Enhance development scripts and automation
*   **🎨 Design**: Improve figures, diagrams, and visual elements
*   **🌍 Localization**: Translate content
*   **🔧 Infrastructure**: Help with build systems and deployment

[**Start Contributing →**](docs/contribute.md)

---

## Quick Start

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

## 🛠️ Development

### Book Binder CLI (Recommended)

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

### Documentation
- [⚡ Binder CLI Guide](docs/BINDER.md) — Fast development with the Book Binder
- [📋 Development Guide](docs/DEVELOPMENT.md) — Comprehensive setup and workflow
- [🛠️ Maintenance Guide](docs/MAINTENANCE_GUIDE.md) — Daily tasks and troubleshooting  
- [🔨 Build Instructions](docs/BUILD.md) — Detailed build process
- [🤝 Contribution Guidelines](docs/contribute.md) — How to contribute effectively

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

---

<div align="center">

**Made with ❤️ for AI learners worldwide**

Our goal is to educate 1 million AI systems engineers for the future at the edge of AI.
</div>