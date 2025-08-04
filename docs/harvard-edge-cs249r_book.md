# Machine Learning Systems: Build Real-World AI Systems

**Learn how to build and deploy cutting-edge AI systems with this open-source textbook and its comprehensive resources.** ([Original Repository](https://github.com/harvard-edge/cs249r_book))

---

## Key Features

*   **Comprehensive Coverage:** Master the full stack of ML systems, from system design to edge AI deployment.
*   **Hands-on Learning:** Go beyond theory with practical labs and real-world examples.
*   **Open Source & Collaborative:** Benefit from a community-driven approach and contribute to the project.
*   **Free & Accessible:** Access the book online or download the PDF for free.
*   **For All Audiences:** Whether you are a student, educator, or contributor, the book has resources and guidance for you.

---

## What You'll Learn

This book equips you with the knowledge and skills to build and deploy real-world AI systems.

*   **System Design:** Architect scalable and maintainable ML systems.
*   **Data Engineering:** Build robust pipelines for data collection, labeling, and processing.
*   **Model Deployment:** Take your models from prototype to production-ready systems.
*   **MLOps & Monitoring:** Ensure reliability and continuous operation of your systems.
*   **Edge AI:** Deploy resource-efficient AI on mobile, embedded, and IoT devices.

---

## Support the Project

Help us expand access to AI systems education worldwide!

*   **Star the Repository:** Show your support and help us reach our goal of 10,000 stars to secure additional funding. [⭐ Star Now](https://github.com/harvard-edge/cs249r_book)
*   **Fund the Mission:** Support educators globally, especially in the Global South, by contributing to TinyML kits and workshops. [💝 Support AI Education](https://opencollective.com/mlsysbook)

---

## Resources

*   **Main Site:** [📚 **Main Site**](https://mlsysbook.org) - The complete learning platform
*   **Educational ML Framework:** [🔥 **TinyTorch**](https://mlsysbook.org/tinytorch)
*   **Discussions:** [💬 **Discussions**](https://github.com/harvard-edge/cs249r_book/discussions) - Ask questions, share insights
*   **Community:** [👥 **Community**](https://mlsysbook.org/community) - Join our global learning community

---

## Getting Started

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

## Contributing

Join our global community and help improve the book!

*   **Content:** Suggest edits, improvements, or new examples.
*   **Tools:** Enhance development scripts and automation.
*   **Design:** Improve figures, diagrams, and visual elements.
*   **Localization:** Translate content for global accessibility.
*   **Infrastructure:** Help with build systems and deployment.

[**Start Contributing →**](docs/contribute.md)

---

## Development

*   **Book Binder CLI:** A fast development CLI for streamlined building and iteration.
*   **Make Commands:** Traditional development commands are also available.
*   **Project Structure:** Understand the structure of the project for efficient contribution.
*   **Documentation:** Access the guide for the CLI, the development workflow, the maintenance tasks, the build instructions and contribution guidelines.

---

## Publishing

*   **Command-line trigger:**
    ```bash
    ./binder publish "Description" COMMIT_HASH
    ```
*   **Interactive wizard:**
    ```bash
    ./binder publish
    ```
*   **Manual steps:**
    ```bash
    ./binder build - html && ./binder build - pdf
    # Then copy PDF to assets and push to main
    ```

**Publishing Options:**
*   **`./binder publish`** — Unified command with interactive and command-line modes
*   **Web Interface** — Manual trigger via GitHub Actions UI

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

This work is licensed under **Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International** (CC BY-NC-SA 4.0). You may share and adapt the material for non-commercial purposes with appropriate credit.

---

<div align="center">

**Made with ❤️ for AI learners worldwide**

Our goal is to educate 1 million AI systems engineers for the future at the edge of AI.
</div>