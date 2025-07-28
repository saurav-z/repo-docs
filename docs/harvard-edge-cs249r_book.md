# Machine Learning Systems: Build Real-World AI

**Master the principles and practices of engineering intelligent systems with this open-source textbook and learn to build production-ready AI solutions.**  Dive deep into the core concepts, practical techniques, and real-world applications of machine learning systems.  [Explore the original repository](https://github.com/harvard-edge/cs249r_book).

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

## About This Book: Your Guide to Production AI

This open-source textbook, developed from Harvard University's CS249r course, teaches you how to build and deploy real-world AI systems. It's used by universities and students worldwide and provides a comprehensive understanding of the entire AI systems lifecycle.

**Key Features:**

*   **Open-Source:** Free and accessible to all.
*   **Practical Focus:** Hands-on labs and real-world examples.
*   **Full-Stack Coverage:** Covers system design, data engineering, model deployment, MLOps, and edge AI.
*   **Community-Driven:** Active discussions and a global learning community.
*   **Continuous Updates:** Content is always being improved and expanded.

### Why This Book Exists

This book addresses the critical need for engineers who can build and maintain the systems that power AI. It goes beyond model training, focusing on efficient, scalable, and sustainable AI systems.

---

## What You'll Learn: From Prototype to Production

Go beyond the model—learn how to build, deploy, and manage full-stack ML systems with this comprehensive guide.

**Topics Covered:**

*   **System Design:** Architecture and design for scalable ML.
*   **Data Engineering:** Building robust data pipelines for training and production.
*   **Model Deployment:** Deploying models to production environments.
*   **MLOps & Monitoring:** Managing and maintaining reliable AI systems.
*   **Edge AI:** Optimizing AI for resource-constrained devices (mobile, embedded, IoT).

---

## Support the Mission

<div align="center">

### Support AI Education

Help us expand access to AI education globally.

**Show Your Support:**

*   **Star this Repository:** Help us demonstrate the value of this educational resource.
    [![Stars](https://img.shields.io/github/stars/harvard-edge/cs249r_book?style=for-the-badge&logo=github&color=gold)](https://github.com/harvard-edge/cs249r_book/stargazers)
*   [**⭐ Star Now**](https://github.com/harvard-edge/cs249r_book)

**Fund the Mission:**

*   **Open Collective:** Support educators worldwide, providing resources like TinyML kits and funding workshops.
    [![Open Collective](https://img.shields.io/badge/💝%20Support%20AI%20Education-Open%20Collective-blue.svg?style=for-the-badge)](https://opencollective.com/mlsysbook)

</div>

---

## Community & Resources

*   [📚 **Main Site**](https://mlsysbook.org): Complete learning platform.
*   [🔥 **TinyTorch**](https://mlsysbook.org/tinytorch): Educational ML framework.
*   [💬 **Discussions**](https://github.com/harvard-edge/cs249r_book/discussions): Ask questions, share insights.
*   [👥 **Community**](https://mlsysbook.org/community): Join our global learning community.

---

## For Different Audiences

### 🎓 Students
*   [📖 Read online](https://mlsysbook.ai)
*   [📄 Download PDF](https://mlsysbook.ai/Machine-Learning-Systems.pdf)
*   [🧪 Try hands-on labs](https://mlsysbook.org)

### 👩‍🏫 Educators
*   [📋 Course materials](https://mlsysbook.org)
*   [🎯 Instructor resources](https://mlsysbook.org)
*   [💡 Teaching guides](https://mlsysbook.org)

### 🛠️ Contributors
*   [🤝 Contribution guide](contribute.md)
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
make preview      # Start development server
```

---

## 🤝 Contributing

We welcome contributions! Improve the book, tools, and content for everyone.

**Ways to Contribute:**

*   📝 **Content:** Suggest edits, improvements, or new examples.
*   🛠️ **Tools:** Enhance development scripts and automation.
*   🎨 **Design:** Improve figures, diagrams, and visual elements.
*   🌍 **Localization:** Translate content for global accessibility.
*   🔧 **Infrastructure:** Help with build systems and deployment.

**Quality Standards:**

*   ✅ **Pre-commit validation:** Automatic cleanup and checks.
*   📋 **Content review:** Formatting and style validation.
*   🧪 **Testing:** Build and link verification.
*   👥 **Peer review:** Community feedback.

[**Start Contributing →**](docs/contribute.md)

---

## 🛠️ Development

### Quick Commands
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
├── book/                    # Main book content (Quarto)
│   ├── contents/            # Chapter content
│   │   ├── core/            # Core chapters
│   │   ├── labs/            # Hands-on labs
│   │   ├── frontmatter/     # Preface, acknowledgments
│   │   └── parts/           # Book parts and sections
│   ├── _quarto.yml          # Book configuration
│   ├── index.qmd            # Main entry point
│   └── assets/              # Images, styles, media
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
│   ├── lua/                 # Lua scripts
│   └── tex/                 # LaTeX templates
├── assets/                  # Global assets (covers, icons)
├── docs/                    # Documentation
│   ├── DEVELOPMENT.md       # Development guide
│   ├── MAINTENANCE_GUIDE.md # Daily workflow guide
│   ├── BUILD.md             # Build instructions
│   └── contribute.md        # Contribution guidelines
└── Makefile                 # Development commands
```

### Documentation
*   [📋 Development Guide](docs/DEVELOPMENT.md) — Comprehensive setup and workflow
*   [🛠️ Maintenance Guide](docs/MAINTENANCE_GUIDE.md) — Daily tasks and troubleshooting
*   [🔨 Build Instructions](docs/BUILD.md) — Detailed build process
*   [🤝 Contribution Guidelines](docs/contribute.md) — How to contribute effectively

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

## 👥 Contributors

A big thank you to all contributors who make this project a success!

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?s=100" width="100px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/hzeljko"><img src="https://avatars.githubusercontent.com/hzeljko?s=100" width="100px;" alt="Zeljko Hrcek"/><br /><sub><b>Zeljko Hrcek</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/jasonjabbour"><img src="https://avatars.githubusercontent.com/jasonjabbour?s=100" width="100px;" alt="jasonjabbour"/><br /><sub><b>jasonjabbour</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/Mjrovai"><img src="https://avatars.githubusercontent.com/Mjrovai?s=100" width="100px;" alt="Marcelo Rovai"/><br /><sub><b>Marcelo Rovai</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/uchendui"><img src="https://avatars.githubusercontent.com/uchendui?s=100" width="100px;" alt="Ikechukwu Uchendu"/><br /><sub><b>Ikechukwu Uchendu</b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/kai4avaya"><img src="https://avatars.githubusercontent.com/kai4avaya?s=100" width="100px;" alt="Kai Kleinbard"/><br /><sub><b>Kai Kleinbard</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/Naeemkh"><img src="https://avatars.githubusercontent.com/Naeemkh?s=100" width="100px;" alt="Naeem Khoshnevis"/><br /><sub><b>Naeem Khoshnevis</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/Sara-Khosravi"><img src="https://avatars.githubusercontent.com/Sara-Khosravi?s=100" width="100px;" alt="Sara Khosravi"/><br /><sub><b>Sara Khosravi</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/18jeffreyma"><img src="https://avatars.githubusercontent.com/18jeffreyma?s=100" width="100px;" alt="Jeffrey Ma"/><br /><sub><b>Jeffrey Ma</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/V0XNIHILI"><img src="https://avatars.githubusercontent.com/V0XNIHILI?s=100" width="100px;" alt="Douwe den Blanken"/><br /><sub><b>Douwe den Blanken</b></sub></a><br /></td>
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
      <td align="center" valign="top" width="20%"><a href="https://github.com/sophiacho1"><img src="https://avatars.githubusercontent.com/sophiacho1?s=100" width="100px;" alt="Sophia Cho"/><br /><sub><b>Sophia Cho</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/andreamurillomtz"><img src="https://avatars.githubusercontent.com/andreamurillomtz?s=100" width="100px;" alt="Andrea"/><br /><sub><b>Andrea</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/alxrod"><img src="https://avatars.githubusercontent.com/alxrod?s=100" width="100px;" alt="Alex Rodriguez"/><br /><sub><b>Alex Rodriguez</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/korneelf1"><img src="https://avatars.githubusercontent.com/korneelf1?s=100" width="100px;" alt="Korneel Van den Berghe"/><br /><sub><b>Korneel Van den Berghe</b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/colbybanbury"><img src="https://avatars.githubusercontent.com/colbybanbury?s=100" width="100px;" alt="Colby Banbury"/><br /><sub><b>Colby Banbury</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/zishenwan"><img src="https://avatars.githubusercontent.com/zishenwan?s=100" width="100px;" alt="Zishen Wan"/><br /><sub><b>Zishen Wan</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/mmaz"><img src="https://avatars.githubusercontent.com/mmaz?s=100" width="100px;" alt="Mark Mazumder"/><br /><sub><b>Mark Mazumder</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/ma3mool"><img src="https://avatars.githubusercontent.com/ma3mool?s=100" width="100px;" alt="Abdulrahman Mahmoud"/><br /><sub><b>Abdulrahman Mahmoud</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/DivyaAmirtharaj"><img src="https://avatars.githubusercontent.com/DivyaAmirtharaj?s=100" width="100px;" alt="Divya Amirtharaj"/><br /><sub><b>Divya Amirtharaj</b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/srivatsankrishnan"><img src="https://avatars.githubusercontent.com/srivatsankrishnan?s=100" width="100px;" alt="Srivatsan Krishnan"/><br /><sub><b>Srivatsan Krishnan</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/arnaumarin"><img src="https://avatars.githubusercontent.com/arnaumarin?s=100" width="100px;" alt="marin-llobet"/><br /><sub><b>marin-llobet</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/James-QiuHaoran"><img src="https://avatars.githubusercontent.com/James-QiuHaoran?s=100" width="100px;" alt="Haoran Qiu"/><br /><sub><b>Haoran Qiu</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/eezike"><img src="https://avatars.githubusercontent.com/eezike?s=100" width="100px;" alt="Emeka Ezike"/><br /><sub><b>Emeka Ezike</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/aptl26"><img src="https://avatars.githubusercontent.com/aptl26?s=100" width="100px;" alt="Aghyad Deeb"/><br /><sub><b>Aghyad Deeb</b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/MichaelSchnebly"><img src="https://avatars.githubusercontent.com/MichaelSchnebly?s=100" width="100px;" alt="Michael Schnebly"/><br /><sub><b>Michael Schnebly</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/ELSuitorHarvard"><img src="https://avatars.githubusercontent.com/ELSuitorHarvard?s=100" width="100px;" alt="ELSuitorHarvard"/><br /><sub><b>ELSuitorHarvard</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/Ekhao"><img src="https://avatars.githubusercontent.com/Ekhao?s=100" width="100px;" alt="Emil Njor"/><br /><sub><b>Emil Njor</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/AditiR-42"><img src="https://avatars.githubusercontent.com/AditiR-42?s=100" width="100px;" alt="Aditi Raju"/><br /><sub><b>Aditi Raju</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/oishib"><img src="https://avatars.githubusercontent.com/oishib?s=100" width="100px;" alt="oishib"/><br /><sub><b>oishib</b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/jared-ni"><img src="https://avatars.githubusercontent.com/jared-ni?s=100" width="100px;" alt="Jared Ni"/><br /><sub><b>Jared Ni</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/jaywonchung"><img src="https://avatars.githubusercontent.com/jaywonchung?s=100" width="100px;" alt="Jae-Won Chung"/><br /><sub><b>Jae-Won Chung</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/leo47007"><img src="https://avatars.githubusercontent.com/leo47007?s=100" width="100px;" alt="Yu-Shun Hsiao"/><br /><sub><b>Yu-Shun Hsiao</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/BaeHenryS"><img src="https://avatars.githubusercontent.com/BaeHenryS?s=100" width="100px;" alt="Henry Bae"/><br /><sub><b>Henry Bae</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/euranofshin"><img src="https://avatars.githubusercontent.com/euranofshin?s=100" width="100px;" alt="Eura Nofshin"/><br /><sub><b>Eura Nofshin</b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/jzhou1318"><img src="https://avatars.githubusercontent.com/jzhou1318?s=100" width="100px;" alt="Jennifer Zhou"/><br /><sub><b>Jennifer Zhou</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/ShvetankPrakash"><img src="https://avatars.githubusercontent.com/ShvetankPrakash?s=100" width="100px;" alt="Shvetank Prakash"/><br /><sub><b>Shvetank Prakash</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/0c931fcfd03cd548d44c90602dd773ba?d=identicon&s=100?s=100" width="100px;" alt="Matthew Stewart"/><br /><sub><b>Matthew Stewart</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/pongtr"><img src="https://avatars.githubusercontent.com/pongtr?s=100" width="100px;" alt="Pong Trairatvorakul"/><br /><sub><b>Pong Trairatvorakul</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/marcozennaro"><img src="https://avatars.githubusercontent.com/marcozennaro?s=100" width="100px;" alt="Marco Zennaro"/><br /><sub><b>Marco Zennaro</b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/aryatschand"><img src="https://avatars.githubusercontent.com/aryatschand?s=100" width="100px;" alt="Arya Tschand"/><br /><sub><b>Arya Tschand</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/arbass22"><img src="https://avatars.githubusercontent.com/arbass22?s=100" width="100px;" alt="Andrew Bass"/><br /><sub><b>Andrew Bass</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/Allen-Kuang"><img src="https://avatars.githubusercontent.com/Allen-Kuang?s=100" width="100px;" alt="Allen-Kuang"/><br /><sub><b>Allen-Kuang</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/BrunoScaglione"><img src="https://avatars.githubusercontent.com/BrunoScaglione?s=100" width="100px;" alt="Bruno Scaglione"/><br /><sub><b>Bruno Scaglione</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/FinAminToastCrunch"><img src="https://avatars.githubusercontent.com/FinAminToastCrunch?s=100" width="100px;" alt="Fin Amin"/><br /><sub><b>Fin Amin</b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/Gjain234"><img src="https://avatars.githubusercontent.com/Gjain234?s=100" width="100px;" alt="Gauri Jain"/><br /><sub><b>Gauri Jain</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/gnodipac886"><img src="https://avatars.githubusercontent.com/gnodipac886?s=100" width="100px;" alt="gnodipac886"/><br /><sub><b>gnodipac886</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/alex-oesterling"><img src="https://avatars.githubusercontent.com/alex-oesterling?s=100" width="100px;" alt="Alex Oesterling"/><br /><sub><b>Alex Oesterling</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/468ef35acc69f3266efd700992daa369?d=identicon&s=100?s=100" width="100px;" alt="Fatima Shah"/><br /><sub><b>Fatima Shah</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/vitasam"><img src="https://avatars.githubusercontent.com/vitasam?s=100" width="100px;" alt="The Random DIY"/><br /><sub><b>The Random DIY</b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/serco425"><img src="https://avatars.githubusercontent.com/serco425?s=100" width="100px;" alt="Sercan Aygün"/><br /><sub><b>Sercan Aygün</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/TheHiddenLayer"><img src="https://avatars.githubusercontent.com/TheHiddenLayer?s=100" width="100px;" alt="TheHiddenLayer"/><br /><sub><b>TheHiddenLayer</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/BravoBaldo"><img src="https://avatars.githubusercontent.com/BravoBaldo?s=100" width="100px;" alt="Baldassarre Cesarano"/><br /><sub><b>Baldassarre Cesarano</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/AbenezerKb"><img src="https://avatars.githubusercontent.com/AbenezerKb?s=100" width="100px;" alt="Abenezer Angamo"/><br /><sub><b>Abenezer Angamo</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/YLab-UChicago"><img src="https://avatars.githubusercontent.com/YLab-UChicago?s=100" width="100px;" alt="yanjingl"/><br /><sub><b>yanjingl</b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/aethernavshulkraven-allain"><img src="https://avatars.githubusercontent.com/aethernavshulkraven-allain?s=100" width="100px;" alt="अरनव शुक्ला &#124; Arnav Shukla"/><br /><sub><b>अरनव शुक्ला &#124; Arnav Shukla</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/YangZhou1997"><img src="https://avatars.githubusercontent.com/YangZhou1997?s=100" width="100px;" alt="Yang Zhou"/><br /><sub><b>Yang Zhou</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/happyappledog"><img src="https://avatars.githubusercontent.com/happyappledog?s=100" width="100px;" alt="happyappledog"/><br /><sub><b>happyappledog</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/abigailswallow"><img src="https://avatars.githubusercontent.com/abigailswallow?s=100" width="100px;" alt="abigailswallow"/><br /><sub><b>abigailswallow</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/arighosh05"><img src="https://avatars.githubusercontent.com/arighosh05?s=100" width="100px;" alt="Aritra Ghosh"/><br /><sub><b>Aritra Ghosh</b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/atcheng