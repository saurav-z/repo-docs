# Machine Learning Systems: Build Real-World AI Systems

**Learn how to build and deploy cutting-edge AI systems with this comprehensive, open-source textbook.** ([Original Repository](https://github.com/harvard-edge/cs249r_book))

<div align="center">

[![Build Status](https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/controller.yml?branch=dev&label=Build)](https://github.com/harvard-edge/cs249r_book/actions/workflows/controller.yml?query=branch%3Adev)
[![Website](https://img.shields.io/website?url=https://mlsysbook.ai&label=Website)](https://mlsysbook.ai)
[![Ecosystem](https://img.shields.io/website?url=https://mlsysbook.org&label=Ecosystem)](https://mlsysbook.org)
[![License](https://img.shields.io/badge/license-CC--BY--NC--SA%204.0-blue)](https://github.com/harvard-edge/cs249r_book/blob/dev/LICENSE)
[![Open Collective](https://img.shields.io/badge/fund%20us-Open%20Collective-blue.svg)](https://opencollective.com/mlsysbook)

**[📖 Read Online](https://mlsysbook.ai)** • **[💾 Download PDF](https://mlsysbook.ai/Machine-Learning-Systems.pdf)** • **[🌐 Explore Ecosystem](https://mlsysbook.org)**

📚 **Hardcopy edition coming 2026 via MIT Press!**

</div>

## Key Features

*   **Comprehensive Coverage:** Master the full stack of real-world ML systems, going beyond just training models.
*   **Practical Focus:** Learn through building, with hands-on labs and real-world examples.
*   **Open Source & Accessible:** Free to read online and download, ensuring global access to AI education.
*   **Community Driven:** Benefit from a thriving community, discussions, and resources.

## What You'll Learn

This book goes beyond model training, teaching you how to build complete, production-ready AI systems.

*   **System Design:** Build scalable and maintainable ML architectures.
*   **Data Engineering:** Create robust data pipelines for collection, labeling, and processing.
*   **Model Deployment:** Deploy your models into production environments.
*   **MLOps & Monitoring:** Implement reliable, continuously operating systems.
*   **Edge AI:** Deploy resource-efficient systems on mobile, embedded, and IoT devices.

## Support the Project

Help us expand AI systems education worldwide!

*   **Star the Repository:** Show your support and help us secure funding for education.

    [![Stars](https://img.shields.io/github/stars/harvard-edge/cs249r_book?style=for-the-badge&logo=github&color=gold)](https://github.com/harvard-edge/cs249r_book/stargazers)
    [**⭐ Star Now**](https://github.com/harvard-edge/cs249r_book)
*   **Fund the Mission:** Support educators globally by donating to our Open Collective, which provides TinyML kits for students and funds workshops.

    [![Open Collective](https://img.shields.io/badge/💝%20Support%20AI%20Education-Open%20Collective-blue.svg?style=for-the-badge)](https://opencollective.com/mlsysbook)

## Resources

*   **[📚 Main Site](https://mlsysbook.org):** Complete learning platform.
*   **[🔥 TinyTorch](https://mlsysbook.org/tinytorch):** Educational ML framework.
*   **[💬 Discussions](https://github.com/harvard-edge/cs249r_book/discussions):** Ask questions and share insights.
*   **[👥 Community](https://mlsysbook.org/community):** Join our global learning community.

## For Different Audiences

*   **🎓 Students:**
    *   [📖 Read online](https://mlsysbook.ai)
    *   [📄 Download PDF](https://mlsysbook.ai/Machine-Learning-Systems.pdf)
    *   [🧪 Try hands-on labs](https://mlsysbook.org)
*   **👩‍🏫 Educators:**
    *   [📋 Course materials](https://mlsysbook.org)
    *   [🎯 Instructor resources](https://mlsysbook.org)
    *   [💡 Teaching guides](https://mlsysbook.org)
*   **🛠️ Contributors:**
    *   [🤝 Contribution guide](docs/contribute.md)
    *   [⚡ Development setup](#development)
    *   [💬 Join discussions](https://github.com/harvard-edge/cs249r_book/discussions)

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
make setup-hooks
make install

# Recommended: Use the binder for development
./binder preview intro
./binder help

# Traditional Make commands
make preview
```

## Contributing

We welcome contributions from the global community!

### Ways to Contribute

*   **📝 Content:** Suggest edits, improvements, or new examples
*   **🛠️ Tools:** Enhance development scripts and automation
*   **🎨 Design:** Improve figures, diagrams, and visual elements
*   **🌍 Localization:** Translate content
*   **🔧 Infrastructure:** Help with build systems and deployment

### Quality Standards

All contributions benefit from automated quality assurance:

*   ✅ Pre-commit validation
*   📋 Content review
*   🧪 Testing
*   👥 Peer review

[**Start Contributing →**](docs/contribute.md)

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

### Make Commands (Traditional)

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
./binder publish
./binder build - html && ./binder build - pdf
```

### Getting Started

```bash
./binder hello
./binder setup
./binder preview intro
```

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