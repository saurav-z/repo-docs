# OpenCue: The Open-Source Render Management System for VFX and Animation

[![OpenCue Logo](/images/opencue_logo_with_text.png)](https://github.com/AcademySoftwareFoundation/OpenCue)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
[![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

**OpenCue is a powerful open-source solution designed to streamline and optimize render management for visual effects (VFX) and animation pipelines.**

## Key Features

*   **Proven in Production:** Used on hundreds of films by Sony Pictures Imageworks.
*   **Highly Scalable Architecture:** Supports numerous concurrent machines and complex jobs.
*   **Flexible Resource Allocation:** Tagging systems for targeted job allocation to specific machine types.
*   **Centralized Rendering:** Jobs processed on a central render farm, freeing up artist workstations.
*   **Native Multi-Threading Support:** Compatible with Katana, Prman, and Arnold.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid environments.
*   **Granular Resource Control:**  Split hosts into procs with reserved cores and memory.
*   **Automated Booking:** Integrated for efficient resource management.
*   **Unlimited Proc Count:** No limitations on the number of procs a job can utilize.

## Getting Started

### Quick Installation and Testing

Explore the [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) for setting up a local OpenCue environment using Docker containers or Python virtual environments. This is perfect for small tests, development, and learning.

### Full Installation

System administrators can find detailed guides for deploying OpenCue components and installing dependencies in the [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

## Documentation

Comprehensive documentation is available at [https://docs.opencue.io](https://docs.opencue.io) and built with Jekyll and hosted on GitHub Pages.  It includes:

*   Installation guides
*   User guides
*   API references
*   Tutorials

**Contributing to Documentation:**  When contributing to OpenCue, update the documentation for any new features or changes. Build and test your documentation locally before submitting a pull request.

## Building and Testing Documentation

1.  **Build and validate the documentation**
    ```bash
    ./docs/build.sh
    ```
2.  **Install bundler binstubs (if needed)**
    ```bash
    cd docs/
    bundle binstubs --all
    ```
3.  **Run the documentation locally**
    ```bash
    cd docs/
    bundle exec jekyll serve --livereload
    ```
4.  **Preview the documentation**

    Open http://localhost:4000 in your browser to review your changes.

For detailed documentation setup instructions, testing procedures, and contribution guidelines, see [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md).

**Note:** Documentation is automatically deployed via GitHub Actions after pull request merges, and is found at [https://docs.opencue.io/](https://docs.opencue.io/).

## Meeting Notes

*   **May 2024 onwards:** [Opencue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home)
*   **Before May 2024:** [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings)

## Contact Us

*   **Discussion Forum:** [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) - <opencue-user@lists.aswf.io>
*   **Slack:** [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q)
*   **Working Group Meetings:** Biweekly at 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6)

**[Visit the OpenCue GitHub Repository](https://github.com/AcademySoftwareFoundation/OpenCue) for the latest updates and contributions.**