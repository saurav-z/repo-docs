![OpenCue](/images/opencue_logo_with_text.png)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

## OpenCue: The Open-Source Render Management System for VFX and Animation

OpenCue is a powerful open-source render management system designed to streamline your visual effects and animation workflows.  Visit the [OpenCue GitHub Repository](https://github.com/AcademySoftwareFoundation/OpenCue) for the source code and more information.

### Key Features

*   **Scalable Architecture:** Supports a large number of concurrent machines for efficient rendering.
*   **Industry Proven:** Based on the in-house render manager used by Sony Pictures Imageworks for hundreds of films.
*   **Flexible Resource Allocation:** Utilize tagging systems to allocate jobs to specific machine types.
*   **Centralized Rendering:** Jobs are processed on a central render farm, freeing up artist workstations.
*   **Multi-Threading Support:** Native support for Katana, Prman, and Arnold.
*   **Deployment Flexibility:**  Supports multi-facility, on-premises, cloud, and hybrid deployments.
*   **Advanced Resource Management:** Split hosts into numerous "procs" with reserved core and memory.
*   **Integrated Booking & Scalability:** Automated booking and no limits on the number of "procs" a job can have.

### Getting Started

*   **Quick Installation & Testing:**  Set up a local OpenCue environment using the [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) and explore the [Quick Starts documentation](https://www.opencue.io/docs/quick-starts/).

*   **Full Installation:**  System administrators can find detailed guides for deploying OpenCue components and installing dependencies in the [OpenCue Documentation - Getting Started](https://www.opencue.io/docs/getting-started).

### Documentation

Comprehensive OpenCue documentation is built with Jekyll and hosted on GitHub Pages, providing:

*   Installation guides
*   User guides
*   API references
*   Tutorials

**Building and Testing Documentation**

1.  **Build and validate the documentation:**
    ```bash
    ./docs/build.sh
    ```
2.  **Install bundler binstubs (if needed):**
    ```bash
    cd docs/
    bundle binstubs --all
    ```
3.  **Run the documentation locally:**
    ```bash
    cd docs/
    bundle exec jekyll serve --livereload
    ```
4.  **Preview the documentation:** Open http://localhost:4000 in your browser to review your changes.

Refer to [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md) for detailed setup instructions, testing procedures, and contribution guidelines.  Documentation updates are automatically deployed to https://docs.opencue.io/ after pull request merges.

### Community & Support

*   **Meeting Notes:** Access meeting notes on the [OpenCue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home) (May 2024 onwards) or in the [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) folder (pre-May 2024).
*   **Join the Community:** Collaborate with other contributors and users in the [OpenCue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   **Bi-Weekly Meetings:** Join the Working Group meetings on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6) at 2 PM PST.
*   **Mailing List:** Discuss OpenCue with users and admins by joining the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or emailing <opencue-user@lists.aswf.io>.

### Contributors

<a href="https://github.com/AcademySoftwareFoundation/OpenCue/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AcademySoftwareFoundation/OpenCue" alt="Contributors image" />
</a>