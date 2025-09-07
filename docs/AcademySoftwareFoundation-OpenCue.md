![OpenCue](/images/opencue_logo_with_text.png)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

# OpenCue: The Open-Source Render Management System for VFX and Animation

**OpenCue empowers studios to efficiently manage and scale their rendering pipelines for visual effects and animation projects.**

[View the original repository on GitHub](https://github.com/AcademySoftwareFoundation/OpenCue)

## Key Features

*   **Production-Proven:** Built upon the render manager used by Sony Pictures Imageworks for hundreds of films.
*   **Highly Scalable:** Supports numerous concurrent machines, ideal for large-scale productions.
*   **Flexible Resource Allocation:**  Utilize tagging systems for job allocation to specific machine types.
*   **Centralized Processing:**  Jobs are processed on a central render farm, freeing up artist workstations.
*   **Native Multi-Threading:** Supports Katana, Prman, and Arnold, optimizing performance.
*   **Deployment Flexibility:** Works across multi-facility, on-premises, cloud, and hybrid environments.
*   **Granular Resource Control:**  Split hosts into procs with reserved core and memory requirements.
*   **Integrated Automation:** Includes automated booking capabilities.
*   **Unlimited Scalability:**  No practical limit on the number of procs a job can utilize.

## Getting Started

### Quick Installation with Sandbox

The OpenCue sandbox environment provides an easy way to run a test OpenCue deployment locally using Docker containers or Python virtual environments. This is perfect for experimentation and learning.
 
-   Learn how to set up the sandbox environment: [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md)
-   For more details, see the [Quick Starts](https://www.opencue.io/docs/quick-starts/).

### Full Installation

Detailed guides for system administrators on deploying OpenCue components and installing dependencies are available in the [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

## Documentation

Comprehensive OpenCue documentation is available, covering installation, user guides, API references, and tutorials.

*   Explore the official [OpenCue Documentation](https://www.opencue.io/docs/)

### Contributing to the Documentation

When contributing to OpenCue, please update the documentation for new features or changes.

**Building and Testing Documentation:**

1.  **Build and validate the documentation**
    ```bash
    ./docs/build.sh
    ```
2.  **Install bundler binstubs (if needed)**

    If you encounter permission errors when installing to system directories:
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

**Note:** Documentation is automatically deployed via GitHub Actions ([.github/workflows/docs.yml](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs.yml)) after pull request merges. The updated documentation is available at https://docs.opencue.io/.

## Community & Support

*   **Discussion Forum:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io>.
*   **Slack:** Join the [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   **Meetings:** Working Group meets biweekly at 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).
*   **Meeting Notes:** All Opencue meeting notes from May 2024 onward are stored on the [Opencue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home).
    For meeting notes before May 2024, please refer to the Opencue repository in the [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) folder.