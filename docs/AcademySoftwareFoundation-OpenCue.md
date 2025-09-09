![OpenCue](/images/opencue_logo_with_text.png)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

# OpenCue: The Open-Source Render Management System for VFX and Animation

**OpenCue**, developed by the Academy Software Foundation, is an open-source render management system designed to streamline the rendering process for visual effects and animation pipelines.  [Learn more about OpenCue on GitHub](https://github.com/AcademySoftwareFoundation/OpenCue).

## Key Features of OpenCue

*   **Proven in Production:** Built on the technology behind Sony Pictures Imageworks' in-house render manager, used on hundreds of films.
*   **Scalable Architecture:** Supports a vast number of concurrent machines for efficient rendering.
*   **Flexible Resource Allocation:** Utilize tagging systems to assign jobs to specific machine types.
*   **Centralized Rendering:** Jobs are processed on a central render farm, freeing up artist workstations.
*   **Native Multi-Threading:** Supports Katana, Prman, and Arnold for optimized performance.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid environments.
*   **Granular Control:** Split hosts into numerous procs, each with reserved cores and memory.
*   **Automated Booking:** Integrated with automated booking systems for efficient resource management.
*   **Unlimited Scalability:** No limit on the number of procs a job can have.

## Getting Started with OpenCue

### Quick Installation and Testing

The [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) provides an easy way to run a test OpenCue deployment locally using Docker containers or Python virtual environments. This is ideal for experimentation and learning.

### Full Installation

System administrators can find detailed guides for deploying OpenCue components and installing dependencies in the [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

## Documentation

Comprehensive documentation is available to guide you through the installation, usage, and administration of OpenCue.

### Building and Testing Documentation

To contribute to the OpenCue documentation, follow these steps:

1.  **Build and validate the documentation**:
    ```bash
    ./docs/build.sh
    ```
2.  **Install bundler binstubs (if needed)**
    ```bash
    cd docs/
    bundle binstubs --all
    ```
3.  **Run the documentation locally**:
    ```bash
    cd docs/
    bundle exec jekyll serve --livereload
    ```
4.  **Preview the documentation**
    Open http://localhost:4000 in your browser to review your changes.

For detailed documentation setup instructions, testing procedures, and contribution guidelines, see [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md).

**Note:** Documentation is automatically deployed via GitHub Actions ([.github/workflows/docs.yml](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs.yml)) after pull requests are merged.  The updated documentation will be available at https://docs.opencue.io/.

## Meeting Notes

*   **May 2024 onwards:**  [Opencue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home)
*   **Before May 2024:**  [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) folder in the repository.

## Connect with the OpenCue Community

*   **Discussion Forum:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io>.
*   **Slack:** [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q)
*   **Working Group:** Bi-weekly meetings at 2 PM PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).