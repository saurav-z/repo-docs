![OpenCue](/images/opencue_logo_with_text.png)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

# OpenCue: The Open-Source Render Management System for VFX and Animation

**OpenCue** empowers visual effects and animation studios to efficiently manage and scale their rendering pipelines. [Learn more about OpenCue on GitHub](https://github.com/AcademySoftwareFoundation/OpenCue).

## Key Features

*   **Production-Proven:** Based on the render manager used by Sony Pictures Imageworks for hundreds of films.
*   **Scalable Architecture:** Supports numerous concurrent machines and complex workloads.
*   **Resource Management:**
    *   Tagging systems for allocating jobs to specific machine types.
    *   Splitting hosts into multiple *procs* with reserved resources.
    *   No limit on the number of procs a job can utilize.
*   **Flexible Deployment:** Supports multi-facility, on-premises, cloud, and hybrid environments.
*   **Optimized Performance:** Native multi-threading support for Katana, PRMan, and Arnold.
*   **Automated Booking:** Integrated functionality for efficient resource allocation.
*   **Centralized Rendering:** Jobs processed on a central render farm, freeing up artist workstations.

## Get Started

### Sandbox Environment

Quickly experiment with OpenCue using the sandbox environment.  The sandbox allows you to run a test OpenCue deployment locally, with all components running in separate Docker containers or Python virtual environments. This is ideal for experimentation and learning.

*   **Learn More:**  [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md)
*   **Run the Sandbox:** Follow the instructions at https://www.opencue.io/docs/quick-starts/

### Full Installation

System administrators can find detailed guides for deploying OpenCue components and installing dependencies in the comprehensive [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

## Documentation

Comprehensive documentation is available to guide you through OpenCue installation, usage, and administration. Explore installation guides, user guides, API references, and tutorials.

### Building and Testing Documentation

Contribute to OpenCue by updating the documentation for new features or changes.

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

4.  **Preview the documentation:**
    Open http://localhost:4000 in your browser to review your changes.

For detailed documentation setup instructions, testing procedures, and contribution guidelines, see [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md).

**Note:** Documentation is automatically deployed to https://docs.opencue.io/ after pull requests are merged.

## Meeting Notes

*   **May 2024 Onward:**  [Opencue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home)
*   **Before May 2024:** [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) in the OpenCue repository.

## Contact & Community

*   **Discussion Forum (Users & Admins):** [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or <opencue-user@lists.aswf.io>
*   **Slack Channel:** [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q)
*   **Bi-weekly Working Group Meeting:** 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6)