![OpenCue](https://github.com/AcademySoftwareFoundation/OpenCue/raw/master/images/opencue_logo_with_text.png)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

# OpenCue: Your Open-Source Solution for Efficient VFX and Animation Rendering

OpenCue is a powerful, open-source render management system designed to streamline complex visual effects and animation pipelines. This allows studios to efficiently manage and scale their rendering workloads. Learn more on the [OpenCue GitHub Repository](https://github.com/AcademySoftwareFoundation/OpenCue).

## Key Features of OpenCue:

*   **Scalable Architecture:** Supports numerous concurrent machines for handling large-scale rendering projects.
*   **Job Management:** Breaks down complex jobs into individual tasks for efficient resource allocation.
*   **Resource Allocation:** Utilizes tagging systems to direct jobs to specific machine types.
*   **Centralized Rendering:**  Jobs are processed on a central render farm, freeing up artist workstations.
*   **Native Multi-threading:** Supports Katana, Prman, and Arnold for optimized rendering.
*   **Deployment Flexibility:**  Offers support for multi-facility, on-premises, cloud, and hybrid deployments.
*   **Flexible Resource Control:**  Allows splitting a host into numerous [procs](https://www.opencue.io/docs/concepts/glossary/#proc) with reserved cores and memory.
*   **Automated Booking:** Integrated automated booking features.
*   **No Job Limits:**  No limitations on the number of [procs](https://www.opencue.io/docs/concepts/glossary/#proc) a job can utilize.
*   **Proven Performance:** Used by Sony Pictures Imageworks on hundreds of films.

## Getting Started with OpenCue

### Quick Installation (Sandbox Environment)

The OpenCue sandbox environment provides an easy way to set up a local OpenCue deployment for testing, development, and learning. All components run in separate Docker containers or Python virtual environments.

*   **Sandbox Benefits:** Ideal for small tests, development work, and for those new to OpenCue.
*   **Quick Start Documentation:** See the [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) and  [OpenCue documentation](https://www.opencue.io/docs/quick-starts/) for detailed instructions.

### Full Installation

System administrators can find detailed guides for deploying OpenCue components and installing dependencies in the [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

## Documentation & Resources

### OpenCue Documentation

Comprehensive documentation is available, including installation guides, user guides, API references, and tutorials to help you get started with OpenCue.  All documentation is built with Jekyll and hosted on GitHub Pages at https://docs.opencue.io/.

*   **Documentation Updates:** Please update the documentation for any new features or changes and include relevant documentation updates in each pull request.
*   **Contribution Guidelines:** For detailed documentation setup instructions, testing procedures, and contribution guidelines, see [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md).

#### Building and Testing Documentation

To build and test the documentation before submitting your PR:

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

    Open http://localhost:4000 in your browser.

### Meeting Notes

*   **Recent Notes:** Starting from May 2024, all Opencue meeting notes are stored on the [Opencue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home).
*   **Older Notes:** For meeting notes before May 2024, please refer to the [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) folder in the repository.

### Contact & Community

*   **User Forum:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io> for discussions.
*   **Slack Channel:** Join the [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   **Working Group:** The Working Group meets biweekly at 2 PM PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).