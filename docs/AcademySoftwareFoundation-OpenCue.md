![OpenCue](/images/opencue_logo_with_text.png)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

# OpenCue: Your Open-Source Solution for VFX & Animation Render Management

**OpenCue** is a robust, open-source render management system designed to streamline and accelerate visual effects and animation pipelines, allowing studios to efficiently manage rendering jobs at scale.  Check out the [original repo here](https://github.com/AcademySoftwareFoundation/OpenCue).

## Key Features

*   **Scalable Architecture:** Supports numerous concurrent machines for efficient rendering.
*   **Advanced Tagging:** Allocate specific jobs to designated machine types.
*   **Centralized Rendering:** Jobs processed on a central render farm, freeing up artist workstations.
*   **Native Multi-Threading:** Optimized for Katana, Prman, and Arnold.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid deployments.
*   **Fine-Grained Resource Control:** Split hosts into numerous procs with reserved core and memory requirements.
*   **Integrated Booking:** Streamlined automated booking process.
*   **Unlimited Scalability:** No job limit on the number of procs a job can have.
*   **Industry Proven:**  Powering hundreds of films, originally developed by Sony Pictures Imageworks.

## Installation & Getting Started

### Quick Installation & Testing

Get up and running quickly with the [OpenCue sandbox environment](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md).  This environment offers an easy way to run a test OpenCue deployment locally, with all components running in separate Docker containers or Python virtual environments.

*   Ideal for small tests, development work, and for those new to OpenCue.
*   [Quick Starts documentation](https://www.opencue.io/docs/quick-starts/) provides further guidance.

### Full Installation

For system administrators, detailed guides for deploying OpenCue components and installing dependencies are available in the [OpenCue Documentation - Getting Started](https://www.opencue.io/docs/getting-started).

## Documentation & Contribution

Comprehensive documentation is available at [https://docs.opencue.io/](https://docs.opencue.io/).

### Documentation Updates

When contributing to OpenCue, please ensure that documentation is updated to reflect any new features or changes.

### Building & Testing Documentation

Follow these steps to build and test changes to the documentation:

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

**Note:** Documentation is automatically deployed via GitHub Actions ([.github/workflows/docs.yml](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/.github/workflows/docs.yml)) upon pull request merges into master.

## Resources

### Meeting Notes

*   **May 2024 onwards:** [OpenCue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home)
*   **Prior to May 2024:** [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings)

### Contributors

See the [OpenCue Contributors](https://github.com/AcademySoftwareFoundation/OpenCue/graphs/contributors) page.
<img src="https://contrib.rocks/image?repo=AcademySoftwareFoundation/OpenCue" alt="Contributors image" />

### Contact & Community

*   **Slack:** Join the [OpenCue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q) for collaboration.
*   **Working Group:** Meets bi-weekly at 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).
*   **Mailing List:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io> for user and admin discussions.