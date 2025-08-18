<div align="center">
  <img src="/images/opencue_logo_with_text.png" alt="OpenCue Logo" width="400"/>
</div>

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

# OpenCue: Revolutionizing VFX Rendering with Open Source Technology

OpenCue is an open-source render management system designed to streamline and optimize the rendering pipeline for visual effects (VFX) and animation production, offering scalability and control. ([View the original repository](https://github.com/AcademySoftwareFoundation/OpenCue))

## Key Features of OpenCue

*   **Industry-Proven:** Leverages the same technology as Sony Pictures Imageworks' in-house render manager, used in hundreds of films.
*   **Highly Scalable:**  Supports large-scale deployments with numerous concurrent machines.
*   **Flexible Resource Allocation:**  Utilizes tagging systems for job allocation to specific machine types.
*   **Centralized Rendering:** Jobs are processed on a central render farm, freeing up artist workstations.
*   **Multi-Threading Support:** Native multi-threading support for Katana, Prman, and Arnold.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid deployments.
*   **Granular Resource Control:** Allows splitting a host into numerous procs, each with reserved cores and memory.
*   **Automated Booking:** Integrated automated booking features.
*   **Unlimited Scalability:** No limits on the number of procs a job can utilize.

## Getting Started with OpenCue

### Quick Installation and Testing

Explore the [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) for setting up a local OpenCue environment using Docker containers or Python virtual environments. Ideal for testing, development, and learning.

### Full Installation

System administrators can find detailed installation guides for deploying OpenCue components and dependencies in the [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

## Documentation

Comprehensive OpenCue documentation is available at [https://www.opencue.io](https://www.opencue.io), including:

*   Installation guides
*   User guides
*   API references
*   Tutorials

### Building and Testing Documentation

To contribute to OpenCue documentation:

1.  **Build and validate the documentation:**  `./docs/build.sh`
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
4.  **Preview the documentation:** Open http://localhost:4000 in your browser.

For detailed documentation setup and contribution guidelines, see [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md).

## Meeting Notes

*   **May 2024 Onward:** [Opencue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home)
*   **Prior to May 2024:** [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) in the OpenCue repository.

## Contact Us

*   **Discussion Forum:** [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io>.
*   **Slack:** [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   **Working Group:** Bi-weekly meetings at 2 pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).