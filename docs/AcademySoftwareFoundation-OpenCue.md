![OpenCue](/images/opencue_logo_with_text.png)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

# OpenCue: Your Open-Source Solution for Scalable Render Management

**Tired of render farm bottlenecks? OpenCue, the open-source render management system, empowers visual effects and animation studios to efficiently manage complex rendering workflows.**

[View the original repository on GitHub](https://github.com/AcademySoftwareFoundation/OpenCue)

## Key Features of OpenCue

*   **Proven in Production:**  Built upon the technology of Sony Pictures Imageworks' in-house render manager, used on hundreds of films.
*   **Highly Scalable Architecture:** Supports numerous concurrent machines, ideal for demanding workloads.
*   **Resource Allocation:** Tagging systems allow for flexible job allocation to specific machine types.
*   **Centralized Processing:** Jobs are processed on a central render farm, freeing up artist workstations.
*   **Native Multi-threading:** Supports industry-standard renderers like Katana, Prman, and Arnold.
*   **Deployment Flexibility:**  Supports multi-facility, on-premises, cloud, and hybrid deployments.
*   **Granular Resource Control:** Split hosts into procs for detailed core and memory management.
*   **Automated Booking:** Integrated automated booking simplifies job scheduling.
*   **Unrestricted Job Size:** No limits on the number of procs a job can utilize.

## Getting Started

### Quick Start with the Sandbox

The [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) provides a simple way to run a local OpenCue environment, making it perfect for testing, development, and learning. This environment uses Docker containers or Python virtual environments for a streamlined setup.

### Full Installation

System administrators can find detailed guides for deploying OpenCue components and installing dependencies in the [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

## Documentation and Community

Comprehensive documentation is available at [https://docs.opencue.io/](https://docs.opencue.io/), offering installation guides, user guides, API references, and tutorials. Contributions to the documentation are welcome and should include updates for new features or changes.

### Building and Testing Documentation

If you make changes to the documentation (`OpenCue/docs`), follow these steps to build and test your changes before submitting a pull request:

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

For detailed documentation setup instructions, testing procedures, and contribution guidelines, see [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md).

## Meeting Notes

*   **May 2024 Onward:** Find meeting notes on the [Opencue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home).
*   **Before May 2024:** Refer to the Opencue repository in the [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) folder.

## Contact Us

*   **Discussion Forum:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io>.
*   **Slack:** Join the [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   **Working Group:** Meets bi-weekly at 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).