[![OpenCue](/images/opencue_logo_with_text.png)](https://github.com/AcademySoftwareFoundation/OpenCue)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

# OpenCue: Revolutionizing VFX & Animation Rendering

OpenCue is a robust, open-source render management system designed to streamline and optimize your visual effects and animation workflows. Visit the [OpenCue GitHub Repository](https://github.com/AcademySoftwareFoundation/OpenCue) for the source code and more information.

## Key Features

*   **Scalable Architecture:**  Supports numerous concurrent machines for handling massive workloads.
*   **Production-Proven:**  Used on hundreds of films by Sony Pictures Imageworks.
*   **Flexible Resource Allocation:** Tagging systems for assigning jobs to specific hardware.
*   **Centralized Rendering:** Jobs processed on a central render farm, freeing up artist workstations.
*   **Native Multi-Threading:** Optimized for Katana, Prman, and Arnold.
*   **Deployment Flexibility:**  Supports multi-facility, on-premises, cloud, and hybrid environments.
*   **Advanced Resource Control:**  Split hosts into procs with dedicated cores and memory.
*   **Integrated Booking:** Automated resource booking.
*   **Unlimited Job Complexity:** No restrictions on the number of procs a job can utilize.

## Quick Start & Installation

### Sandbox Environment

Get up and running quickly with a local OpenCue environment using Docker containers or Python virtual environments. Perfect for testing and development. See the [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) for instructions.

### Full Installation

Detailed guides for system administrators covering OpenCue component deployment and dependency installation are available in the [OpenCue Documentation - Getting Started](https://www.opencue.io/docs/getting-started).

## Documentation and Resources

*   **Comprehensive Documentation:**  Explore installation guides, user guides, API references, and tutorials at [www.opencue.io](https://www.opencue.io).
*   **Video Tutorials:**  Learn more on the [OpenCue YouTube playlist](https://www.youtube.com/playlist?list=PL9dZxafYCWmzSBEwVT2AQinmZolYqBzdp) by the Academy Software Foundation (ASWF).

### Contributing to Documentation

When contributing to OpenCue, ensure that new features or changes are documented appropriately. Always update the documentation with your pull requests.

#### Building and Testing Documentation
Follow these steps to build and test documentation changes:

1.  **Build and validate the documentation:**

    ```bash
    ./docs/build.sh
    ```
2.  **Install bundler binstubs (if needed)**
    If you encounter permission errors when installing to system directories:
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

**Note:** Documentation updates are automatically deployed via GitHub Actions after pull requests are merged ([.github/workflows/docs.yml](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/.github/workflows/docs.yml)). The updated documentation will be available at https://docs.opencue.io/.

## Meeting Notes

*   **Recent Meetings:** Find meeting notes from May 2024 onwards on the [OpenCue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home).
*   **Older Meetings:** Access notes from before May 2024 in the [OpenCue repository](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings).

## Contributors

<a href="https://github.com/AcademySoftwareFoundation/OpenCue/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AcademySoftwareFoundation/OpenCue" alt="Contributors image" />
</a>

## Contact & Community

*   **Slack Channel:** Join the [OpenCue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q) for collaboration.
*   **Working Group:**  Bi-weekly meetings at 2 PM PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).
*   **Mailing List:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email the group directly at <opencue-user@lists.aswf.io> for discussions.