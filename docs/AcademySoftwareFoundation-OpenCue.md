![OpenCue Logo](images/opencue_logo_with_text.png)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

# OpenCue: Your Open-Source Solution for Robust Render Management

OpenCue is a powerful, open-source render management system designed to streamline and accelerate visual effects (VFX) and animation pipelines.  [Learn more on the original repository](https://github.com/AcademySoftwareFoundation/OpenCue).

## Key Features

*   **Production-Proven:** Leverages the same technology as Sony Pictures Imageworks, used on hundreds of films.
*   **Scalable Architecture:** Supports high-volume, concurrent machine setups.
*   **Flexible Job Management:** Utilize tagging systems for optimized job allocation.
*   **Centralized Rendering:** Jobs are processed on a central render farm, freeing up artist workstations.
*   **Multi-Threading Support:** Native support for Katana, Prman, and Arnold rendering engines.
*   **Deployment Flexibility:** Works across multi-facility, on-premises, cloud, and hybrid environments.
*   **Resource Optimization:** Split a host into multiple "procs" for tailored core and memory allocation.
*   **Automated Booking & No Limits:** Integrated automated booking and no restrictions on job proc counts.

## Getting Started

### Quick Installation with Sandbox

Experiment with OpenCue locally using the sandbox environment.  The sandbox allows you to run a test OpenCue deployment in Docker containers or Python virtual environments, ideal for testing and learning.  Detailed setup instructions can be found in the [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) and at [www.opencue.io/docs/quick-starts/](https://www.opencue.io/docs/quick-starts/).

### Full Installation

System administrators can find comprehensive guides for deploying OpenCue components and installing dependencies within the [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

## Documentation

Comprehensive documentation, including installation guides, user guides, API references, and tutorials, is available to help you get started with OpenCue.  The documentation is built with Jekyll and hosted on GitHub Pages.

### Documentation Updates

When contributing to OpenCue, please update the documentation to reflect any new features or changes.  Each pull request should include the relevant documentation updates where applicable.

### Building & Testing Documentation

1.  **Build and validate:** `./docs/build.sh`
2.  **Install bundler binstubs (if needed):**
    ```bash
    cd docs/
    bundle binstubs --all
    ```
3.  **Run locally:** `cd docs/ && bundle exec jekyll serve --livereload`
4.  **Preview:** Open http://localhost:4000 in your browser.

For further guidance on documentation setup, testing, and contribution guidelines, refer to [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md).

Once a pull request is merged, the documentation will be automatically updated via GitHub Actions and available at https://docs.opencue.io/.

## Meeting Notes

*   **May 2024 onwards:**  Consult the [Opencue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home).
*   **Prior to May 2024:** Browse the [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) folder in the repository.

## Contact & Community

*   **User & Admin Forum:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io>.
*   **Slack:** Connect via the [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   **Bi-weekly Working Group:** Meets at 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).