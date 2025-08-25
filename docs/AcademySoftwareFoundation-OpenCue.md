# OpenCue: The Open-Source Render Management System for VFX and Animation

**OpenCue** is a powerful, open-source render management system designed to streamline and accelerate visual effects (VFX) and animation production.  [Learn more on the OpenCue GitHub Repo](https://github.com/AcademySoftwareFoundation/OpenCue).

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

## Key Features of OpenCue:

*   **Scalable Architecture:** Supports numerous concurrent machines for efficient rendering.
*   **Production-Proven:**  Used in-house at Sony Pictures Imageworks for hundreds of films.
*   **Flexible Resource Allocation:**  Tagging systems to direct jobs to specific machine types.
*   **Centralized Processing:**  Jobs run on a central render farm, freeing up artist workstations.
*   **Native Multi-Threading:**  Supports Katana, Prman, and Arnold for optimized performance.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid deployments.
*   **Granular Resource Control:**  Split hosts into multiple procs with reserved cores and memory.
*   **Integrated Automated Booking:** Streamlines job submission and scheduling.
*   **Unlimited Job Size:** No limit on the number of procs a job can utilize.

## Getting Started with OpenCue

### Quick Installation & Testing

Quickly get up and running with OpenCue using the sandbox environment.  This is ideal for experimenting and learning.

*   Explore the [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) to set up a local testing environment.
*   The sandbox uses Docker containers or Python virtual environments.
*   See [OpenCue documentation](https://www.opencue.io/docs/quick-starts/) for running the sandbox.

### Full Installation

Detailed guides for system administrators on deploying OpenCue components and installing dependencies are available in the [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

## OpenCue Documentation

Comprehensive documentation is available to help you install, use, and contribute to OpenCue:

*   **Hosted:** The documentation is built with Jekyll and hosted on GitHub Pages.
*   **Content:** Includes installation guides, user guides, API references, and tutorials.
*   **Contribution:** When making changes to OpenCue, update the documentation and follow the build and testing steps below.

### Building and Testing Documentation

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
    Open [http://localhost:4000](http://localhost:4000) in your browser.

For detailed documentation setup instructions, testing procedures, and contribution guidelines, see [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md).

**Note:** Documentation is automatically deployed via GitHub Actions ([.github/workflows/docs.yml](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/.github/workflows/docs.yml)) after pull requests are merged.

The OpenCue documentation is now available at https://docs.opencue.io/.

## Stay Connected

### Meeting Notes
*   Starting from May 2024, all Opencue meeting notes are stored on the [Opencue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home).
*   For meeting notes before May 2024, please refer to the Opencue repository in the [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) folder.

### Community Resources

*   Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io> for user and admin discussions.
*   Join the [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   Working Group meets biweekly at 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).