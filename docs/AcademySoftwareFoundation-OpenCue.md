# OpenCue: The Open-Source Render Management System for VFX and Animation

**Tired of rendering bottlenecks?** OpenCue is a powerful, open-source render management system designed to streamline your VFX and animation pipelines. ([Original Repo](https://github.com/AcademySoftwareFoundation/OpenCue))

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

## Key Features of OpenCue

*   **Production-Proven:** Based on Sony Pictures Imageworks' in-house render manager, used on hundreds of films.
*   **Scalable Architecture:** Supports a vast number of concurrent machines for handling demanding workloads.
*   **Flexible Resource Allocation:** Tagging systems allow you to allocate specific jobs to specific machine types.
*   **Centralized Rendering:** Jobs are processed on a central render farm, freeing up artist workstations.
*   **Multi-Threading Support:** Native multi-threading compatible with Katana, Prman, and Arnold.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid deployments.
*   **Fine-Grained Control:** Split hosts into procs with reserved cores and memory.
*   **Integrated Booking:** Includes automated booking functionality.
*   **Unlimited Job Size:** No limits on the number of procs a job can utilize.

## Get Started with OpenCue

### Quick Installation and Testing

The [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) provides instructions on how to set up a local OpenCue environment using Docker containers or Python virtual environments. This is ideal for testing, development, and learning.

### Full Installation

System admins can find detailed guides for deploying OpenCue components and installing dependencies in the [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

## OpenCue Documentation

Comprehensive documentation, built with Jekyll and hosted on GitHub Pages, is available at [https://docs.opencue.io/](https://docs.opencue.io/). This includes installation guides, user guides, API references, and tutorials.

**Contributing to Documentation:**

When contributing to OpenCue, update the documentation for any new features or changes. Ensure that your pull requests include relevant documentation updates.

**Building and Testing Documentation:**

1.  **Build and validate:** `./docs/build.sh`
2.  **Install bundler binstubs (if needed):**
    ```bash
    cd docs/
    bundle binstubs --all
    ```
3.  **Run locally:** `cd docs/ && bundle exec jekyll serve --livereload`
4.  **Preview:** Open `http://localhost:4000` in your browser.

For detailed instructions, see [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md). Documentation is automatically deployed via GitHub Actions after merges.

## Community and Support

*   **Meeting Notes:** All Opencue meeting notes are stored on the [Opencue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home).
*   **Discussion Forum:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) to connect with users and admins.
*   **Slack:** Join the [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   **Working Group:** Meets biweekly at 2 PM PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).