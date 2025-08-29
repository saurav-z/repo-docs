# OpenCue: Your Open-Source Render Management Solution for VFX and Animation

**[OpenCue](https://github.com/AcademySoftwareFoundation/OpenCue) empowers visual effects and animation studios to efficiently manage and scale their rendering pipelines.**

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

## Key Features of OpenCue

*   **Scalable Architecture:** Designed to handle demanding workloads, supporting numerous concurrent machines.
*   **Job Management:** Break down complex jobs into individual tasks and submit them to a configurable dispatch queue.
*   **Resource Allocation:** Utilize tagging systems to allocate specific jobs to specific machine types.
*   **Centralized Rendering:** Process jobs on a central render farm, freeing up artist workstations.
*   **Native Multi-Threading:** Supports industry-standard renderers like Katana, Prman, and Arnold.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid deployments.
*   **Advanced Resource Control:** Split hosts into numerous [procs](https://www.opencue.io/docs/concepts/glossary/#proc), allowing for granular control over core and memory allocation.
*   **Integrated Booking:** Includes automated booking capabilities for streamlined workflow.
*   **No Job Limits:** No restrictions on the number of [procs](https://www.opencue.io/docs/concepts/glossary/#proc) a job can utilize.
*   **Production-Proven:** Based on the Sony Pictures Imageworks in-house render manager used on hundreds of films.

## Get Started with OpenCue

### Quick Installation and Testing

The [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) provides a simple way to run a local OpenCue environment using Docker containers or Python virtual environments. This is ideal for testing, development, and learning. See [https://www.opencue.io/docs/quick-starts/](https://www.opencue.io/docs/quick-starts/) for details on running the sandbox.

### Full Installation

System administrators can find detailed guides for deploying OpenCue components and installing dependencies in the [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

## OpenCue Documentation

Comprehensive documentation is available to help you get started with OpenCue, including installation guides, user guides, API references, and tutorials.  The documentation is built with Jekyll and hosted on GitHub Pages.

### Building and Testing Documentation

If you are contributing to OpenCue, follow these steps to build, test, and preview your documentation changes:

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
4.  **Preview the documentation:** Open http://localhost:4000 in your browser.

For detailed documentation setup instructions, testing procedures, and contribution guidelines, see [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md).  The documentation is automatically deployed via GitHub Actions after a pull request is merged. The updated documentation will be available at https://docs.opencue.io/.

## Meeting Notes

OpenCue meeting notes are available on the [Opencue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home) (starting May 2024) or the [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) folder (for notes before May 2024).

## Connect with the OpenCue Community

*   **Discussion Forum:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io>.
*   **Slack:** Join the [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   **Working Group:** Meets biweekly at 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).