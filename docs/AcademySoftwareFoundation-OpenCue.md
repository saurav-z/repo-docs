# OpenCue: The Open-Source Render Management System for VFX and Animation

**Streamline your rendering workflow and scale your visual effects pipeline with OpenCue, a powerful and flexible open-source render management system.** [Learn more about OpenCue on GitHub](https://github.com/AcademySoftwareFoundation/OpenCue).

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

## Key Features of OpenCue

*   **Production-Proven:** Based on the render manager used by Sony Pictures Imageworks on hundreds of films.
*   **Highly Scalable:** Designed for managing rendering jobs across numerous concurrent machines.
*   **Flexible Resource Allocation:** Utilize tagging systems to assign specific jobs to specific machine types.
*   **Centralized Rendering:** Jobs processed on a central render farm, freeing up artist workstations.
*   **Native Multi-Threading:** Supports leading rendering engines like Katana, Prman, and Arnold.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid deployments.
*   **Granular Control:** Split hosts into numerous [procs](https://www.opencue.io/docs/concepts/glossary/#proc) with reserved cores and memory.
*   **Integrated Automation:** Includes automated booking capabilities.
*   **Unlimited Job Capacity:** No restrictions on the number of [procs](https://www.opencue.io/docs/concepts/glossary/#proc) a job can utilize.

## Getting Started with OpenCue

### Quick Installation & Testing

Easily set up a local OpenCue environment using the [OpenCue sandbox](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md). This is ideal for testing, development, and learning.

*   The sandbox uses Docker containers or Python virtual environments for a simplified setup.
*   It's perfect for small-scale tests and experimentation.

Find detailed instructions on running the sandbox at https://www.opencue.io/docs/quick-starts/.

### Full Installation

System administrators can find comprehensive guides for deploying OpenCue components and installing dependencies in the [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

## Documentation

Comprehensive OpenCue documentation is available at [https://docs.opencue.io/](https://docs.opencue.io/), built using Jekyll and hosted on GitHub Pages. It includes:

*   Installation guides
*   User guides
*   API references
*   Tutorials

### Contributing to Documentation

When contributing to OpenCue, ensure your changes are reflected in the documentation.  Build and test your changes before submitting a pull request:

1.  **Build and validate:** `./docs/build.sh`
2.  **Install bundler binstubs (if needed):** `cd docs/; bundle binstubs --all`
3.  **Run locally:** `cd docs/; bundle exec jekyll serve --livereload`
4.  **Preview:** Open http://localhost:4000 in your browser.

For detailed documentation setup, testing, and contribution guidelines, refer to [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md).

## Meeting Notes

*   **May 2024 onwards:**  OpenCue meeting notes are available on the [OpenCue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home).
*   **Before May 2024:** Access meeting notes in the [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) directory.

## Contact Us

*   **User & Admin Forum:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io>.
*   **Slack:** Join the [OpenCue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   **Working Group:** Bi-weekly meetings at 2 PM PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).