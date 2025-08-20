<!-- Improved README for OpenCue - Academy Software Foundation -->
![OpenCue](/images/opencue_logo_with_text.png)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

## OpenCue: The Open-Source Render Management System for VFX and Animation

**OpenCue**, developed by the Academy Software Foundation, empowers visual effects and animation studios to efficiently manage and scale their rendering pipelines. This open-source system provides robust tools for breaking down complex jobs and allocating computational resources effectively.  Explore the full capabilities on the [OpenCue GitHub repository](https://github.com/AcademySoftwareFoundation/OpenCue).

### Key Features:

*   **Scalable Architecture:** Designed to handle numerous concurrent machines, ensuring efficient processing of demanding workloads.
*   **Advanced Job Management:** Break down complex jobs into individual tasks and submit them to a configurable dispatch queue.
*   **Flexible Resource Allocation:** Utilize tagging systems to assign specific jobs to specialized machine types.
*   **Centralized Processing:** Jobs are processed on a central render farm, freeing up artist workstations.
*   **Native Multi-Threading:** Supports popular rendering engines like Katana, Prman, and Arnold.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid deployment models.
*   **Fine-Grained Resource Control:** Split hosts into multiple "procs" for dedicated core and memory allocation.
*   **Integrated Automation:** Features integrated automated booking for streamlined workflows.
*   **Unlimited Scalability:** No limit on the number of "procs" a job can utilize.

### Getting Started

Learn more about OpenCue and how to install, use, and administer it by visiting the official [OpenCue website](https://www.opencue.io).

*   **Sandbox Environment:** Set up a local OpenCue environment for testing and development using the [sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md). This is ideal for those new to OpenCue, providing a simple setup for experimentation. Find quick start guides on https://www.opencue.io/docs/quick-starts/.
*   **Full Installation:** Comprehensive guides for system administrators deploying OpenCue are available in the [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

### Documentation

The official [OpenCue documentation](https://docs.opencue.io) is built with Jekyll and hosted on GitHub Pages.  It includes:

*   Installation guides
*   User guides
*   API references
*   Tutorials

**Contributing to the Documentation:** When contributing to OpenCue, ensure documentation is updated to reflect new features or changes.  Build and test your documentation before submitting pull requests using the following steps:

1.  **Build and validate the documentation:**  `./docs/build.sh`
2.  **Install bundler binstubs (if needed):** `cd docs/; bundle binstubs --all`
3.  **Run the documentation locally:** `cd docs/; bundle exec jekyll serve --livereload`
4.  **Preview the documentation:** Open http://localhost:4000 in your browser.

Detailed documentation setup instructions and contribution guidelines can be found in [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md).

### Meeting Notes

*   Meeting notes starting May 2024 are available on the [Opencue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home).
*   Meeting notes before May 2024 can be found in the [Opencue repository](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings).

### Contact & Community

*   Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io> to join the OpenCue discussion forum.
*   Join the [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   Working Group meets biweekly at 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).