<div align="center">
  <a href="https://github.com/AcademySoftwareFoundation/OpenCue">
    <img src="/images/opencue_logo_with_text.png" alt="OpenCue Logo" width="400">
  </a>
  <br>
  [![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
  [![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/)
  [![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)
</div>

# OpenCue: The Open-Source Render Management Solution for VFX and Animation

**OpenCue is a powerful, open-source render management system designed to streamline and scale your visual effects and animation pipelines.**  

[Visit the OpenCue GitHub Repository](https://github.com/AcademySoftwareFoundation/OpenCue)

## Key Features

*   **Production-Proven:** Based on the in-house render manager used by Sony Pictures Imageworks for hundreds of films.
*   **Highly Scalable:** Supports numerous concurrent machines for handling demanding workloads.
*   **Flexible Resource Allocation:** Tagging systems allow you to target specific machine types for particular jobs.
*   **Centralized Rendering:** Jobs are processed on a central render farm, freeing up artist workstations.
*   **Multi-Threading Support:** Native support for Katana, Prman, and Arnold, enabling efficient rendering.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid deployment models.
*   **Granular Resource Control:** Split hosts into numerous [procs](https://www.opencue.io/docs/concepts/glossary/#proc) with dedicated cores and memory.
*   **Integrated Automation:** Automated booking for efficient resource management.
*   **No Limits:** Jobs can have an unlimited number of [procs](https://www.opencue.io/docs/concepts/glossary/#proc).

## Getting Started

### Quick Installation with Sandbox Environment

Easily set up a local OpenCue environment for testing and development using the sandbox.  Learn how:

*   [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md)
*   Ideal for small tests, development work, and learning.

### Full Installation

For system administrators, comprehensive guides are available for deploying OpenCue components:

*   [OpenCue documentation](https://www.opencue.io/docs/getting-started/)

## Documentation

Comprehensive OpenCue documentation is available to help you get started:

*   Installation guides
*   User guides
*   API references
*   Tutorials

### Contributing to Documentation

Please update the documentation for new features or changes, and include relevant documentation updates with your pull requests. 

To build and test the documentation:
1.  Build and validate the documentation: `./docs/build.sh`
2.  Run the documentation locally: `cd docs/ && bundle exec jekyll serve --livereload`
3.  Preview the documentation by opening http://localhost:4000 in your browser.

See [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md) for detailed documentation setup and contribution guidelines.

## Stay Connected

*   **Discussion Forum:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io> for user and admin discussions.
*   **Slack Channel:** Join the [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   **Working Group:** Bi-weekly meetings at 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).
*   **Meeting Notes:** Access meeting notes on the [Opencue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home) (starting May 2024) or in the [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) folder for notes before May 2024.