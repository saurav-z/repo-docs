[![OpenCue](/images/opencue_logo_with_text.png)](https://github.com/AcademySoftwareFoundation/OpenCue)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

# OpenCue: The Open-Source Render Management System for VFX and Animation

OpenCue is an open-source render management system, empowering visual effects and animation studios to efficiently manage complex rendering pipelines. ([See the original repository](https://github.com/AcademySoftwareFoundation/OpenCue)).

## Key Features

*   **Scalable Architecture:** Supports numerous concurrent machines for handling large-scale rendering jobs.
*   **Production-Proven:** Based on the Sony Pictures Imageworks in-house render manager, used on hundreds of films.
*   **Flexible Resource Allocation:** Tagging systems enable job allocation to specific machine types.
*   **Centralized Rendering:** Jobs are processed on a central render farm, freeing up artist workstations.
*   **Multi-Threading Support:** Native multi-threading for Katana, Prman, and Arnold.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid deployments.
*   **Efficient Resource Management:** Split hosts into procs, each with reserved core and memory.
*   **Integrated Booking:** Features automated booking for streamlined workflow.
*   **Unrestricted Job Size:** No limit on the number of procs a job can utilize.

## Get Started

### Quick Installation and Testing

Experiment with OpenCue locally using the sandbox environment:

*   Set up a local OpenCue environment easily with all components running in separate Docker containers or Python virtual environments.
*   Ideal for small tests, development work, and learning.
*   Learn how to run the sandbox environment in the [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) and [quick-starts](https://www.opencue.io/docs/quick-starts/).

### Full Installation

System admins can find guides for deploying OpenCue components and installing dependencies in the [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

## Documentation and Support

Comprehensive documentation is available to guide you through installation, usage, and administration.

*   **OpenCue Documentation:** [www.opencue.io](https://www.opencue.io) provides installation guides, user guides, API references, and tutorials.
*   **Contribute:** Update the documentation for any new features or changes.
*   **Documentation Build and Test:**  Build and test the documentation locally:  `./docs/build.sh`. Run the documentation locally:  `bundle exec jekyll serve --livereload`. Preview documentation at `http://localhost:4000`. For detailed documentation setup instructions, testing procedures, and contribution guidelines, see [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md).
*   **Documentation Deployment:**  Documentation is automatically deployed via GitHub Actions ([.github/workflows/docs.yml](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/.github/workflows/docs.yml)) and available at https://docs.opencue.io/.

## Community and Meetings

*   **Meeting Notes:** Access meeting notes on the [Opencue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home) (May 2024 onwards) and in the [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) folder (for notes before May 2024).
*   **Discussion Forum:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io>.
*   **Slack Channel:** Join the [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   **Working Group Meetings:**  Meets biweekly at 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).