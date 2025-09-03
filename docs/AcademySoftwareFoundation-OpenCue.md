# OpenCue: The Open-Source Render Management System for VFX and Animation

[![OpenCue Logo](/images/opencue_logo_with_text.png)](https://github.com/AcademySoftwareFoundation/OpenCue)

**OpenCue empowers visual effects and animation studios to efficiently manage and scale rendering workflows, offering a robust and flexible solution for complex production pipelines.**

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

## Key Features of OpenCue:

*   **Scalable Architecture:** Designed to handle massive rendering demands with numerous concurrent machines.
*   **Production-Proven:**  Based on the render manager used by Sony Pictures Imageworks, powering hundreds of films.
*   **Flexible Resource Allocation:** Utilize tagging systems to assign jobs to specific machine types for optimal performance.
*   **Centralized Rendering:** Offloads rendering tasks from artist workstations to a dedicated render farm.
*   **Multi-Threading Support:** Native support for leading rendering engines like Katana, Prman, and Arnold.
*   **Deployment Flexibility:** Compatible with multi-facility, on-premises, cloud, and hybrid deployment models.
*   **Advanced Resource Control:** Split hosts into multiple "procs" with defined core and memory requirements.
*   **Integrated Automation:** Features integrated automated booking to streamline workflow.
*   **Unlimited Job Size:** No practical limit on the number of procs a job can utilize.

## Getting Started

### Quick Installation with Sandbox

Get up and running quickly with a local OpenCue environment using the sandbox:

*   **Sandbox Environment:**  Run OpenCue locally using Docker containers or Python virtual environments for easy testing and development.
*   **Ideal for:** Small tests, development work, and learning the basics.

Read the [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) and the [OpenCue documentation](https://www.opencue.io/docs/quick-starts/) for setup instructions.

### Full Installation

For system administrators, comprehensive guides for deploying OpenCue components and installing dependencies are available in the [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

## Documentation

OpenCue documentation is comprehensive and well-maintained to help you get started:

*   **Hosted Documentation:**  Built with Jekyll and hosted on GitHub Pages at [https://docs.opencue.io/](https://docs.opencue.io/).
*   **Content:** Includes installation guides, user guides, API references, and tutorials.
*   **Contribution:** Please update the documentation for any new features or changes in your pull requests.
*   **Building and Testing:** See instructions in the original README for building and testing the documentation.

## Community & Support

*   **Discussion Forum:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email the group at <opencue-user@lists.aswf.io>.
*   **Slack:** Connect with the community on the [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   **Meetings:** The OpenCue Working Group meets biweekly at 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).
*   **Meeting Notes:** Review the [Opencue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home) for notes from May 2024 onwards.  For earlier notes, see the [OpenCue repository](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings).

**Explore the source code and contribute on GitHub: [https://github.com/AcademySoftwareFoundation/OpenCue](https://github.com/AcademySoftwareFoundation/OpenCue)**