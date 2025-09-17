# OpenCue: The Open-Source Render Management System for VFX and Animation

[![OpenCue Logo](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/images/opencue_logo_with_text.png?raw=true)](https://github.com/AcademySoftwareFoundation/OpenCue)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
[![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

**OpenCue** is an open-source render management system designed to streamline complex rendering workflows in visual effects and animation.

[View the original repository on GitHub](https://github.com/AcademySoftwareFoundation/OpenCue)

## Key Features

OpenCue offers a robust set of features to manage your rendering jobs efficiently:

*   **Industry Proven:** Built on the foundation of Sony Pictures Imageworks' in-house render manager, used on hundreds of films.
*   **Scalable Architecture:** Supports numerous concurrent machines, enabling efficient rendering at scale.
*   **Flexible Resource Allocation:** Tagging systems allow allocation of specific jobs to specific machine types.
*   **Centralized Rendering:** Jobs are processed on a central render farm, freeing up artist workstations.
*   **Native Multi-Threading:** Supports Katana, Prman, and Arnold, enhancing rendering speed.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid deployments.
*   **Granular Resource Control:**  Split a host into numerous [procs](https://www.opencue.io/docs/concepts/glossary/#proc) with reserved cores and memory.
*   **Automated Booking:** Integrated automated booking to streamline your workflow.
*   **Unlimited Job Size:** No limit on the number of [procs](https://www.opencue.io/docs/concepts/glossary/#proc) a job can have.

## Getting Started

### Quick Installation & Testing

The [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) provides an easy way to run a local OpenCue environment using Docker containers or Python virtual environments.  This is ideal for testing, development, and learning.

### Full Installation

System administrators can find comprehensive guides for deploying OpenCue components and installing dependencies within the [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

## Documentation

Comprehensive documentation is available to guide you through the use and administration of OpenCue:

*   **[OpenCue Documentation](https://www.opencue.io)**: Includes installation guides, user guides, API references, and tutorials.
*   **Contribution Guidelines:** When contributing, update the documentation for any new features or changes. Each pull request should include relevant documentation updates when applicable.
*   **Documentation Build and Testing:** Instructions for building and testing documentation can be found in this README, but more detailed documentation setup instructions, testing procedures, and contribution guidelines, see [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md).

## Meeting Notes

*   **Recent Notes:**  Available on the [OpenCue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home) (May 2024 onwards).
*   **Older Notes:**  Found in the [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) folder of the OpenCue repository (before May 2024).

## Contributors

[<img src="https://contrib.rocks/image?repo=AcademySoftwareFoundation/OpenCue" alt="Contributors image" />](https://github.com/AcademySoftwareFoundation/OpenCue/graphs/contributors)

## Contact

*   **Mailing List:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io> to connect with users and admins.
*   **Slack:** Join the [OpenCue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q) for real-time discussions.
*   **Working Group:**  The OpenCue Working Group meets bi-weekly at 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).