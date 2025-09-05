![OpenCue](/images/opencue_logo_with_text.png)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

# OpenCue: Your Open-Source Solution for VFX and Animation Rendering Management

**OpenCue** is an open-source render management system designed to streamline and optimize visual effects and animation production pipelines. ([Original Repository](https://github.com/AcademySoftwareFoundation/OpenCue))

## Key Features

*   **Production-Proven:** Originally developed and used at Sony Pictures Imageworks on hundreds of films.
*   **Highly Scalable:** Supports numerous concurrent machines for efficient rendering.
*   **Flexible Resource Allocation:** Utilize tagging systems to assign jobs to specific machine types.
*   **Centralized Rendering:** Jobs are processed on a central render farm, freeing up artist workstations.
*   **Native Multi-Threading:** Supports Katana, Prman, and Arnold for optimal performance.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid environments.
*   **Fine-Grained Control:** Split hosts into numerous procs, each with reserved resources (cores and memory).
*   **Integrated Automation:** Includes automated booking features.
*   **Unlimited Scalability:** No limitations on the number of procs per job.

## Get Started

### Quick Installation with Sandbox

Easily test and experiment with OpenCue using the sandbox environment. All components run in separate Docker containers or Python virtual environments. Perfect for small tests, development, and learning.

*   **Learn how to run the sandbox:** [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md)
*   **For more info:** [OpenCue sandbox documentation](https://www.opencue.io/docs/quick-starts/)

### Full Installation

System administrators can find comprehensive guides for deploying OpenCue components and installing dependencies in the main OpenCue documentation.

## Documentation

Comprehensive documentation is available to help you get started with OpenCue, including installation guides, user guides, API references, and tutorials.

*   **Official Documentation:** [www.opencue.io](https://www.opencue.io)
*   **OpenCue Documentation:** [https://docs.opencue.io/](https://docs.opencue.io/)
*   **For detailed documentation setup instructions, testing procedures, and contribution guidelines:** [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md)

### Contributing to Documentation

Update the documentation with every new feature or change. Ensure all pull requests include documentation updates where applicable.

### Building and Testing Documentation

Follow these steps to build and test the documentation:

1.  **Build and validate the documentation**
    ```bash
    ./docs/build.sh
    ```
2.  **Install bundler binstubs (if needed)**
    ```bash
    cd docs/
    bundle binstubs --all
    ```
3.  **Run the documentation locally**
    ```bash
    cd docs/
    bundle exec jekyll serve --livereload
    ```
4.  **Preview the documentation**
    Open http://localhost:4000 in your browser.

## Resources

*   **YouTube:** Watch videos on the [OpenCue Playlist](https://www.youtube.com/playlist?list=PL9dZxafYCWmzSBEwVT2AQinmZolYqBzdp)
*   **Meeting notes:**
    *   Starting from May 2024, all Opencue meeting notes are stored on the [Opencue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home).
    *   For meeting notes before May 2024, please refer to the Opencue repository in the [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) folder.

## Contact and Support

*   **Mailing List:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io> for user and admin discussions.
*   **Slack:** Join the [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   **Working Group:** Meets biweekly at 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).