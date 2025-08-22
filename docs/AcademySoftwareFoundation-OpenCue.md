[![OpenCue](/images/opencue_logo_with_text.png)](https://github.com/AcademySoftwareFoundation/OpenCue)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

# OpenCue: Your Open-Source Solution for VFX and Animation Render Management

## About OpenCue

OpenCue is an open-source render management system designed to streamline the complex rendering workflows of visual effects (VFX) and animation production.  Developed by the Academy Software Foundation (ASWF), OpenCue allows you to break down large jobs into manageable tasks, efficiently utilizing computational resources for faster and more reliable rendering.  

[Visit the original OpenCue repository on GitHub](https://github.com/AcademySoftwareFoundation/OpenCue) to learn more.

## Key Features

*   **Industry-Proven**: Based on the in-house render manager used by Sony Pictures Imageworks, powering hundreds of films.
*   **Scalable Architecture**: Supports massive render farms with numerous concurrent machines.
*   **Flexible Resource Allocation**: Utilize tagging systems to assign jobs to specific machine types.
*   **Centralized Processing**:  Jobs are rendered on a central farm, freeing up artist workstations.
*   **Native Multi-Threading**: Supports Katana, Prman, and Arnold for optimal performance.
*   **Deployment Flexibility**:  Works seamlessly across multi-facility, on-premises, cloud, and hybrid environments.
*   **Advanced Resource Control**: Split hosts into numerous procs, each with custom core and memory allocations.
*   **Automated Booking**: Integrated for efficient resource management.
*   **Unlimited Job Complexity**:  No limitations on the number of procs a job can utilize.

## Getting Started

### Quick Installation with Sandbox

Explore OpenCue with the easy-to-use sandbox environment:

*   Run a test OpenCue deployment locally using Docker containers or Python virtual environments.
*   Ideal for testing, development, and learning.
*   Learn how to run the sandbox: [OpenCue Sandbox Documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) 

### Full Installation

For system administrators, comprehensive guides for deploying OpenCue components are available in the OpenCue documentation:  [OpenCue Documentation](https://www.opencue.io/docs/getting-started/)

## Documentation

Comprehensive documentation is available at [https://docs.opencue.io/](https://docs.opencue.io/)

The documentation includes installation guides, user guides, API references, and tutorials to help you get started with OpenCue.

### Contributing to the Documentation

When contributing to OpenCue, please update the documentation for any new features or changes. Each pull request should include relevant documentation updates when applicable.

#### Building and Testing Documentation

If you make changes to `OpenCue/docs`, please build and test the documentation before submitting your PR:

1.  **Build and validate the documentation**
    ```bash
    ./docs/build.sh
    ```

2.  **Install bundler binstubs (if needed)**

    If you encounter permission errors when installing to system directories:
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

    Open http://localhost:4000 in your browser to review your changes.

For detailed documentation setup instructions, testing procedures, and contribution guidelines, see [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md).

**Note:** Once your pull request is merged into master, the documentation will be automatically deployed via GitHub Actions ([.github/workflows/docs.yml](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/.github/workflows/docs.yml)). The updated documentation will be available at https://docs.opencue.io/.

## Meeting Notes

*   **Latest:** Meeting notes from May 2024 onwards are stored on the [Opencue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home).
*   **Older:**  Meeting notes before May 2024 can be found in the [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) folder of the repository.

## Get in Touch

*   **Discussion Forum**: Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io>.
*   **Slack**: Join the [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   **Working Group**: Meets bi-weekly at 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).