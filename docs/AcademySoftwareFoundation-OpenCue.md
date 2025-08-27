![OpenCue](/images/opencue_logo_with_text.png)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

# OpenCue: The Open-Source Render Management System for VFX and Animation

OpenCue empowers visual effects and animation studios to efficiently manage and scale their rendering pipelines. Learn more and contribute on [GitHub](https://github.com/AcademySoftwareFoundation/OpenCue).

## Key Features

*   **Scalable Architecture:** Designed to handle massive workloads with numerous concurrent machines.
*   **Production-Proven:** Built upon the render manager used by Sony Pictures Imageworks on hundreds of films.
*   **Flexible Resource Allocation:** Utilize tagging systems to direct jobs to specific machine types.
*   **Centralized Rendering:** Jobs are processed on a central render farm, freeing up artist workstations.
*   **Native Multi-Threading:** Optimized for Katana, Prman, and Arnold.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid environments.
*   **Granular Resource Control:** Split hosts into multiple "procs" with dedicated core and memory requirements.
*   **Integrated Automation:** Features automated booking.
*   **Unlimited Scalability:** No limitations on the number of "procs" a job can have.

## Getting Started

### Sandbox Environment

Quickly get started with OpenCue using the sandbox environment. This setup allows you to run a test OpenCue deployment locally using Docker containers or Python virtual environments. It is great for experimentation and learning.

For detailed instructions, see: [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) and the [quick start guide](https://www.opencue.io/docs/quick-starts/).

### Full Installation

System administrators can find comprehensive guides for deploying OpenCue components and managing dependencies within the official [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

## Documentation

Comprehensive OpenCue documentation is available, including installation guides, user guides, API references, and tutorials.  Contribute to the documentation to keep it up to date.

### Building and Testing Documentation

Follow these steps to build and test documentation locally before submitting a pull request:

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

4.  **Preview the documentation:**

    Open http://localhost:4000 in your browser to review your changes.

For detailed documentation setup instructions, testing procedures, and contribution guidelines, see [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md).

**Note:** Documentation is automatically deployed via GitHub Actions after pull requests are merged.  Updated documentation will be accessible at https://docs.opencue.io/.

## Meeting Notes

*   **May 2024 onwards:**  OpenCue meeting notes are stored on the [Opencue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home).
*   **Prior to May 2024:**  Refer to the Opencue repository in the [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) folder.

## Community and Support

*   **Discussion Forum:**  Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) to connect with users and admins.
*   **Slack Channel:**  Join the [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   **Bi-Weekly Meetings:** The Working Group meets biweekly at 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).