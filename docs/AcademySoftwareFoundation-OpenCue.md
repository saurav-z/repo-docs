![OpenCue](/images/opencue_logo_with_text.png)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

# OpenCue: Your Open-Source Solution for Render Management

**OpenCue is a powerful, open-source render management system designed to streamline visual effects and animation pipelines.**

[View the original repository on GitHub](https://github.com/AcademySoftwareFoundation/OpenCue)

## Key Features of OpenCue

*   **Industry-Proven:** Used by Sony Pictures Imageworks on hundreds of films.
*   **Scalable Architecture:**  Supports a vast number of concurrent machines.
*   **Flexible Resource Allocation:**  Utilizes tagging systems to direct jobs to specific machine types.
*   **Centralized Rendering:**  Jobs are processed on a central render farm, freeing up artist workstations.
*   **Multi-Threading Support:** Native support for Katana, Prman, and Arnold.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid environments.
*   **Efficient Resource Management:** Allows splitting hosts into numerous procs with dedicated cores and memory.
*   **Automated Booking:** Integrated automated booking simplifies job scheduling.
*   **Unlimited Job Size:** No restriction on the number of procs a job can have.

## Getting Started

### Quick Installation with Sandbox

The OpenCue sandbox environment offers an easy way to run a test OpenCue deployment locally using Docker containers or Python virtual environments. This is ideal for experimentation, small tests, and for newcomers to OpenCue.

Learn how to run the sandbox environment via the [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md).

### Full Installation

For system administrators deploying OpenCue components and installing dependencies, comprehensive guides are available in the [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

## Documentation

Comprehensive documentation, including installation guides, user guides, API references, and tutorials, is available at [https://docs.opencue.io/](https://docs.opencue.io/).  This documentation is built with Jekyll and hosted on GitHub Pages.  Please update the documentation with any new features or changes when contributing.

### Building and Testing Documentation

1.  **Build and validate the documentation:**
    ```bash
    ./docs/build.sh
    ```

2.  **Install bundler binstubs (if needed):**

    If you encounter permission errors:
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

    Open [http://localhost:4000](http://localhost:4000) in your browser.

For detailed documentation setup instructions, testing procedures, and contribution guidelines, see [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md).

**Note:** Documentation is automatically deployed via GitHub Actions after pull requests are merged into master.

## Community and Support

*   **Discussion Forum:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io>.
*   **Slack:** Join the [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   **Working Group:** Meets biweekly at 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).
*   **Meeting Notes:**
    *   Since May 2024: [Opencue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home)
    *   Before May 2024:  [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings)