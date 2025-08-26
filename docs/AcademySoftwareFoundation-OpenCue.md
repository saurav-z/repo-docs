# OpenCue: The Open Source Render Management System for VFX and Animation

**Manage and scale your rendering jobs with OpenCue, the powerful open-source render management system.** ([Original Repository](https://github.com/AcademySoftwareFoundation/OpenCue))

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

## Key Features

*   **Scalable Architecture:** Supports numerous concurrent machines for handling large-scale rendering jobs.
*   **Job Management:** Break down complex jobs into individual tasks and submit them to a configurable dispatch queue.
*   **Resource Allocation:** Utilize tagging systems to direct jobs to specific machine types.
*   **Centralized Processing:** Jobs are processed on a central render farm, freeing up artist workstations.
*   **Native Multi-Threading:** Supports Katana, Prman, and Arnold for optimized performance.
*   **Deployment Flexibility:** Compatible with multi-facility, on-premises, cloud, and hybrid environments.
*   **Granular Control:** Split hosts into numerous procs, each with dedicated core and memory resources.
*   **Automated Booking:** Integrated automated booking simplifies resource management.
*   **Scalability:** No limits on the number of procs a job can have.

## Getting Started

### Quick Installation & Testing

Quickly set up a local OpenCue environment using the sandbox environment. This is ideal for testing, development, and learning purposes. Instructions are available in the [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) and at [www.opencue.io/docs/quick-starts/](https://www.opencue.io/docs/quick-starts/).

### Full Installation

System administrators can find guides for deploying OpenCue components and installing dependencies in the [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

## Documentation and Resources

### OpenCue Documentation

The comprehensive OpenCue documentation is built with Jekyll and hosted on GitHub Pages. It provides:

*   Installation guides
*   User guides
*   API references
*   Tutorials

**Documentation Updates:** Please update the documentation whenever you contribute new features or make changes to the code. Ensure that you build and test the documentation before submitting your pull request.

#### Building and Testing Documentation

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

    Open http://localhost:4000 in your browser to review your changes.

For detailed instructions, refer to [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md).

**Note:** Documentation is automatically deployed via GitHub Actions after a merge, and updated at https://docs.opencue.io/.

### Additional Resources

*   **Learn more:** Visit [www.opencue.io](https://www.opencue.io).
*   **YouTube:** Explore the [OpenCue Playlist](https://www.youtube.com/playlist?list=PL9dZxafYCWmzSBEwVT2AQinmZolYqBzdp) of the Academy Software Foundation (ASWF).

## Community and Support

### Meeting Notes

*   **May 2024 onwards:** [Opencue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home)
*   **Before May 2024:** [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings)

### Contact Us

*   **Discussion Forum:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io>.
*   **Slack:** Join the [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   **Working Group:** Biweekly meetings at 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).