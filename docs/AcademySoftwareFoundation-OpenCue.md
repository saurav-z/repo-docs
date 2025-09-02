![OpenCue](/images/opencue_logo_with_text.png)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

# OpenCue: The Open-Source Render Management System for VFX and Animation

**OpenCue empowers visual effects and animation studios to efficiently manage and scale their rendering pipelines.**  [Learn more at the original repository](https://github.com/AcademySoftwareFoundation/OpenCue).

## Key Features

*   **Production-Proven:** Built on the same render management system used by Sony Pictures Imageworks on hundreds of films.
*   **Scalable Architecture:** Supports a vast number of concurrent machines for handling large-scale rendering jobs.
*   **Flexible Resource Allocation:** Utilize tagging systems for targeted job allocation to specific machine types.
*   **Centralized Rendering:**  Jobs are processed on a central render farm, freeing up artist workstations.
*   **Native Multi-Threading:**  Optimized for performance with Katana, Prman, and Arnold.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid deployments.
*   **Advanced Resource Control:** Split hosts into numerous procs with custom core and memory requirements.
*   **Automated Booking:** Integrated for streamlined workflow.
*   **Unlimited Job Scalability:** No artificial limits on the number of procs per job.

## Quick Installation and Testing

Quickly set up and test OpenCue with the sandbox environment:

*   The sandbox environment provides an easy way to run a test OpenCue deployment locally, with all components running in separate Docker containers or Python virtual environments.
*   Ideal for small tests, development work, and for those new to OpenCue who want a simple setup for experimentation and learning.

Learn how to run the sandbox environment at https://www.opencue.io/docs/quick-starts/.

## Full Installation and Documentation

Comprehensive guides for system administrators deploying OpenCue components and installing dependencies are available in the [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

The documentation includes:

*   Installation Guides
*   User Guides
*   API References
*   Tutorials

## Contributing to OpenCue

To contribute to OpenCue, update the documentation for any new features or changes.  Each pull request should include relevant documentation updates when applicable.

### Building and Testing Documentation

Before submitting your PR, build and test the documentation:

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

**Note:** Once your pull request is merged into master, the documentation will be automatically deployed via GitHub Actions ([.github/workflows/docs.yml](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs.yml)). The updated documentation will be available at https://docs.opencue.io/.

The OpenCue documentation is now available at https://docs.opencue.io/.

## Meeting Notes

Meeting notes are available on the [Opencue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home).

For meeting notes before May 2024, refer to the [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) folder.

## Contact Us

*   **Mailing List:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io>.
*   **Slack:** Join the [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   **Working Group Meetings:** Bi-weekly at 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).