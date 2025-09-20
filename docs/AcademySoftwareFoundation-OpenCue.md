[![OpenCue](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/images/opencue_logo_with_text.png?raw=true)](https://github.com/AcademySoftwareFoundation/OpenCue)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

# OpenCue: The Open Source Render Management System

**OpenCue is an open-source render management system designed to streamline and optimize visual effects and animation production pipelines.**  ([View the original repository](https://github.com/AcademySoftwareFoundation/OpenCue))

## Key Features of OpenCue:

*   **Scalable Architecture:** Supports numerous concurrent machines for handling large-scale rendering jobs.
*   **Advanced Tagging:** Allocate specific jobs to specific machine types for optimal resource utilization.
*   **Centralized Rendering:** Process jobs on a central render farm, freeing up artist workstations.
*   **Multi-Threading Support:** Native multi-threading capabilities for popular renderers like Katana, Prman, and Arnold.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid deployment models.
*   **Proc Management:** Split hosts into numerous "procs" with reserved cores and memory for granular control.
*   **Integrated Booking & Unlimited Procs:** Automated booking and no limitations on the number of procs a job can have.
*   **Production-Proven:** Used in production on hundreds of films by Sony Pictures Imageworks.

## Get Started with OpenCue:

*   **Installation:** Explore the [OpenCue Quick Starts documentation](https://www.opencue.io/docs/quick-starts/) to quickly set up a local OpenCue environment using the sandbox. For full installation guides, see the [OpenCue Documentation - Getting Started](https://www.opencue.io/docs/getting-started).

*   **Documentation:** Comprehensive documentation is available at [www.opencue.io](https://www.opencue.io), including installation guides, user guides, API references, and tutorials.

*   **Sandbox Environment:**  The [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) provides an easy way to run a test OpenCue deployment locally using Docker containers or Python virtual environments.

## Contributing and Documentation:

*   When contributing to OpenCue, please update the documentation for any new features or changes. Each pull request should include relevant documentation updates when applicable.

*   For detailed documentation setup instructions, testing procedures, and contribution guidelines, see [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md).

*   The OpenCue documentation is automatically deployed via GitHub Actions and is available at [https://docs.opencue.io/](https://docs.opencue.io/).

## Building and Testing Documentation:

If you make changes to `OpenCue/docs`, follow these steps before submitting your PR:

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

## Meeting Notes:

*   Starting from May 2024, all OpenCue meeting notes are stored on the [OpenCue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home).
*   For meeting notes before May 2024, refer to the OpenCue repository in the [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) folder.

## Connect with the OpenCue Community:

*   **Slack:** Join the [OpenCue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q) for collaboration.
*   **Mailing List:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io> for user and admin discussions.
*   **Working Group:**  The Working Group meets biweekly at 2 pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).

## Contributors:

<a href="https://github.com/AcademySoftwareFoundation/OpenCue/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AcademySoftwareFoundation/OpenCue" alt="Contributors image" />
</a>