<!-- OpenCue Logo -->
![OpenCue](/images/opencue_logo_with_text.png)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

# OpenCue: The Open-Source Render Management System for VFX and Animation

OpenCue is an open-source render management system, empowering visual effects and animation studios to efficiently manage and scale their rendering pipelines. ([See the original repo](https://github.com/AcademySoftwareFoundation/OpenCue)).

## Key Features of OpenCue

*   **Proven in Production:**  Used in hundreds of films, leveraging the experience of Sony Pictures Imageworks.
*   **Highly Scalable Architecture:** Supports numerous concurrent machines, perfect for large-scale projects.
*   **Flexible Resource Allocation:**  Utilizes tagging systems to assign jobs to specific machine types.
*   **Centralized Rendering:** Jobs processed on a central render farm, freeing up artist workstations.
*   **Native Multi-threading:** Compatible with Katana, Prman, and Arnold for optimized performance.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid deployments.
*   **Fine-grained Resource Control:** Allows splitting a host into procs with reserved core and memory.
*   **Integrated Automated Booking:** Streamlines resource allocation.
*   **Unlimited Job Size:**  No restriction on the number of procs a job can utilize.

## Quick Installation and Testing

The [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) provides instructions on how to set up a local OpenCue environment.  This environment is ideal for testing, development, and for those new to OpenCue.

## OpenCue Documentation

Comprehensive documentation is available at [https://www.opencue.io/docs/](https://www.opencue.io/docs/) including:

*   Installation guides
*   User guides
*   API references
*   Tutorials

### Building and Testing Documentation

To contribute, ensure your documentation updates are built, tested, and previewed before submitting a pull request.

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

For detailed documentation setup instructions, testing procedures, and contribution guidelines, see [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md).

**Note:**  Documentation is automatically deployed via GitHub Actions after a merge.

## Meeting Notes

*   **May 2024 onwards:**  [OpenCue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home)
*   **Before May 2024:**  [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings)

## Contact Us

*   **Discussion Forum (Users & Admins):**  [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) ([opencue-user@lists.aswf.io](mailto:opencue-user@lists.aswf.io))
*   **Slack:** [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q)
*   **Bi-weekly Working Group:** Meets at 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6)