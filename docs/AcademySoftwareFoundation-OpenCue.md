# OpenCue: Your Open-Source Render Management Solution for VFX and Animation

[![OpenCue Logo](images/opencue_logo_with_text.png)](https://github.com/AcademySoftwareFoundation/OpenCue)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
[![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)]()
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

**OpenCue** empowers visual effects and animation studios to efficiently manage and scale their rendering pipelines with a robust, open-source solution.

## Key Features of OpenCue

*   **Industry-Proven:** Based on the in-house render manager developed by Sony Pictures Imageworks and used on hundreds of films.
*   **Highly Scalable:**  Supports numerous concurrent machines for handling demanding workloads.
*   **Flexible Resource Allocation:** Use tagging systems to direct jobs to specific machine types.
*   **Centralized Rendering:** Jobs are processed on a central render farm, freeing up artists' workstations.
*   **Native Multi-threading:** Provides support for popular rendering software like Katana, Prman, and Arnold.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid environments.
*   **Granular Resource Control:** Split hosts into numerous "procs" with dedicated cores and memory.
*   **Integrated Automated Booking:** Streamlines job scheduling and resource allocation.
*   **No Job Limits:** Jobs can have an unlimited number of "procs".

## Getting Started with OpenCue

### Quick Installation and Testing

The [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) provides a simple way to run a local OpenCue environment, ideal for testing and development.

*   The sandbox utilizes Docker containers or Python virtual environments.
*   Perfect for experimentation and learning.

To learn how to run the sandbox environment, see the [OpenCue Quick Starts documentation](https://www.opencue.io/docs/quick-starts/).

### Full Installation

System administrators can find detailed guides for deploying OpenCue components and installing dependencies in the comprehensive [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

## Documentation

Comprehensive documentation is available at [www.opencue.io](https://www.opencue.io) including:

*   Installation guides
*   User guides
*   API references
*   Tutorials

To contribute to OpenCue, please update the documentation for any new features or changes. Each pull request should include relevant documentation updates when applicable.

### Building and Testing Documentation

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

**Note:**  Documentation is automatically deployed via GitHub Actions after merges.  Updated documentation is available at https://docs.opencue.io/.

## Meeting Notes

*   **May 2024 Onward:** [OpenCue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home)
*   **Prior to May 2024:** [OpenCue repository](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings)

## Contributors

[<img src="https://contrib.rocks/image?repo=AcademySoftwareFoundation/OpenCue" alt="Contributors image" />](https://github.com/AcademySoftwareFoundation/OpenCue/graphs/contributors)

## Get Involved

*   **Join the discussion forum:** [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io>.
*   **Join the Slack channel:** [OpenCue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   **Working Group:** Meets biweekly at 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).

**[View the original repository on GitHub](https://github.com/AcademySoftwareFoundation/OpenCue)**