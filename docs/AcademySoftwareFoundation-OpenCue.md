<!-- OpenCue Logo -->
![OpenCue](/images/opencue_logo_with_text.png)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

# OpenCue: Your Open-Source Solution for VFX and Animation Render Management

OpenCue, an open-source render management system, empowers visual effects and animation studios to streamline complex rendering workflows.  [**Learn more on GitHub**](https://github.com/AcademySoftwareFoundation/OpenCue).

## Key Features of OpenCue

*   **Production-Proven:** Developed from the in-house render manager used on hundreds of films by Sony Pictures Imageworks.
*   **Highly Scalable:** Supports numerous concurrent machines for efficient rendering.
*   **Flexible Resource Allocation:** Utilize tagging systems to assign jobs to specific machine types.
*   **Centralized Processing:**  Jobs are processed on a central render farm, freeing up artist workstations.
*   **Native Multi-Threading:**  Supports Katana, Prman, and Arnold for optimized performance.
*   **Deployment Flexibility:** Works in multi-facility, on-premises, cloud, and hybrid environments.
*   **Granular Control:** Divide hosts into procs, each with dedicated cores and memory.
*   **Integrated Booking:** Includes automated booking capabilities.
*   **Unrestricted Scaling:** No limits on the number of procs a job can utilize.

## Get Started with OpenCue

### Quick Installation and Testing

The [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) provides a simple way to set up a local OpenCue environment. This is ideal for small tests, development, and those new to OpenCue.

### Full Installation

System administrators can find comprehensive guides for deploying OpenCue components and installing dependencies within the [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

## Documentation

### OpenCue Documentation
Comprehensive documentation, built with Jekyll and hosted on GitHub Pages, is available to help users of all skill levels get started with OpenCue. Find installation guides, user guides, API references, and tutorials at https://docs.opencue.io/.  Contributions are welcome and should include documentation updates for any new features or changes.

### Building and Testing Documentation

Before submitting a pull request with documentation changes, follow these steps:

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

4.  **Preview the documentation**
    Open http://localhost:4000 in your browser to review your changes.

For detailed setup instructions, testing procedures, and contribution guidelines, see [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md).

**Note:** Merged pull requests will automatically update the documentation via GitHub Actions ([.github/workflows/docs.yml](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs.yml)) and be available at https://docs.opencue.io/.

## Community and Support

### Meeting Notes

*   **May 2024 onwards:** Access all OpenCue meeting notes on the [OpenCue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home).
*   **Before May 2024:** Review past meeting notes in the [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) folder within the repository.

### Contact

*   **Mailing List:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io> to connect with other users and administrators.
*   **Slack:** Join the [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q)
*   **Working Group:**  Bi-weekly meetings at 2 PM PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).