![OpenCue](/images/opencue_logo_with_text.png)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

# OpenCue: The Open-Source Render Management System for VFX & Animation

**OpenCue, the open-source render management system, empowers visual effects and animation studios to efficiently manage and scale their rendering pipelines.** Developed by the Academy Software Foundation, OpenCue offers robust features for handling complex jobs and optimizing resource allocation.  [Learn more on GitHub](https://github.com/AcademySoftwareFoundation/OpenCue).

## Key Features

*   **Production-Proven:** Built upon the render manager used in-house at Sony Pictures Imageworks and employed on hundreds of films.
*   **Highly Scalable:** Supports numerous concurrent machines for efficient rendering.
*   **Flexible Resource Allocation:** Utilize tagging systems to assign jobs to specific machine types.
*   **Centralized Rendering:** Jobs are processed on a central render farm, freeing up artist workstations.
*   **Native Multi-Threading:**  Provides seamless support for Katana, Prman, and Arnold.
*   **Deployment Options:** Supports multi-facility, on-premises, cloud, and hybrid deployments.
*   **Granular Resource Control:** Split hosts into numerous [procs](https://www.opencue.io/docs/concepts/glossary/#proc), each with dedicated core and memory.
*   **Integrated Booking:** Includes automated job booking capabilities.
*   **Unlimited Job Capacity:** No limitations on the number of [procs](https://www.opencue.io/docs/concepts/glossary/#proc) per job.

## Get Started

*   **Learn More:** Visit the official website at [www.opencue.io](https://www.opencue.io) and explore the [OpenCue YouTube Playlist](https://www.youtube.com/playlist?list=PL9dZxafYCWmzSBEwVT2AQinmZolYqBzdp) for helpful videos.
*   **Quick Installation & Testing:** Set up a local OpenCue environment quickly with the [sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) that uses Docker containers.
*   **Full Installation:** Review system admin guides for deploying OpenCue components and dependencies within the official [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

## Documentation & Contribution

Comprehensive documentation is available at [https://docs.opencue.io/](https://docs.opencue.io/). This includes installation guides, user guides, API references, and tutorials.

**Contributing:** When contributing, always update the documentation for new features or changes.  Build and test your documentation changes by following these steps:

1.  Build and validate the documentation:
    ```bash
    ./docs/build.sh
    ```
2.  Install bundler binstubs (if needed):
    ```bash
    cd docs/
    bundle binstubs --all
    ```
3.  Run the documentation locally:
    ```bash
    cd docs/
    bundle exec jekyll serve --livereload
    ```
4.  Preview the documentation by opening http://localhost:4000 in your browser.

See [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md) for full documentation setup instructions, testing procedures, and contribution guidelines. Documentation is automatically deployed after pull requests are merged.

## Meeting Notes

*   **May 2024 Onward:** Access meeting notes on the [OpenCue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home).
*   **Before May 2024:** Review meeting notes within the [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) folder in the repository.

## Contact

*   **Mailing List:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) at <opencue-user@lists.aswf.io>.
*   **Slack:** Join the [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   **Working Group:** Participate in bi-weekly meetings at 2 pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).