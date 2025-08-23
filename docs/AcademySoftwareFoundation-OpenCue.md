[![OpenCue](/images/opencue_logo_with_text.png)](https://github.com/AcademySoftwareFoundation/OpenCue)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

# OpenCue: The Open-Source Render Management System for VFX and Animation

OpenCue is the ultimate open-source solution for managing your rendering pipeline, designed to handle complex jobs at scale, streamlining the production of visual effects and animation.

## Key Features

*   **Production-Proven:** Built upon the foundation of Sony Pictures Imageworks' in-house render manager, used on hundreds of films.
*   **Highly Scalable:** Supports numerous concurrent machines for efficient rendering.
*   **Flexible Resource Allocation:** Tagging systems allow for precise job assignment to specific machine types.
*   **Centralized Rendering:** Jobs are processed on a central render farm, freeing up artist workstations.
*   **Native Multi-Threading:** Optimized for popular rendering software like Katana, Prman, and Arnold.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid environments.
*   **Fine-Grained Control:** Allows splitting a host into multiple [procs](https://www.opencue.io/docs/concepts/glossary/#proc), each with dedicated resources.
*   **Integrated Automated Booking:** Streamlines resource management.
*   **Unlimited Job Size:** No limit on the number of [procs](https://www.opencue.io/docs/concepts/glossary/#proc) a job can utilize.

## Getting Started

### Quick Installation & Testing

Quickly set up a local OpenCue environment using the sandbox:

*   The sandbox environment offers an easy way to run a test OpenCue deployment locally, with all components running in separate Docker containers or Python virtual environments.
*   Ideal for small tests, development work, and for those new to OpenCue who want a simple setup for experimentation and learning.

See the [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) and [quick-starts](https://www.opencue.io/docs/quick-starts/) for more information.

### Full Installation & Documentation

Comprehensive guides for system admins deploying OpenCue and installing dependencies are available in the main [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

## Documentation

The OpenCue documentation, hosted on GitHub Pages, includes installation guides, user guides, API references, and tutorials.  Find the documentation at [https://docs.opencue.io/](https://docs.opencue.io/).

### Contributing to Documentation

When contributing to OpenCue, remember to update the documentation for new features or changes. Each pull request should include documentation updates when applicable.

**Building & Testing Documentation**

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

**Note:**  Documentation is automatically deployed via GitHub Actions after a merge to the `master` branch.

## Resources

*   **Learn More:** [www.opencue.io](https://www.opencue.io)
*   **YouTube:** [OpenCue Playlist](https://www.youtube.com/playlist?list=PL9dZxafYCWmzSBEwVT2AQinmZolYqBzdp)
*   **GitHub Repository:** [AcademySoftwareFoundation/OpenCue](https://github.com/AcademySoftwareFoundation/OpenCue)

## Meeting Notes

*   **Current:** All OpenCue meeting notes from May 2024 onwards are stored on the [OpenCue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home).
*   **Previous:** Meeting notes before May 2024 are in the [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) folder of this repository.

## Contact & Community

*   **Discussion Forum:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io>.
*   **Slack:** Join the [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   **Working Group:** Meets biweekly at 2 PM PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).