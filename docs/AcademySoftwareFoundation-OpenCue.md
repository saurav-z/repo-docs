![OpenCue](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/images/opencue_logo_with_text.png?raw=true)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

# OpenCue: The Open Source Render Management System for VFX and Animation

**OpenCue** is an open-source, high-performance render management system designed to streamline and accelerate visual effects and animation pipelines. This is the place for documentation, tutorials, and resources.  See the [original repository](https://github.com/AcademySoftwareFoundation/OpenCue) for more information.

## Key Features

*   **Scalable Architecture:** Designed to handle numerous concurrent machines and complex rendering jobs at scale.
*   **Proven Performance:** Based on the Sony Pictures Imageworks in-house render manager, used on hundreds of films.
*   **Flexible Resource Allocation:**  Utilize tagging systems to target specific machine types and optimize rendering.
*   **Centralized Job Processing:** Jobs are managed on a central render farm, freeing up artist workstations.
*   **Native Multi-Threading:** Supports Katana, Prman, and Arnold for optimized performance.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid deployments.
*   **Granular Resource Control:**  Split hosts into numerous [procs](https://www.opencue.io/docs/concepts/glossary/#proc), each with reserved cores and memory.
*   **Integrated Automated Booking:** Simplifies resource management.
*   **Unlimited Job Size:** No limitations on the number of [procs](https://www.opencue.io/docs/concepts/glossary/#proc) a job can utilize.

## Quick Start

Get up and running quickly with the OpenCue sandbox environment.

*   The sandbox environment offers a simplified way to run OpenCue locally, using Docker containers or Python virtual environments.
*   Ideal for initial testing, development, and learning the core concepts of OpenCue.

See the [OpenCue Quick Starts documentation](https://www.opencue.io/docs/quick-starts/) to get started.

## Installation and Documentation

### Full Installation

System administrators can find detailed guides for deploying OpenCue components and installing dependencies in the  [OpenCue Documentation - Getting Started](https://www.opencue.io/docs/getting-started).

### Building and Testing Documentation

If you plan to contribute, remember to build and test any documentation changes.

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
    Open http://localhost:4000 in your browser.

For detailed documentation setup instructions, testing procedures, and contribution guidelines, see [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md).

**Note:** Documentation is automatically deployed via GitHub Actions after a successful pull request. The updated documentation is available at https://docs.opencue.io/.

## Resources

*   **Learn More:** Visit [www.opencue.io](https://www.opencue.io) for in-depth information on installation, usage, and administration.
*   **YouTube:** Explore the [OpenCue Playlist](https://www.youtube.com/playlist?list=PL9dZxafYCWmzSBEwVT2AQinmZolYqBzdp) by the Academy Software Foundation (ASWF).
*   **Meeting Notes:** Find meeting notes on the [OpenCue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home) (starting May 2024).  For notes prior to May 2024, see the [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) folder.

## Community

*   **Slack:** Join the [OpenCue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q) for collaboration.
*   **Working Group:**  Attend biweekly meetings at 2 PM PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).
*   **Mailing List:**  Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io> for user and admin discussions.

## Contributors

<a href="https://github.com/AcademySoftwareFoundation/OpenCue/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AcademySoftwareFoundation/OpenCue" alt="Contributors image" />
</a>