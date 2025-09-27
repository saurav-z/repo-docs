[![OpenCue Logo](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/images/opencue_logo_with_text.png?raw=true)](https://github.com/AcademySoftwareFoundation/OpenCue)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

## OpenCue: The Open Source Render Management System for VFX and Animation

OpenCue is an open-source render management system designed to streamline and optimize rendering workflows for visual effects and animation studios.  This powerful system, originally developed by Sony Pictures Imageworks, empowers studios to efficiently manage complex rendering jobs at scale.

**[Visit the OpenCue GitHub Repository](https://github.com/AcademySoftwareFoundation/OpenCue)**

### Key Features of OpenCue:

*   **Scalable Architecture:** Supports a large number of concurrent machines for efficient rendering.
*   **Job Management:** Break down complex jobs into individual tasks, processed on a central render farm.
*   **Resource Allocation:** Utilize tagging systems for allocating jobs to specific machine types.
*   **Multi-Threading Support:** Native multi-threading that supports Katana, Prman, and Arnold.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid deployments.
*   **Procs:** Split a host into a large number of [procs](https://www.opencue.io/docs/concepts/glossary/#proc), each with their own reserved core and memory requirements.
*   **Automated Booking:** Integrated automated booking.
*   **No Limits:** No limit on the number of [procs](https://www.opencue.io/docs/concepts/glossary/#proc) a job can have.
*   **Production Proven:** Proven in-house render manager used on hundreds of films by Sony Pictures Imageworks.

### Learn More

*   **Documentation:** Comprehensive documentation is available at [www.opencue.io](https://www.opencue.io). It includes installation guides, user guides, API references, and tutorials.
*   **YouTube:** Explore the [OpenCue Playlist](https://www.youtube.com/playlist?list=PL9dZxafYCWmzSBEwVT2AQinmZolYqBzdp) on the Academy Software Foundation (ASWF) YouTube channel.

### Quick Installation and Tests

*   **Sandbox Environment:** The [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) provides instructions for setting up a local OpenCue environment using Docker containers or Python virtual environments.  This is ideal for testing and development.
*   **Quick Starts:**  Get started quickly with the [OpenCue Quick Starts documentation](https://www.opencue.io/docs/quick-starts/).

### Full Installation

System administrators can find detailed guides for deploying OpenCue components and installing dependencies in the [OpenCue Documentation - Getting Started](https://www.opencue.io/docs/getting-started).

### Documentation Contributions

Contribute to the OpenCue documentation!  Follow these steps:

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
4.  **Preview the documentation:** Open http://localhost:4000 in your browser.

For detailed documentation setup instructions, testing procedures, and contribution guidelines, see [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md).

**Note:** Documentation is automatically deployed via GitHub Actions ([.github/workflows/docs.yml](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/.github/workflows/docs.yml)) to https://docs.opencue.io/.

### Meeting Notes

*   **May 2024 onwards:**  Meeting notes are stored on the [OpenCue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home).
*   **Prior to May 2024:**  Refer to the [OpenCue repository](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings).

### Contributors

See the amazing community contributions:
<a href="https://github.com/AcademySoftwareFoundation/OpenCue/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AcademySoftwareFoundation/OpenCue" alt="Contributors image" />
</a>

### Contact Us

*   **Slack:** Join the [OpenCue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   **Working Group:** Meets biweekly at 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).
*   **Mailing List:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io>.
```
Key improvements and explanations:

*   **SEO-Optimized Title and Hook:** The title clearly states what OpenCue is and the one-sentence hook immediately grabs attention.
*   **Clear Headings:**  Uses standard Markdown headings for easy navigation.
*   **Bulleted Key Features:**  Highlights important features, making them easy to scan.
*   **Concise Summaries:**  Each section is summarized to remove unnecessary verbiage while retaining important information.
*   **Actionable Calls to Action:**  Links to the repo and documentation are prominent.
*   **Contributor Visualization:** Included the contributors image.
*   **Contact Information:** Included contact methods for the community.
*   **Removed redundant links:** Removed redundant internal links to simplify the structure.