![OpenCue](/images/opencue_logo_with_text.png)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

## OpenCue: Your Open-Source Solution for Efficient Render Management

**OpenCue is a powerful open-source render management system designed to streamline visual effects and animation production.**  This document provides an overview of OpenCue's features, installation, and resources.  For more in-depth information, please visit the [OpenCue GitHub Repository](https://github.com/AcademySoftwareFoundation/OpenCue).

## Key Features of OpenCue

OpenCue offers a comprehensive suite of features to optimize your rendering workflow:

*   **Production-Proven:** Built upon the foundation of Sony Pictures Imageworks' in-house render manager, used on hundreds of films.
*   **Scalable Architecture:** Supports a massive number of concurrent machines for demanding workloads.
*   **Flexible Tagging System:** Allocate specific jobs to specific machine types for optimal resource utilization.
*   **Centralized Processing:**  Jobs are processed on a central render farm, freeing up artist workstations.
*   **Native Multi-Threading:** Supports Katana, Prman, and Arnold for improved performance.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid deployments.
*   **Granular Resource Control:** Divide a host into multiple [procs](https://www.opencue.io/docs/concepts/glossary/#proc), each with its own dedicated core and memory.
*   **Integrated Booking:** Automated booking to simplify job submission.
*   **Unlimited Scalability:** No limitations on the number of [procs](https://www.opencue.io/docs/concepts/glossary/#proc) a job can utilize.

## Getting Started

### Quick Installation and Testing

The [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) provides an easy way to set up a local OpenCue environment using Docker containers or Python virtual environments. This is ideal for testing, development, and learning.

To learn how to run the sandbox environment, see the [OpenCue Quick Starts documentation](https://www.opencue.io/docs/quick-starts/).

### Full Installation

System administrators can find detailed guides for deploying OpenCue components and installing dependencies in the [OpenCue Documentation - Getting Started](https://www.opencue.io/docs/getting-started).

## Documentation & Resources

*   **Comprehensive Documentation:**  Access the official documentation at [www.opencue.io](https://www.opencue.io).  It provides installation guides, user guides, API references, and tutorials.
*   **Video Tutorials:** Watch video tutorials on the [OpenCue Playlist](https://www.youtube.com/playlist?list=PL9dZxafYCWmzSBEwVT2AQinmZolYqBzdp) by the Academy Software Foundation (ASWF).

## Contributing to OpenCue

When contributing to OpenCue, always update the documentation to reflect any new features or changes. Ensure that each pull request includes the necessary documentation updates.

### Building and Testing Documentation

1.  **Build and validate the documentation:**
    ```bash
    ./docs/build.sh
    ```
2.  **Install bundler binstubs (if needed):**  If you encounter permission errors, run these commands from within the `docs/` directory:
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

**Note:** Documentation is automatically deployed via GitHub Actions ([.github/workflows/docs.yml](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/.github/workflows/docs.yml)) after a pull request is merged. The updated documentation is available at https://docs.opencue.io/.

## Meeting Notes

*   **Current Notes:**  Starting from May 2024, all meeting notes are stored on the [OpenCue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home).
*   **Older Notes:**  For meeting notes before May 2024, refer to the OpenCue repository in the [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) folder.

## Contributors

<a href="https://github.com/AcademySoftwareFoundation/OpenCue/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AcademySoftwareFoundation/OpenCue" alt="Contributors image" />
</a>

## Contact Us

*   **Slack Channel:** Join the [OpenCue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q) for collaboration.
*   **Working Group:**  The Working Group meets biweekly at 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).
*   **Mailing List:**  Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io> for user and admin discussions.
```
Key improvements and SEO considerations:

*   **Clear and Concise Hook:** "OpenCue is a powerful open-source render management system designed to streamline visual effects and animation production." immediately grabs attention.
*   **Strategic Keyword Usage:**  Keywords like "render management," "open-source," "visual effects," "animation production," "scalability," and names of supported software like Katana, Prman, and Arnold are used naturally throughout the document.
*   **Descriptive Headings:** Headings clearly define sections, improving readability and SEO.
*   **Bulleted Key Features:** Highlights the most important features, making the benefits easily scannable.
*   **Internal and External Links:**  Provides links to key resources, including the main GitHub repo (mentioned in the hook), documentation, and other helpful pages within the project.  This improves user experience and SEO.
*   **Call to Action:** Encourages users to explore the project and get involved.
*   **Well-Organized Structure:**  The structure is logical, making it easy for users to find information.
*   **Contextual Information:**  Provides relevant information about the project's history and use cases.
*   **Contributor Section:** Includes a contributor graphic and mentions the Slack channel, mailing list, and bi-weekly meetings for improved community participation.
*   **Documentation Emphasis:** Clearly states how to contribute to documentation with testing instructions.
*   **Alt text for the image:**  I've made sure the images are also well described.