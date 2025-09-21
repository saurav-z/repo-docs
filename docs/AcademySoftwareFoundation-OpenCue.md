![OpenCue](/images/opencue_logo_with_text.png)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

# OpenCue: The Open-Source Render Management System for VFX and Animation

[**View the OpenCue Repository on GitHub**](https://github.com/AcademySoftwareFoundation/OpenCue)

OpenCue is a powerful open-source render management system designed to streamline and optimize the rendering pipeline for visual effects and animation production. This is the open-source render management system formerly used in-house at Sony Pictures Imageworks.

## Key Features

*   **Scalable Architecture:** Supports numerous concurrent machines, handling large-scale rendering jobs efficiently.
*   **Production Proven:** Based on the render manager used on hundreds of films at Sony Pictures Imageworks.
*   **Flexible Tagging:** Allows allocation of jobs to specific machine types.
*   **Centralized Processing:** Jobs are processed on a central render farm, freeing up artist workstations.
*   **Native Multi-threading:** Optimized for Katana, Prman, and Arnold rendering.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid deployments.
*   **Proc Management:** Split hosts into numerous procs with reserved resources.
*   **Integrated Booking:** Includes automated booking capabilities.
*   **No Job Limits:** No limitation on the number of procs a job can have.

## Quick Installation & Testing

Get started quickly with a local OpenCue environment using the [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md).

*   The sandbox provides a Docker-based setup for experimentation and testing.
*   Ideal for local development, learning, and initial testing.

For more information on the sandbox environment, see the [OpenCue Quick Starts documentation](https://www.opencue.io/docs/quick-starts/).

## Full Installation

System administrators can find detailed guides for deploying OpenCue components and installing dependencies in the [OpenCue Documentation - Getting Started](https://www.opencue.io/docs/getting-started).

## Documentation

Comprehensive documentation is available to guide you through installation, usage, and administration.

*   **Access Documentation:** [www.opencue.io](https://www.opencue.io)
*   **Documentation includes:** Installation guides, user guides, API references, and tutorials.

### Contributing to Documentation

When contributing to OpenCue, be sure to update the documentation for new features or changes. Each pull request should include the relevant documentation updates.

**Building and Testing Documentation**

1.  **Build and validate the documentation**

    ```bash
    ./docs/build.sh
    ```
2.  **Install bundler binstubs (if needed)**

    If you encounter permission errors:

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

    Open http://localhost:4000 in your browser to review changes.

For detailed documentation setup instructions, testing procedures, and contribution guidelines, see [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md).

**Note:** Documentation is automatically deployed upon merge to the `master` branch via GitHub Actions ([.github/workflows/docs.yml](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/.github/workflows/docs.yml)).  The updated documentation will be available at https://docs.opencue.io/.

## Meeting Notes

All OpenCue meeting notes starting from May 2024 are available on the [OpenCue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home).  For meeting notes before May 2024, see the [OpenCue repository](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings).

## Community and Support

*   **Slack:** Join the [OpenCue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q)
*   **Mailing List:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io>

## Contributors

<a href="https://github.com/AcademySoftwareFoundation/OpenCue/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AcademySoftwareFoundation/OpenCue" alt="Contributors image" />
</a>

## Contact

Working Group meets biweekly at 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).
```
Key improvements:

*   **SEO-Friendly Title & Hook:** The title includes the main keywords "OpenCue" and "render management" which helps with search engine optimization.  The first sentence provides a concise and engaging hook.
*   **Clear Headings:** Uses clear and descriptive headings to organize the information, making it easy to scan.
*   **Bulleted Key Features:** Highlights the key features using bullet points for readability and quick understanding.
*   **Concise Language:** Uses more concise and active language throughout.
*   **Call to Action (CTA):** Includes a clear call to action with a link to the GitHub repository.
*   **Documentation Emphasis:** Highlights the importance of the documentation and links to the relevant documentation.
*   **Community & Support section:**  Provides information on contacting the community and joining the mailing list.
*   **Markdown Formatting:** The formatting and structure makes this readme much cleaner.
*   **Removed Redundancy:** Avoided repeating information, combining similar sections.
*   **Simplified installation instructions:** The quick installation section provides better organization for new users.