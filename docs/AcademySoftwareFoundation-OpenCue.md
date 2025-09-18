# OpenCue: The Open Source Render Management System for VFX and Animation

[![OpenCue](https://raw.githubusercontent.com/AcademySoftwareFoundation/OpenCue/master/images/opencue_logo_with_text.png)](https://github.com/AcademySoftwareFoundation/OpenCue)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

**OpenCue is an open-source render management system designed to streamline and optimize rendering workflows for visual effects (VFX) and animation pipelines.**

## Key Features

*   **Scalable Architecture:** Supports a vast number of concurrent machines for handling large-scale rendering jobs.
*   **Production Proven:** Built upon the render manager originally used by Sony Pictures Imageworks on hundreds of films.
*   **Resource Allocation:** Utilize tagging systems to assign specific jobs to specific machine types, ensuring optimal resource allocation.
*   **Centralized Processing:** All jobs are processed on a central render farm, freeing up artist workstations.
*   **Multi-Threading Support:** Native multi-threading support for Katana, Prman, and Arnold.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid deployments.
*   **Granular Control:** Split a host into a large number of [procs](https://www.opencue.io/docs/concepts/glossary/#proc), each with its own reserved core and memory requirements.
*   **Automated Booking:** Integrated automated booking for efficient resource management.
*   **Unlimited Procs:** No limit on the number of [procs](https://www.opencue.io/docs/concepts/glossary/#proc) a job can have.

## Getting Started

To learn more about installing, using, and administering OpenCue, visit the official website: [www.opencue.io](https://www.opencue.io).

### Quick Installation and Testing

For a simple setup to experiment with OpenCue, explore the [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) to set up a local OpenCue environment.

### Full Installation

System administrators can find guides for deploying OpenCue components and installing dependencies in the [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

## Documentation

OpenCue documentation is built with Jekyll and hosted on GitHub Pages. It includes installation guides, user guides, API references, and tutorials to help users get started with OpenCue.

When contributing, update the documentation for new features or changes. Each pull request should include relevant documentation updates.

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

**Note:** Once your pull request is merged into master, the documentation will be automatically deployed via GitHub Actions ([.github/workflows/docs.yml](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/.github/workflows/docs.yml)). The updated documentation will be available at https://docs.opencue.io/.

## Meeting Notes

Meeting notes are available on the [OpenCue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home) (starting May 2024).

For older meeting notes, see the [OpenCue repository](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) in the `opencue/tsc/meetings` folder.

## Contributors

<a href="https://github.com/AcademySoftwareFoundation/OpenCue/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AcademySoftwareFoundation/OpenCue" alt="Contributors image" />
</a>

## Contact and Support

*   **Mailing List:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io> for user and admin discussions.
*   **Slack:** Join the [OpenCue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   **Working Group:** The Working Group meets bi-weekly at 2 PM PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).

**[View the OpenCue Repository on GitHub](https://github.com/AcademySoftwareFoundation/OpenCue)**