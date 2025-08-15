![OpenCue](/images/opencue_logo_with_text.png)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

# OpenCue: The Open-Source Render Management System for VFX and Animation

**OpenCue, an open-source render management system, empowers visual effects and animation studios to efficiently manage complex rendering workflows at scale.**  ([View the original repository](https://github.com/AcademySoftwareFoundation/OpenCue))

## Key Features

*   **Proven Technology:** Developed from Sony Pictures Imageworks' in-house render manager, used on hundreds of films.
*   **Highly Scalable Architecture:** Supports numerous concurrent machines for large-scale rendering.
*   **Flexible Resource Allocation:** Utilize tagging systems to assign jobs to specific machine types and efficiently manage resources.
*   **Centralized Processing:** Jobs are processed on a central render farm, freeing up artist workstations.
*   **Optimized Performance:** Native multi-threading support for Katana, Prman, and Arnold.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid deployment options.
*   **Granular Resource Control:** Split hosts into procs with reserved core and memory, allowing for precise resource allocation.
*   **Integrated Automation:** Offers integrated automated booking for streamlined workflows.
*   **Scalability:** No limits on the number of procs a job can have.

## Getting Started

### Installation

*   Refer to the [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) for a quick local test environment.
*   For full installation and deployment guides, see the main [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

### Documentation

Comprehensive documentation, including installation guides, user guides, API references, and tutorials, is available at [https://www.opencue.io](https://www.opencue.io).

### Building and Testing Documentation

If you make changes to `OpenCue/docs`, please build and test the documentation before submitting your PR:

1.  **Build and validate the documentation**
    ```bash
    ./docs/build.sh
    ```

2.  **Install bundler binstubs (if needed)**

    If you encounter permission errors when installing to system directories:
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

**Note:** Once your pull request is merged into master, the documentation will be automatically deployed via GitHub Actions ([.github/workflows/docs.yml](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md)). The updated documentation will be available at https://docs.opencue.io/.

The OpenCue documentation is now available at https://docs.opencue.io/.

## Meeting Notes

*   Starting from May 2024, all OpenCue meeting notes are stored on the [OpenCue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home).
*   For meeting notes before May 2024, refer to the Opencue repository in the [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) folder.

## Community and Support

*   **Mailing List:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io> for user and admin discussions.
*   **Slack:** Join the [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   **Working Group:** The Working Group meets biweekly at 2 pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).