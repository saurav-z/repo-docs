![OpenCue](/images/opencue_logo_with_text.png)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

# OpenCue: The Open-Source Render Management System for VFX and Animation

**OpenCue** is an open-source render management system, offering a robust and scalable solution for managing complex rendering pipelines in visual effects and animation. ([Original Repository](https://github.com/AcademySoftwareFoundation/OpenCue))

## Key Features

*   **Industry-Proven**: Built upon the foundation of Sony Pictures Imageworks' in-house render manager, used in hundreds of films.
*   **Highly Scalable**: Designed to support numerous concurrent machines and handle large-scale rendering jobs.
*   **Flexible Resource Allocation**: Tagging systems allow for specific job allocation to specific machine types, optimizing resource utilization.
*   **Centralized Rendering**: Jobs are processed on a central render farm, freeing up artist workstations.
*   **Native Multi-Threading**: Supports Katana, Prman, and Arnold, enhancing rendering performance.
*   **Deployment Flexibility**: Supports multi-facility, on-premises, cloud, and hybrid deployments.
*   **Advanced Resource Management**: Offers the ability to split hosts into numerous "procs" with reserved cores and memory.
*   **Integrated Automated Booking**: Streamlines the job submission process.
*   **Scalability**: No limit on the number of "procs" a job can have.

## Quick Installation and Testing

Get up and running quickly with OpenCue using the sandbox environment. The sandbox provides an easy way to run a test OpenCue deployment locally, with all components running in separate Docker containers or Python virtual environments.

Read the [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) 
to learn how to set up a local OpenCue environment.

To learn how to run the sandbox environment, see https://www.opencue.io/docs/quick-starts/.

## Full Installation

Comprehensive guides for system administrators deploying OpenCue components and installing dependencies are available in the 
[OpenCue documentation](https://www.opencue.io/docs/getting-started/).

## Documentation

Detailed documentation is built with Jekyll and hosted on GitHub Pages, covering:

*   Installation Guides
*   User Guides
*   API References
*   Tutorials

### Building and Testing Documentation

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

## Meeting Notes

*   **May 2024 onwards:** [Opencue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home)
*   **Before May 2024:** [OpenCue repository in the opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) folder.

## Contact

*   **Mailing List:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io>
*   **Slack:** [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q)
*   **Working Group:** Meets biweekly at 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).