[![OpenCue](/images/opencue_logo_with_text.png)](https://github.com/AcademySoftwareFoundation/OpenCue)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

# OpenCue: The Open-Source Render Management System for VFX and Animation

OpenCue is a powerful, open-source render management system designed to streamline and optimize your visual effects and animation pipelines. ([View the original repo](https://github.com/AcademySoftwareFoundation/OpenCue))

## Key Features of OpenCue

*   **Industry-Proven:** Leverages the architecture of Sony Pictures Imageworks' in-house render manager, used on hundreds of films.
*   **Scalable Architecture:** Supports a large number of concurrent machines for demanding workloads.
*   **Flexible Resource Allocation:**  Provides tagging systems for targeting jobs to specific machine types.
*   **Centralized Processing:** Jobs are processed on a central render farm, freeing up artist workstations.
*   **Native Multi-Threading Support:** Optimized for Katana, Prman, and Arnold rendering engines.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid deployments.
*   **Advanced Resource Management:** Allows splitting hosts into numerous procs with reserved cores and memory.
*   **Integrated Automated Booking:** Streamlines job submission and resource allocation.
*   **Unlimited Job Capacity:** No inherent limit on the number of procs a job can utilize.

## Get Started with OpenCue

### Quick Installation (Sandbox)

Easily test and experiment with OpenCue using the sandbox environment:

*   Run a local OpenCue deployment using Docker containers or Python virtual environments.
*   Ideal for small tests, development work, and learning the system.
*   See the [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) and [quick start guides](https://www.opencue.io/docs/quick-starts/) for details.

### Full Installation

System administrators can find detailed guides for deploying OpenCue components and installing dependencies in the [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

## OpenCue Documentation

Comprehensive documentation is available to help you with installation, usage, and administration.  The documentation includes:

*   Installation guides
*   User guides
*   API references
*   Tutorials

To contribute to OpenCue, update the documentation for any new features or changes. Ensure you build and test the documentation before submitting your PR.

### Building and Testing Documentation

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
    Open http://localhost:4000 in your browser to review your changes.

For detailed documentation setup instructions, testing procedures, and contribution guidelines, see [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md).

**Note:** The documentation will be automatically deployed upon merging a pull request into master via GitHub Actions ([.github/workflows/docs.yml](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/.github/workflows/docs.yml)). The updated documentation will be accessible at https://docs.opencue.io/.

## Meeting Notes

Find the latest OpenCue meeting notes:

*   **May 2024 onwards:** [Opencue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home)
*   **Before May 2024:** [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) folder in the repository.

## Contact Us

*   **Mailing List:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io> for user and admin discussions.
*   **Slack:** Join the [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   **Working Group:** Bi-weekly meetings at 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).