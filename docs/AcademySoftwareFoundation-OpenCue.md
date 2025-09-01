# OpenCue: The Open-Source Render Management System for VFX & Animation

![OpenCue](/images/opencue_logo_with_text.png)

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

**OpenCue empowers visual effects and animation studios to streamline their rendering pipelines with a robust, scalable, and open-source solution.** Originally developed by Sony Pictures Imageworks, OpenCue is now available for anyone to use.

[View the OpenCue Repository on GitHub](https://github.com/AcademySoftwareFoundation/OpenCue)

## Key Features of OpenCue

*   **Industry-Proven Technology:** Built upon the same render management system used on hundreds of films by Sony Pictures Imageworks.
*   **Highly Scalable Architecture:** Designed to handle large-scale rendering jobs with numerous concurrent machines.
*   **Flexible Resource Allocation:** Utilize tagging systems for job-specific resource allocation based on machine types.
*   **Centralized Rendering:** Offloads rendering tasks from artist workstations to a central render farm.
*   **Native Multi-Threading Support:** Compatible with industry-standard rendering software like Katana, Prman, and Arnold.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid rendering deployments.
*   **Efficient Resource Management:** Allows splitting a host into multiple "procs" for fine-grained control over core and memory allocation.
*   **Integrated Automated Booking:** Streamlines the job submission process.
*   **Unlimited Job Scalability:** No limits on the number of "procs" a job can have, providing incredible flexibility.

## Getting Started

### Quick Installation with Sandbox Environment

For easy testing and development, use the [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md) to set up a local OpenCue environment. This environment runs all components in separate Docker containers or Python virtual environments, ideal for small tests and learning. See https://www.opencue.io/docs/quick-starts/ to learn more.

### Full Installation

System administrators can find detailed guides for deploying OpenCue components and installing dependencies in the [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

## Documentation & Resources

*   **Official Documentation:** Access comprehensive documentation, including installation guides, user guides, API references, and tutorials at [www.opencue.io](https://www.opencue.io).
*   **YouTube Playlist:** Learn more about OpenCue through the official [OpenCue Playlist](https://www.youtube.com/playlist?list=PL9dZxafYCWmzSBEwVT2AQinmZolYqBzdp) of the Academy Software Foundation (ASWF).

## Contributing to OpenCue

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

**Note:** Once your pull request is merged into master, the documentation will be automatically deployed via GitHub Actions ([.github/workflows/docs.yml](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs.yml)). The updated documentation will be available at https://docs.opencue.io/.

## Community & Support

*   **Discussion Forum:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io> to connect with users and admins.
*   **Slack Channel:** Join the [Opencue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q) for real-time discussions.
*   **Bi-weekly Meetings:** The Working Group meets biweekly at 2pm PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).
*   **Meeting Notes:** For meeting notes starting May 2024, refer to the [Opencue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home). For prior meeting notes, see the [opencue/tsc/meetings](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) folder in the repository.