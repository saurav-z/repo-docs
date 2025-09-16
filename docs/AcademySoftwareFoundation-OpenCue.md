<div align="center">
  <a href="https://github.com/AcademySoftwareFoundation/OpenCue">
    <img src="/images/opencue_logo_with_text.png" alt="OpenCue Logo" width="400"/>
  </a>
</div>

[![Supported VFX Platform Versions](https://img.shields.io/badge/vfx%20platform-2021--2024-lightgrey.svg)](http://www.vfxplatform.com/)
![Supported Python Versions](https://img.shields.io/badge/python-3.6+-blue.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2837/badge)](https://bestpractices.coreinfrastructure.org/projects/2837)

# OpenCue: The Open-Source Render Management System for VFX and Animation

**OpenCue**, developed by the Academy Software Foundation, empowers visual effects and animation studios with a powerful and scalable render management solution.  ([View the Original Repo](https://github.com/AcademySoftwareFoundation/OpenCue))

## Key Features of OpenCue

OpenCue offers a robust set of features designed to streamline and optimize your rendering workflow:

*   **Scalable Architecture:** Supports numerous concurrent machines for handling large-scale rendering jobs.
*   **Production-Proven:**  Based on the in-house render manager used by Sony Pictures Imageworks on hundreds of films.
*   **Resource Allocation:** Tagging systems allow for jobs to be assigned to specific machine types, along with the ability to split a host into procs, each with their own core and memory requirements.
*   **Centralized Rendering:**  Jobs are processed on a central render farm, freeing up artist workstations.
*   **Native Multi-Threading Support:** Works seamlessly with industry-standard renderers like Katana, Prman, and Arnold.
*   **Deployment Flexibility:** Supports multi-facility, on-premises, cloud, and hybrid deployments.
*   **Automated Booking:** Integrated automated booking.
*   **No Limits:** No limit on the number of procs a job can have.

## Getting Started with OpenCue

### Installation and Testing

*   **Sandbox Environment:** Explore a local OpenCue environment with the [OpenCue sandbox documentation](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/sandbox/README.md). This is ideal for testing, development, and learning.  All components run in separate Docker containers or Python virtual environments.
*   **Quick Starts:** Get started quickly with the [OpenCue Quick Starts documentation](https://www.opencue.io/docs/quick-starts/).
*   **Full Installation:** System administrators can find detailed guides for deploying OpenCue components and installing dependencies in the [OpenCue documentation](https://www.opencue.io/docs/getting-started/).

### Documentation and Resources

*   **Comprehensive Documentation:** The official documentation is available at [https://www.opencue.io](https://www.opencue.io), including installation guides, user guides, API references, and tutorials.
*   **YouTube Tutorials:** Learn more through the [OpenCue Playlist](https://www.youtube.com/playlist?list=PL9dZxafYCWmzSBEwVT2AQinmZolYqBzdp) on the Academy Software Foundation (ASWF) YouTube channel.

### Contributing to the Documentation

*   **Build and Test:** Before submitting any pull request, make sure to build and test your documentation changes:
    1.  Build and validate the documentation:
        ```bash
        ./docs/build.sh
        ```
    2.  Install bundler binstubs (if needed):
        ```bash
        cd docs/
        bundle binstubs --all
        ```
    3.  Run the documentation locally:
        ```bash
        cd docs/
        bundle exec jekyll serve --livereload
        ```
    4.  Preview the documentation in your browser: http://localhost:4000
*   **Deployment:** The documentation is automatically deployed via GitHub Actions (.github/workflows/docs.yml) upon merging into the `master` branch.  Updates are available at https://docs.opencue.io/.
*   **Detailed instructions:** Detailed documentation setup instructions, testing procedures, and contribution guidelines are available in [docs/README.md](https://github.com/AcademySoftwareFoundation/OpenCue/blob/master/docs/README.md).

## Meeting Notes

*   Meeting notes are stored on the [OpenCue Confluence page](http://wiki.aswf.io/display/OPENCUE/OpenCue+Home) (starting May 2024).
*   For meeting notes before May 2024, please refer to the [OpenCue repository's meeting notes](https://github.com/AcademySoftwareFoundation/OpenCue/tree/master/tsc/meetings) folder.

## Contributors

[![Contributors](https://contrib.rocks/image?repo=AcademySoftwareFoundation/OpenCue)](https://github.com/AcademySoftwareFoundation/OpenCue/graphs/contributors)

## Contact Us

*   **User Forum:** Join the [opencue-user mailing list](https://lists.aswf.io/g/opencue-user) or email <opencue-user@lists.aswf.io> to discuss OpenCue with users and admins.
*   **Slack Channel:** Connect with the community on the [OpenCue Slack channel](https://academysoftwarefdn.slack.com/archives/CMFPXV39Q).
*   **Working Group:**  Join the bi-weekly Working Group meetings at 2 PM PST on [Zoom](https://www.google.com/url?q=https://zoom-lfx.platform.linuxfoundation.org/meeting/95509555934?password%3Da8d65f0e-c5f0-44fb-b362-d3ed0c22b7c1&sa=D&source=calendar&ust=1717863981078692&usg=AOvVaw1zRcYz7VPAwfwOXeBPpoM6).