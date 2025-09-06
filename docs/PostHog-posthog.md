<p align="center">
  <img alt="PostHog Logo" src="https://user-images.githubusercontent.com/65415371/205059737-c8a4f836-4889-4654-902e-f302b160.png">
</p>

<p align="center">
  <a href='https://posthog.com/contributors'><img alt="GitHub contributors" src="https://img.shields.io/github/contributors/posthog/posthog"/></a>
  <a href='http://makeapullrequest.com'><img alt='PRs Welcome' src='https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=shields'/></a>
  <img alt="Docker Pulls" src="https://img.shields.io/docker/pulls/posthog/posthog"/>
  <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/posthog/posthog"/>
  <img alt="GitHub closed issues" src="https://img.shields.io/github/issues-closed/posthog/posthog"/>
</p>

<p align="center">
  <a href="https://posthog.com/docs">Docs</a> - <a href="https://posthog.com/community">Community</a> - <a href="https://posthog.com/roadmap">Roadmap</a> - <a href="https://posthog.com/why">Why PostHog?</a> - <a href="https://posthog.com/changelog">Changelog</a> - <a href="https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md">Bug reports</a>
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=2jQco8hEvTI">
    <img src="https://res.cloudinary.com/dmukukwp6/image/upload/demo_thumb_68d0d8d56d" alt="PostHog Demonstration">
  </a>
</p>

## PostHog: The Open-Source Product Analytics Platform for Growth 🚀

PostHog is an all-in-one, open-source product analytics platform designed to empower product teams to build better products and understand user behavior.

**[Explore the PostHog Repository](https://github.com/PostHog/posthog)**

**Key Features:**

*   **Product Analytics:** Understand user behavior with event-based analytics, autocapture, and SQL queries.
*   **Web Analytics:** Monitor web traffic, user sessions, conversion rates, and web vitals with a GA-like dashboard.
*   **Session Replays:** Watch recordings of real user sessions to diagnose issues and understand user interactions.
*   **Feature Flags:**  Safely roll out features to select users or cohorts.
*   **Experiments:**  Test changes and measure their impact using no-code experiment setup.
*   **Error Tracking:** Track and resolve errors to improve your product.
*   **Surveys:** Gather user feedback with no-code survey templates.
*   **Data Warehouse:** Sync data from external tools and query it alongside your product data.
*   **Data Pipelines (CDP):** Transform data, send it to tools in real time, or batch export it to your warehouse.
*   **LLM Analytics:**  Analyze traces, generations, latency, and cost for LLM-powered apps.

**Get Started with PostHog:**

*   **PostHog Cloud (Recommended):** Sign up for free at [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup).
*   **Self-hosting (Advanced):** Deploy a hobby instance using Docker (requires 4GB memory) with the following command:
    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
    ```
    *   For more info, see our [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer).

**Setting up PostHog:**

Integrate PostHog using:
*   [JavaScript Web Snippet](https://posthog.com/docs/getting-started/install?tab=snippet)
*   [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks)
*   [API](https://posthog.com/docs/getting-started/install?tab=api)

**SDKs and Libraries:**

PostHog supports many popular languages and frameworks:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

And more! See the documentation for [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

**Learn More:**

*   [Company Handbook](https://posthog.com/handbook) (strategy, culture, processes)
*   [Winning with PostHog Guide](https://posthog.com/docs/new-to-posthog/getting-hogpilled)

**Contribute:**

*   [Roadmap](https://posthog.com/roadmap) (vote on features)
*   Open a PR (see instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally))
*   [Feature Request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md)
*   [Bug Report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md)

**Open-source vs. Paid:**

*   This repository is licensed under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory (see its [license](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)).
*   For a truly 100% FOSS version, see [posthog-foss](https://github.com/PostHog/posthog-foss).
*   See [pricing](https://posthog.com/pricing).

**We're Hiring!**

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

Join our growing team! Check out our [careers](https://posthog.com/careers).