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

## PostHog: Open-Source Product Analytics and User Insights Platform

**PostHog empowers product teams to build better products by providing a powerful, open-source platform for product analytics, feature flags, session replays, and more.** ([See the original repository](https://github.com/PostHog/posthog))

**Key Features:**

*   **Product Analytics:** Understand user behavior with event-based analytics, visualizations, and SQL queries.
*   **Web Analytics:** Monitor web traffic, user sessions, conversions, and key web vitals in an intuitive dashboard.
*   **Session Replays:** Watch real user sessions to diagnose issues and understand user interactions.
*   **Feature Flags:** Safely roll out new features to specific user segments with feature flags.
*   **Experiments:** Test changes and measure their impact on key metrics with no-code experiment setup.
*   **Error Tracking:** Track errors, receive alerts, and resolve issues to improve product stability.
*   **Surveys:** Gather user feedback using no-code survey templates or a custom survey builder.
*   **Data Warehouse Integration:** Seamlessly sync data from external tools like Stripe and Hubspot.
*   **Data Pipelines:** Transform and route your data in real time to your preferred destinations.
*   **LLM Analytics:** Track traces, generations, and costs for your AI-powered applications.

**Getting Started:**

*   **PostHog Cloud (Recommended):**  Sign up for a free account at [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup) to get started quickly. Enjoy a generous free tier!
*   **Self-Hosting (Advanced):** Deploy a hobby instance on Linux with Docker using this command:

    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
    ```
    _(Open source deployments scale to approximately 100k events per month. Self-hosting support is limited.)_

**Setting Up PostHog:**

Integrate PostHog into your project using our [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), SDKs, or API. We offer SDKs and libraries for:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

**Learn More:**

*   [Product Documentation](https://posthog.com/docs/product-os)
*   [Winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled) - A guide to activation, retention, and revenue tracking.
*   [Open Source Handbook](https://posthog.com/handbook)

**Contributing:**

We welcome contributions!

*   [Feature Request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [Bug Report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md)
*   Open a Pull Request (see [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally)).
*   Vote on features in our [roadmap](https://posthog.com/roadmap).

**Open-Source vs. Paid:**

This repository is licensed under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), excluding the `ee` directory, which has its own license. For a completely free and open-source version, check out our [posthog-foss](https://github.com/PostHog/posthog-foss) repository.  See our [pricing page](https://posthog.com/pricing) for more details.

**Join Our Team!**

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

We're hiring!  Explore our open positions at [https://posthog.com/careers](https://posthog.com/careers).