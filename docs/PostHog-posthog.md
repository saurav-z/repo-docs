<p align="center">
  <img alt="PostHog Logo" src="https://user-images.githubusercontent.com/65415371/205059737-c8a4f836-4889-4654-902e-f302b187b6a0.png">
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

## PostHog: The Open-Source Product Analytics Platform for Growth

PostHog is a powerful, open-source platform that provides a complete suite of tools for product analytics, helping you understand your users and build successful products.  **[Explore the source code on GitHub!](https://github.com/PostHog/posthog)**

**Key Features:**

*   **Product Analytics:** Understand user behavior with event-based analytics, including autocapture and manual instrumentation.
*   **Web Analytics:** Monitor web traffic, user sessions, conversion, web vitals, and revenue.
*   **Session Replays:** Watch recordings of real user sessions to diagnose issues and understand user interactions.
*   **Feature Flags:** Safely roll out new features to specific user segments.
*   **Experiments:**  A/B test changes and measure their impact with no-code setup.
*   **Error Tracking:** Track errors, receive alerts, and resolve issues to improve your product.
*   **Surveys:** Gather user feedback with no-code survey templates or custom builders.
*   **Data Warehouse:** Integrate data from external tools and query it alongside your product data.
*   **Data Pipelines:** Transform and route incoming data to 25+ tools or your data warehouse.
*   **LLM Observability:** Monitor traces, generations, latency, and costs for LLM-powered applications.

**Get Started Today!**

*   **PostHog Cloud (Recommended):** Sign up for a free account with a generous monthly free tier.
    *   [PostHog Cloud US](https://us.posthog.com/signup)
    *   [PostHog Cloud EU](https://eu.posthog.com/signup)

*   **Self-Hosting (Advanced):** Deploy a hobby instance with Docker (4GB memory recommended):
    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
    ```
    *   See [Self-Hosting Documentation](https://posthog.com/docs/self-host) for more information.

## Setting Up PostHog

Integrate PostHog into your project using:

*   [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet)
*   One of our [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks)
*   [Our API](https://posthog.com/docs/getting-started/install?tab=api)

SDKs and libraries are available for a wide range of languages and frameworks, including:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

See the [Product Documentation](https://posthog.com/docs/product-os) for detailed instructions on each feature.

## Learning More

*   [Company Handbook](https://posthog.com/handbook): Learn about our strategy, culture, and processes.
*   [Winning with PostHog Guide](https://posthog.com/docs/new-to-posthog/getting-hogpilled): Learn how to measure activation, retention, and revenue.

## Contributing

We welcome contributions of all sizes!

*   [Roadmap](https://posthog.com/roadmap): Vote on features and access beta functionality.
*   Submit a [Pull Request](https://posthog.com/handbook/engineering/developing-locally).
*   Submit a [Feature Request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [Bug Report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md).

## Open-Source vs. Paid

This repository is available under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory (which has its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)) if applicable.

For a fully open-source version without any proprietary code, check out the [posthog-foss](https://github.com/PostHog/posthog-foss) repository.

See [pricing page](https://posthog.com/pricing) for paid plan details.

## We're Hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

Join our growing team!  [Apply to join us!](https://posthog.com/careers)