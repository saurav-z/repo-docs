<p align="center">
  <img alt="PostHog Logo" src="https://user-images.githubusercontent.com/65415371/205059737-c8a4f836-4889-4654-902e-f302b16a0.png">
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

## PostHog: The Open-Source Product Analytics Platform for Building Successful Products

[PostHog](https://posthog.com/) is an open-source product analytics platform, providing a comprehensive suite of tools to help you understand, analyze, and improve your product. **[Explore the PostHog repository on GitHub](https://github.com/PostHog/posthog) and revolutionize how you build and grow your product.**

**Key Features:**

*   **Product Analytics:** Understand user behavior with event-based analytics, data visualization, and SQL querying.
*   **Web Analytics:** Monitor web traffic, user sessions, conversion rates, web vitals, and revenue with a GA-like dashboard.
*   **Session Replays:** Watch real user sessions to diagnose issues and understand user behavior.
*   **Feature Flags:** Safely roll out features with feature flags to select users or cohorts.
*   **Experiments:** A/B test changes and measure their impact on goal metrics, including no-code options.
*   **Error Tracking:** Track, get alerts, and resolve errors to improve your product.
*   **Surveys:** Collect feedback with no-code survey templates and custom survey builders.
*   **Data Warehouse:** Sync data from external tools (Stripe, Hubspot, etc.) and query it alongside product data.
*   **Data Pipelines:** Run custom filters and transformations on incoming data and send it to 25+ tools or webhooks.
*   **LLM Analytics:** Capture traces, generations, latency, and cost for LLM-powered applications.

**Getting Started**

PostHog offers flexible deployment options to suit your needs.

*   **PostHog Cloud (Recommended):** The easiest way to get started! Sign up for a free account at [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup) and enjoy a generous free tier.
*   **Self-hosting (Advanced):** Deploy a hobby instance using Docker on Linux with a single command:
    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
    ```
    Self-hosted deployments are recommended for up to 100k events per month. See our [self-hosting docs](https://posthog.com/docs/self-host) for more information.

**Setup and Integration**

Integrate PostHog into your product by using our:

*   [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet)
*   [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks)
*   [API](https://posthog.com/docs/getting-started/install?tab=api).

SDKs are available for a wide range of languages and frameworks, including:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Detailed setup guides are available for [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

**Learn More**

*   [Product Docs](https://posthog.com/docs/product-os): Comprehensive documentation for all features.
*   [Company Handbook](https://posthog.com/handbook): Learn about our strategy, culture, and processes.
*   [Winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled): A guide to maximizing your PostHog usage.

**Contribute**

We welcome contributions of all sizes:

*   [Roadmap](https://posthog.com/roadmap): Vote on features.
*   [Pull Requests](https://github.com/PostHog/posthog/pulls): Open a PR (see our [local development guide](https://posthog.com/handbook/engineering/developing-locally)).
*   [Feature Requests](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md)
*   [Bug Reports](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md)

**Licensing**

This repository is licensed under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory which has its [own license](https://github.com/PostHog/posthog/blob/master/ee/LICENSE).  For a fully free and open source version without proprietary features, see our [posthog-foss](https://github.com/PostHog/posthog-foss) repository.

**We're Hiring!**

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

Join our growing team! Explore [PostHog careers](https://posthog.com/careers).