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

PostHog is an open-source, all-in-one platform designed to help you build and scale successful products, offering a suite of tools for product analytics, user behavior analysis, and much more.  [Explore the source code on GitHub](https://github.com/PostHog/posthog).

**Key Features:**

*   **Product Analytics:**  Gain in-depth insights into user behavior with event-based analytics, including autocapture and manual instrumentation. Analyze data using visualizations or SQL.
*   **Web Analytics:** Monitor website traffic, conversions, web vitals, and revenue with a GA-like dashboard.
*   **Session Replays:** Understand user interactions by watching real user sessions on your website or mobile app.
*   **Feature Flags:**  Safely roll out new features to specific user groups or cohorts.
*   **Experiments:** Run A/B tests to measure the impact of your changes on key metrics, with a no-code setup option.
*   **Error Tracking:** Track errors, receive alerts, and resolve issues quickly to improve product quality.
*   **Surveys:** Gather user feedback using pre-built templates or create custom surveys with our survey builder.
*   **Data Warehouse:** Integrate data from external sources like Stripe and HubSpot. Query it alongside your product data.
*   **Data Pipelines:** Transform and route your data to various tools and warehouses in real-time or in batches.
*   **LLM Observability:** Monitor traces, generations, latency, and cost for your LLM-powered applications.

**Getting Started:**

Choose the best option for you:

*   **PostHog Cloud (Recommended):**  Sign up for a free account at [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup) and get started instantly. The free tier includes 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 250 survey responses each month.
*   **Self-Hosting (Advanced):** Deploy a hobby instance on Linux with Docker using the following command (requires 4GB memory):

    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
    ```
    *Note: Self-hosted deployments are not guaranteed, and customer support isn't offered. See the [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer) for more information.*

**Setting Up PostHog:**

Integrate PostHog into your product by:

*   Installing the [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet)
*   Using one of our [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks)
*   Utilizing the [API](https://posthog.com/docs/getting-started/install?tab=api)

SDKs are available for popular languages and frameworks:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Find more setup guides for other languages and frameworks in our documentation. Once installed, explore product features through our [product docs](https://posthog.com/docs/product-os).

**Learn More:**

*   Explore our [company handbook](https://posthog.com/handbook).
*   Get started with our guide to [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled).

**Contributing:**

We welcome contributions!

*   Vote on features or get early access on our [roadmap](https://posthog.com/roadmap).
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md).
*   Open a Pull Request (see instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally)).

**Open Source vs. Paid:**

This project is licensed under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), excluding the `ee` directory. For a 100% free and open-source version, check out our [posthog-foss](https://github.com/PostHog/posthog-foss) repository. Transparent pricing for paid plans is available on [our pricing page](https://posthog.com/pricing).

---

**Join Our Team!**

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

We are rapidly growing and hiring!  [Check out our careers page](https://posthog.com/careers) to learn about open opportunities.