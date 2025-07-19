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

## PostHog: The Open Source Platform for Product-Led Growth

**PostHog is a comprehensive, open-source platform designed to empower your product's success by providing a suite of tools for product analytics, user behavior analysis, and more.** [Explore the PostHog repository](https://github.com/PostHog/posthog) to build and grow your product.

**Key Features:**

*   **Product Analytics:** Understand user behavior with event-based analytics, data visualization, and SQL queries.
*   **Web Analytics:** Monitor website traffic, user sessions, conversions, and key metrics like web vitals and revenue.
*   **Session Replays:** Watch recordings of user interactions to diagnose issues and gain insights into user behavior.
*   **Feature Flags:** Safely roll out new features to specific user segments using feature flags.
*   **Experiments:** Conduct A/B tests and measure the impact of changes on your goals without code.
*   **Error Tracking:** Capture and resolve errors to improve product stability.
*   **Surveys:** Gather user feedback with customizable surveys and no-code templates.
*   **Data Warehouse Integration:** Sync data from external tools for comprehensive analysis.
*   **Data Pipelines:** Transform and route your data in real time or batch export it to your data warehouse.
*   **LLM Observability:** Gain insights into traces, generations, latency, and costs of LLM-powered applications.

Get started for free with a generous monthly free tier: [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup).

### Table of Contents

-   [PostHog: The Open Source Platform for Product-Led Growth](#posthog-the-open-source-platform-for-product-led-growth)
    -   [Key Features](#key-features)
    -   [Table of Contents](#table-of-contents)
    -   [Getting Started with PostHog](#getting-started-with-posthog)
        -   [PostHog Cloud (Recommended)](#posthog-cloud-recommended)
        -   [Self-hosting (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
    -   [Setting Up PostHog](#setting-up-posthog)
    -   [Learning More About PostHog](#learning-more-about-posthog)
    -   [Contributing](#contributing)
    -   [Open-source vs. Paid](#open-source-vs-paid)
    -   [We’re Hiring!](#were-hiring)

### Getting Started with PostHog

#### PostHog Cloud (Recommended)

Sign up for a free [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup) account to quickly get started. Your first 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 250 survey responses are free monthly.

#### Self-hosting (Advanced)

You can self-host a hobby instance of PostHog with Docker:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Open source deployments scale to approximately 100k events per month. We recommend [migrating to a PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud) after that.

_We do not_ provide customer support or offer guarantees for open source deployments. See our [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer) for more info.

### Setting Up PostHog

Integrate PostHog into your product with:

*   [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet)
*   One of [our SDKs](https://posthog.com/docs/getting-started/install?tab=sdks)
*   [Our API](https://posthog.com/docs/getting-started/install?tab=api)

SDKs and libraries available:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Additional documentation is provided for [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

Once installed, explore product docs for product analytics, web analytics, session replays, feature flags, experiments, error tracking, surveys, data warehouse, and more.

### Learning More About PostHog

Explore our [company handbook](https://posthog.com/handbook) for details on our strategy, ways of working, and processes.

To make the most of PostHog, check out our guide: [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled). Learn about measuring activation, tracking retention, and capturing revenue.

### Contributing

Join our community:

-   Suggest and vote on features in our [roadmap](https://posthog.com/roadmap)
-   Contribute code by opening a PR (see [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally))
-   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md)

### Open-source vs. Paid

This repo is available under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory (which has its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)) if applicable.

Need 100% FOSS? Check out our [posthog-foss](https://github.com/PostHog/posthog-foss) repository.

View pricing information on our [pricing page](https://posthog.com/pricing).

### We’re Hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

Join our team, explore current opportunities at [PostHog Careers](https://posthog.com/careers).