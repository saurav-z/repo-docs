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

## PostHog: The Open-Source Product Analytics Platform for Building Successful Products

**PostHog** is an all-in-one, open-source platform empowering product teams with comprehensive analytics and user behavior insights, offering a suite of tools to build successful products.  [Explore the original repository](https://github.com/PostHog/posthog).

**Key Features:**

*   **Product Analytics:** Understand user behavior with event-based analytics, including autocapture and custom event tracking, and analyze data through visualizations or SQL.
*   **Web Analytics:** Monitor website traffic, track key metrics, and analyze user sessions with a GA-like dashboard to easily monitor conversions, web vitals, and revenue.
*   **Session Replays:** Watch real user sessions to diagnose issues, understand user behavior, and improve the user experience.
*   **Feature Flags:** Safely roll out new features to specific user cohorts or A/B test changes, and instantly kill a feature if it's buggy.
*   **Experiments:** Run A/B tests and measure the statistical impact of your changes on your goal metrics.
*   **Error Tracking:** Capture and resolve errors to improve your product's stability and user experience.
*   **Surveys:** Gather user feedback with no-code survey templates or create custom surveys.
*   **Data Warehouse:** Integrate data from external tools and query it alongside your product data.
*   **Data Pipelines:** Run custom filters and transformations on your incoming data, and send it to other tools.
*   **LLM Observability:** Capture traces, generations, latency, and cost for your LLM-powered app.

**Free Tier:** PostHog offers a generous free tier for all products.  [Sign up today](https://us.posthog.com/signup) or [EU](https://eu.posthog.com/signup).

## Table of Contents

-   [PostHog: The Open-Source Product Analytics Platform for Building Successful Products](#posthog-the-open-source-product-analytics-platform-for-building-successful-products)
    -   [Key Features](#key-features)
-   [Getting Started](#getting-started)
    -   [PostHog Cloud (Recommended)](#posthog-cloud-recommended)
    -   [Self-hosting the open-source hobby deploy (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
-   [Setting up PostHog](#setting-up-posthog)
-   [Learning More about PostHog](#learning-more-about-posthog)
-   [Contributing](#contributing)
-   [Open-source vs. paid](#open-source-vs-paid)
-   [We’re hiring!](#were-hiring)

## Getting Started

### PostHog Cloud (Recommended)

Get started quickly and reliably by signing up for free to [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup). Enjoy a generous free tier (1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 250 survey responses free per month).

### Self-hosting the open-source hobby deploy (Advanced)

Deploy a hobby instance with Docker (recommended 4GB memory) with this one-liner on Linux:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Open-source deployments are recommended for approximately 100k events per month, after which migrating to [PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud) is recommended.

_We do not provide customer support or guarantees for open-source deployments._  Consult our [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer) for more information.

## Setting up PostHog

Integrate PostHog into your product using our [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), one of [our SDKs](https://posthog.com/docs/getting-started/install?tab=sdks), or [our API](https://posthog.com/docs/getting-started/install?tab=api).

SDKs and libraries are available for:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Additional resources: [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

Explore our [product docs](https://posthog.com/docs/product-os) for detailed setup information: [product analytics](https://posthog.com/docs/product-analytics/capture-events), [web analytics](https://posthog.com/docs/web-analytics/getting-started), [session replays](https://posthog.com/docs/session-replay/how-to-watch-recordings), [feature flags](https://posthog.com/docs/feature-flags/creating-feature-flags), [experiments](https://posthog.com/docs/experiments/creating-an-experiment), [error tracking](https://posthog.com/docs/error-tracking/installation#setting-up-exception-autocapture), [surveys](https://posthog.com/docs/surveys/installation), [data warehouse](https://posthog.com/docs/cdp/sources), and more.

## Learning More about PostHog

Our company transparency extends to our [company handbook](https://posthog.com/handbook), which includes our [strategy](https://posthog.com/handbook/why-does-posthog-exist), [ways of working](https://posthog.com/handbook/company/culture), and [processes](https://posthog.com/handbook/team-structure).

For helpful insights, read our guide on [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled).

## Contributing

We welcome contributions of all kinds:

*   Vote on features and get early access on our [roadmap](https://posthog.com/roadmap)
*   Submit a PR. [Develop locally](https://posthog.com/handbook/engineering/developing-locally)
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md).

## Open-source vs. paid

This repository is available under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory (which has its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)).

Need 100% FOSS? Check out our [posthog-foss](https://github.com/PostHog/posthog-foss) repository.

[Our pricing page](https://posthog.com/pricing) provides complete transparency.

## We’re hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

Join our growing team!  We'd love for you to apply [here](https://posthog.com/careers).