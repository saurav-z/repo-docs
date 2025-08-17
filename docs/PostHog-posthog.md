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

## PostHog: Build Better Products with Open-Source Product Analytics

**PostHog is an open-source product analytics platform that gives you the tools to build successful products.**  Analyze user behavior, track key metrics, and make data-driven decisions with ease – all while retaining control of your data.

**[Explore the PostHog Repository on GitHub](https://github.com/PostHog/posthog)**

**Key Features:**

*   **Product Analytics:** Understand user behavior with event-based analytics, visualizations, and SQL queries.
*   **Web Analytics:** Monitor website traffic, track conversions, and analyze web vitals.
*   **Session Replays:** Watch user sessions to diagnose issues and understand user interactions.
*   **Feature Flags:** Release new features safely to targeted user cohorts.
*   **Experiments:**  Test changes and measure their statistical impact.
*   **Error Tracking:** Identify and resolve issues quickly to improve product quality.
*   **Surveys:**  Gather user feedback with no-code survey templates.
*   **Data Warehouse & Pipelines:** Integrate with external tools, transform data, and send it to various destinations.
*   **LLM Observability:** Track traces, generations, and costs for LLM-powered applications.

PostHog offers a generous [free tier](https://posthog.com/pricing) and is available as a cloud-hosted service or can be self-hosted.

## Table of Contents

-   [PostHog: Build Better Products with Open-Source Product Analytics](#posthog-build-better-products-with-open-source-product-analytics)
-   [Table of Contents](#table-of-contents)
-   [Getting Started with PostHog](#getting-started-with-posthog)
    -   [PostHog Cloud (Recommended)](#posthog-cloud-recommended)
    -   [Self-hosting the open-source hobby deploy (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
-   [Setting up PostHog](#setting-up-posthog)
-   [Learning More About PostHog](#learning-more-about-posthog)
-   [Contributing](#contributing)
-   [Open-Source vs. Paid](#open-source-vs-paid)
-   [We’re Hiring!](#were-hiring)

## Getting Started with PostHog

### PostHog Cloud (Recommended)

The easiest and most reliable way to get started is by signing up for a free account at [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup). Enjoy a free tier for up to 1 million events monthly.

### Self-hosting the open-source hobby deploy (Advanced)

For self-hosting, deploy a hobby instance using Docker (4GB memory recommended):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Self-hosted deployments are suitable for approximately 100k events per month.  For higher volumes, we recommend [migrating to PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud). Please refer to the [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer) for more info.

## Setting up PostHog

Integrate PostHog with your project using the [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), one of our [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks), or our [API](https://posthog.com/docs/getting-started/install?tab=api).

We support the following languages and frameworks:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Additional resources are available for [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

After installation, explore the [product docs](https://posthog.com/docs/product-os) for details on setting up [product analytics](https://posthog.com/docs/product-analytics/capture-events), [web analytics](https://posthog.com/docs/web-analytics/getting-started), [session replays](https://posthog.com/docs/session-replay/how-to-watch-recordings), [feature flags](https://posthog.com/docs/feature-flags/creating-feature-flags), [experiments](https://posthog.com/docs/experiments/creating-an-experiment), [error tracking](https://posthog.com/docs/error-tracking/installation#setting-up-exception-autocapture), [surveys](https://posthog.com/docs/surveys/installation), [data warehouse](https://posthog.com/docs/cdp/sources), and more.

## Learning More About PostHog

Discover PostHog's open-source company resources, including our [handbook](https://posthog.com/handbook) detailing our [strategy](https://posthog.com/handbook/why-does-posthog-exist), [ways of working](https://posthog.com/handbook/company/culture), and [processes](https://posthog.com/handbook/team-structure).

For actionable insights, check out our guide to [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled) for [measuring activation](https://posthog.com/docs/new-to-posthog/activation), [tracking retention](https://posthog.com/docs/new-to-posthog/retention), and [capturing revenue](https://posthog.com/docs/new-to-posthog/revenue).

## Contributing

We welcome contributions of all sizes:

-   Vote on features or get early access to beta functionality in our [roadmap](https://posthog.com/roadmap)
-   Submit a Pull Request (PR) - see our [local development instructions](https://posthog.com/handbook/engineering/developing-locally).
-   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md).

## Open-Source vs. Paid

This repository uses the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory (which has its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)).

For a fully free and open-source version, explore our [posthog-foss](https://github.com/PostHog/posthog-foss) repository, which excludes all proprietary code.

Review our [pricing page](https://posthog.com/pricing) for transparent details on our paid plans.

## We’re Hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

Interested in joining our team? If you enjoy reading documentation, you might be a great fit! [Explore our careers page](https://posthog.com/careers).