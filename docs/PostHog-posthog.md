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

## PostHog: The Open-Source Platform for Product Success

PostHog is an open-source product analytics platform that provides all the tools you need to build successful products, offering a complete solution for product development and user understanding. ([View the original repository](https://github.com/PostHog/posthog))

**Key Features:**

*   **Product Analytics:** Track user behavior with autocapture or manual instrumentation, analyze data with visualizations or SQL.
*   **Web Analytics:** Monitor web traffic, user sessions, conversion, web vitals, and revenue in a GA-like dashboard.
*   **Session Replays:** Watch real user sessions to diagnose issues and understand user behavior.
*   **Feature Flags:** Safely roll out features to specific users or cohorts.
*   **Experiments:** Test changes and measure their impact on key metrics with no-code tools.
*   **Error Tracking:** Identify and resolve issues with error tracking and alerts.
*   **Surveys:** Gather user feedback using no-code survey templates or a custom builder.
*   **Data Warehouse:** Sync data from external tools like Stripe, Hubspot, or your data warehouse and query it alongside product data.
*   **Data Pipelines:** Transform incoming data and send it to 25+ tools or any webhook.
*   **LLM Observability:** Capture traces, generations, latency, and cost for your LLM-powered app.

PostHog offers a generous free tier; get started with [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup).

## Table of Contents

*   [PostHog: The Open-Source Platform for Product Success](#posthog-the-open-source-platform-for-product-success)
*   [Key Features](#key-features)
*   [Table of Contents](#table-of-contents)
*   [Getting Started with PostHog](#getting-started-with-posthog)
    *   [PostHog Cloud (Recommended)](#posthog-cloud-recommended)
    *   [Self-hosting the open-source hobby deploy (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
*   [Setting up PostHog](#setting-up-posthog)
*   [Learning more about PostHog](#learning-more-about-posthog)
*   [Contributing](#contributing)
*   [Open-source vs. paid](#open-source-vs-paid)
*   [We’re hiring!](#were-hiring)

## Getting Started with PostHog

### PostHog Cloud (Recommended)

Sign up for free to [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup) for the easiest and most reliable start. Enjoy 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 250 survey responses free monthly.

### Self-hosting the open-source hobby deploy (Advanced)

Deploy a hobby instance on Linux with Docker using:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Self-hosted deployments can handle roughly 100k events per month. Consider migrating to [PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud) for higher volumes. Note that customer support and guarantees are not provided for self-hosted open-source deployments. See the [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer).

## Setting up PostHog

After creating a PostHog instance, install the [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), one of our [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks), or use the [API](https://posthog.com/docs/getting-started/install?tab=api).

Available SDKs and libraries:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Additional resources include documentation and guides for [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

Consult our [product docs](https://posthog.com/docs/product-os) for setup instructions for [product analytics](https://posthog.com/docs/product-analytics/capture-events), [web analytics](https://posthog.com/docs/web-analytics/getting-started), [session replays](https://posthog.com/docs/session-replay/how-to-watch-recordings), [feature flags](https://posthog.com/docs/feature-flags/creating-feature-flags), [experiments](https://posthog.com/docs/experiments/creating-an-experiment), [error tracking](https://posthog.com/docs/error-tracking/installation#setting-up-exception-autocapture), [surveys](https://posthog.com/docs/surveys/installation), [data warehouse](https://posthog.com/docs/cdp/sources), and more.

## Learning more about PostHog

Explore our open-source [company handbook](https://posthog.com/handbook) for insights into our [strategy](https://posthog.com/handbook/why-does-posthog-exist), [culture](https://posthog.com/handbook/company/culture), and [processes](https://posthog.com/handbook/team-structure).

For guidance, check out the guide to [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled), covering [measuring activation](https://posthog.com/docs/new-to-posthog/activation), [tracking retention](https://posthog.com/docs/new-to-posthog/retention), and [capturing revenue](https://posthog.com/docs/new-to-posthog/revenue).

## Contributing

Contribute to PostHog:

*   Vote on features or get early access in our [roadmap](https://posthog.com/roadmap).
*   Submit a PR following instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally).
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md).

## Open-source vs. paid

This repository is licensed under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory, which has its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE).

For a fully FOSS version, explore [posthog-foss](https://github.com/PostHog/posthog-foss).

View our transparent pricing on [our pricing page](https://posthog.com/pricing).

## We’re hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

Join our fast-growing team! Check out [our careers](https://posthog.com/careers).