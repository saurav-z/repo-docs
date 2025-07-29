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

PostHog is an all-in-one, open-source platform providing a comprehensive suite of tools to help you build successful products.  [Explore the PostHog code on GitHub](https://github.com/PostHog/posthog).

**Key Features:**

*   **Product Analytics:** Understand user behavior through event-based analytics with powerful data visualization and SQL querying.
*   **Web Analytics:** Monitor website traffic, user sessions, conversions, web vitals, and revenue in a GA-like dashboard.
*   **Session Replays:** Diagnose issues and gain user insights by watching real user sessions of your website or mobile app.
*   **Feature Flags:** Safely roll out new features to specific user groups or cohorts using feature flags.
*   **Experiments:** Test changes and measure their impact on key metrics with no-code experiments.
*   **Error Tracking:** Track errors, receive alerts, and resolve issues to improve your product quality.
*   **Surveys:** Gather valuable user feedback using our no-code survey templates or custom survey builder.
*   **Data Warehouse Integration:** Sync data from external tools like Stripe and HubSpot, query it alongside your product data.
*   **Data Pipelines:** Transform and route data to various tools or warehouses in real-time or batch.
*   **LLM Observability:** Capture traces, generations, latency, and cost for your LLM-powered applications.

PostHog offers a generous [free tier](https://posthog.com/pricing), making it easy to get started, and includes all of the above tools.

## Table of Contents

*   [PostHog: The Open-Source Product Analytics Platform for Growth](#posthog-the-open-source-product-analytics-platform-for-growth)
*   [Table of Contents](#table-of-contents)
*   [Getting Started with PostHog](#getting-started-with-posthog)
    *   [PostHog Cloud (Recommended)](#posthog-cloud-recommended)
    *   [Self-hosting (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
*   [Setting Up PostHog](#setting-up-posthog)
*   [Learning More About PostHog](#learning-more-about-posthog)
*   [Contributing](#contributing)
*   [Open-source vs. Paid](#open-source-vs-paid)
*   [We’re Hiring!](#were-hiring)

## Getting Started with PostHog

### PostHog Cloud (Recommended)

Sign up for a free account on [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup) to get started quickly and reliably. Enjoy a free tier including 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 250 survey responses per month.

### Self-hosting (Advanced)

You can self-host a hobby instance of PostHog using Docker on Linux with a simple one-line command (requires 4GB memory).

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Open-source deployments are designed to handle approximately 100k events per month.  We recommend migrating to [PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud) as your data needs grow.

For open-source deployments, customer support is not provided. Consult the [self-hosting documentation](https://posthog.com/docs/self-host), the [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and the [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer) for more information.

## Setting Up PostHog

Once you have your PostHog instance up and running, install the [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), one of [our SDKs](https://posthog.com/docs/getting-started/install?tab=sdks), or use our [API](https://posthog.com/docs/getting-started/install?tab=api).

We offer SDKs and libraries for the following languages and frameworks:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Additional documentation and guides are available for [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), and [Webflow](https://posthog.com/docs/libraries/webflow).

Refer to the [product docs](https://posthog.com/docs/product-os) for comprehensive information on setting up [product analytics](https://posthog.com/docs/product-analytics/capture-events), [web analytics](https://posthog.com/docs/web-analytics/getting-started), [session replays](https://posthog.com/docs/session-replay/how-to-watch-recordings), [feature flags](https://posthog.com/docs/feature-flags/creating-feature-flags), [experiments](https://posthog.com/docs/experiments/creating-an-experiment), [error tracking](https://posthog.com/docs/error-tracking/installation#setting-up-exception-autocapture), [surveys](https://posthog.com/docs/surveys/installation), and [data warehouse](https://posthog.com/docs/cdp/sources).

## Learning More About PostHog

Explore our open-source [company handbook](https://posthog.com/handbook) to learn about our [strategy](https://posthog.com/handbook/why-does-posthog-exist), [company culture](https://posthog.com/handbook/company/culture), and [work processes](https://posthog.com/handbook/team-structure).

For guidance on maximizing PostHog's potential, consult our guide to [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled) which covers the basics of [measuring activation](https://posthog.com/docs/new-to-posthog/activation), [tracking retention](https://posthog.com/docs/new-to-posthog/retention), and [capturing revenue](https://posthog.com/docs/new-to-posthog/revenue).

## Contributing

We welcome contributions of all sizes:

*   Vote on features or get early access to beta functionality in our [roadmap](https://posthog.com/roadmap)
*   Submit a pull request (PR) – learn how to [develop PostHog locally](https://posthog.com/handbook/engineering/developing-locally).
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md).

## Open-source vs. Paid

This repository is licensed under the [MIT Expat License](https://github.com/PostHog/posthog/blob/master/LICENSE), with the exception of the `ee` directory, which has its own [license](https://github.com/PostHog/posthog/blob/master/ee/LICENSE).

If you need a 100% FOSS solution, explore our [posthog-foss](https://github.com/PostHog/posthog-foss) repository.

View our transparent [pricing page](https://posthog.com/pricing) for details on our paid plans.

## We’re Hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

If you've made it this far, consider joining our team! We are growing quickly and would love for you to join us - find our [open positions](https://posthog.com/careers).