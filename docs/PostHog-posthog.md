<p align="center">
  <img alt="PostHog Logo" src="https://user-images.githubusercontent.com/65415371/205059737-c8a4f836-4889-4654-902e-f302b1600.png">
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

## PostHog: The Open-Source Platform to Build Successful Products

PostHog is an open-source product analytics platform packed with features that empower you to understand your users and build better products.  Explore [the original repository](https://github.com/PostHog/posthog).

**Key Features:**

*   **Product Analytics:** Gain deep insights into user behavior with event-based analytics, visualizations, and SQL querying.
*   **Web Analytics:** Monitor website traffic, user sessions, conversions, web vitals, and revenue in a GA-like dashboard.
*   **Session Replays:**  Watch recordings of real user sessions to diagnose issues and understand user interactions.
*   **Feature Flags:**  Safely roll out new features to specific users or cohorts using feature flags.
*   **Experiments (A/B Testing):** Test changes and measure their statistical impact on key metrics. Set up A/B tests without any code.
*   **Error Tracking:** Track errors, receive alerts, and resolve issues rapidly to improve product quality.
*   **Surveys:** Gather user feedback with no-code survey templates or build custom surveys to understand your audience.
*   **Data Warehouse:** Sync data from Stripe, Hubspot, and other sources alongside your product data for comprehensive analysis.
*   **Data Pipelines (CDP):** Run custom filters and transformations on your incoming data and send it to over 25 tools in real time or batch export to your data warehouse.
*   **LLM Analytics:** Capture traces, generations, latency, and costs for your LLM-powered applications.

**Get Started:**  Enjoy a [generous monthly free tier](https://posthog.com/pricing) to experience the power of PostHog! Sign up for [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup) today.

## Table of Contents

*   [PostHog: The Open-Source Platform to Build Successful Products](#posthog-the-open-source-platform-to-build-successful-products)
*   [Table of Contents](#table-of-contents)
*   [Getting started with PostHog](#getting-started-with-posthog)
    *   [PostHog Cloud (Recommended)](#posthog-cloud-recommended)
    *   [Self-hosting the open-source hobby deploy (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
*   [Setting up PostHog](#setting-up-posthog)
*   [Learning more about PostHog](#learning-more-about-posthog)
*   [Contributing](#contributing)
*   [Open-source vs. paid](#open-source-vs-paid)
*   [We’re hiring!](#were-hiring)

## Getting started with PostHog

### PostHog Cloud (Recommended)

The quickest and most reliable way to start using PostHog is by signing up for free at [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup). Enjoy free usage for your first 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 1500 survey responses per month. Paid plans are available for higher usage levels.

### Self-hosting the open-source hobby deploy (Advanced)

If you'd like to self-host PostHog, deploy a hobby instance in one line on Linux with Docker (4GB memory recommended):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Open source deployments are typically suited for up to 100k events per month. For larger volumes, we recommend [migrating to a PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud).

We don't provide customer support or guarantees for open source deployments. See our [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer) for more info.

## Setting up PostHog

Once you have a PostHog instance, integrate it by installing our [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), one of [our SDKs](https://posthog.com/docs/getting-started/install?tab=sdks), or by [using our API](https://posthog.com/docs/getting-started/install?tab=api).

We offer SDKs and libraries for a wide range of languages and frameworks:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

We also have extensive documentation and guides for [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and many more.

After installation, learn how to fully leverage PostHog with our [product docs](https://posthog.com/docs/product-os) covering [product analytics](https://posthog.com/docs/product-analytics/capture-events), [web analytics](https://posthog.com/docs/web-analytics/getting-started), [session replays](https://posthog.com/docs/session-replay/how-to-watch-recordings), [feature flags](https://posthog.com/docs/feature-flags/creating-feature-flags), [experiments](https://posthog.com/docs/experiments/creating-an-experiment), [error tracking](https://posthog.com/docs/error-tracking/installation#setting-up-exception-autocapture), [surveys](https://posthog.com/docs/surveys/installation), [data warehouse](https://posthog.com/docs/cdp/sources), and more.

## Learning more about PostHog

Our commitment to open-source extends beyond code.  Explore our [company handbook](https://posthog.com/handbook) to learn about our [strategy](https://posthog.com/handbook/why-does-posthog-exist), [culture](https://posthog.com/handbook/company/culture), and [processes](https://posthog.com/handbook/team-structure).

Want to become a PostHog power user? Our guide to [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled) will guide you through the basics of [measuring activation](https://posthog.com/docs/new-to-posthog/activation), [tracking retention](https://posthog.com/docs/new-to-posthog/retention), and [capturing revenue](https://posthog.com/docs/new-to-posthog/revenue).

## Contributing

We welcome contributions of all sizes:

*   Vote on features or get early access to beta functionality in our [roadmap](https://posthog.com/roadmap)
*   Submit a pull request (see our instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally))
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md)

## Open-source vs. paid

This repository is licensed under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory (which has its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)) if applicable.

For a completely free and open-source version, explore our [posthog-foss](https://github.com/PostHog/posthog-foss) repository, which excludes all proprietary code and features.

Transparency is key! Find our pricing details for paid plans on [our pricing page](https://posthog.com/pricing).

## We’re hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

If you've made it this far, you're a dedicated reader, and we'd love to have you on our team! We're growing fast, and [we're hiring](https://posthog.com/careers)!