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

## PostHog: The Open-Source Product Analytics Platform 🚀

PostHog is an open-source, all-in-one platform designed to help you build successful products with comprehensive product analytics, session replays, feature flags, and more.  **[Explore the PostHog project on GitHub](https://github.com/PostHog/posthog).**

**Key Features:**

*   📊 **Product Analytics:** Understand user behavior with event-based analytics, data visualization, and SQL querying.
*   🌐 **Web Analytics:** Monitor web traffic, user sessions, conversions, web vitals, and revenue.
*   🎬 **Session Replays:** Watch real user sessions to diagnose issues and understand user interactions.
*   🚩 **Feature Flags:** Safely roll out features to specific users or cohorts using feature flags.
*   🧪 **Experiments:** Test changes and measure their impact with no-code experiment setup.
*   🐞 **Error Tracking:** Track errors, receive alerts, and resolve issues to improve product quality.
*   💬 **Surveys:** Gather user feedback with no-code survey templates and a custom survey builder.
*   🔄 **Data Warehouse & Pipelines:** Integrate data from external tools and sync data with your data warehouse.
*   🤖 **LLM Observability:** Track traces, generations, latency, and cost for LLM-powered applications.

PostHog offers a generous free tier, allowing you to get started at no cost. Sign up for [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup) today.

**Table of Contents**

*   [PostHog: The Open-Source Product Analytics Platform](#posthog-the-open-source-product-analytics-platform-)
*   [Key Features](#key-features)
*   [Getting Started with PostHog](#getting-started-with-posthog)
    *   [PostHog Cloud (Recommended)](#posthog-cloud-recommended)
    *   [Self-hosting (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
*   [Setting Up PostHog](#setting-up-posthog)
*   [Learning More About PostHog](#learning-more-about-posthog)
*   [Contributing](#contributing)
*   [Open-source vs. Paid](#open-source-vs-paid)
*   [We're Hiring!](#were-hiring)

## Getting Started with PostHog

### PostHog Cloud (Recommended)

The fastest and most reliable way to get started with PostHog is by signing up for a free PostHog Cloud account.  Get started with [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup). You get 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 250 survey responses free every month.

### Self-hosting (Advanced)

If you prefer to self-host, deploy a hobby instance in one line with Docker:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Self-hosted deployments are suitable for up to 100k events per month; beyond that, consider [migrating to PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud).

Note:  Customer support and guarantees are not provided for self-hosted instances. Refer to the [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer) for further information.

## Setting Up PostHog

Integrate PostHog into your project by installing our [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), using one of our [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks), or via our [API](https://posthog.com/docs/getting-started/install?tab=api).

SDKs and Libraries are available for a wide range of languages and frameworks, including:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Additional documentation and guides are available for [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

After installation, explore the [product docs](https://posthog.com/docs/product-os) for detailed instructions on [product analytics](https://posthog.com/docs/product-analytics/capture-events), [web analytics](https://posthog.com/docs/web-analytics/getting-started), [session replays](https://posthog.com/docs/session-replay/how-to-watch-recordings), [feature flags](https://posthog.com/docs/feature-flags/creating-feature-flags), [experiments](https://posthog.com/docs/experiments/creating-an-experiment), [error tracking](https://posthog.com/docs/error-tracking/installation#setting-up-exception-autocapture), [surveys](https://posthog.com/docs/surveys/installation), [data warehouse](https://posthog.com/docs/cdp/sources), and other features.

## Learning More About PostHog

Learn about PostHog's strategy, culture and processes by viewing the [company handbook](https://posthog.com/handbook).

Discover how to make the most of PostHog with this [guide to winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled), covering topics such as [measuring activation](https://posthog.com/docs/new-to-posthog/activation), [tracking retention](https://posthog.com/docs/new-to-posthog/retention), and [capturing revenue](https://posthog.com/docs/new-to-posthog/revenue).

## Contributing

We welcome contributions!

-   Vote on features or get early access to beta functionality in our [roadmap](https://posthog.com/roadmap)
-   Open a PR (see our instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally))
-   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md)

## Open-source vs. Paid

This repository uses the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory (see [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE) if applicable).

For a completely FOSS solution, explore the [posthog-foss](https://github.com/PostHog/posthog-foss) repository.

Find complete transparency on our paid plan pricing on the [pricing page](https://posthog.com/pricing).

## We’re hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

If you're reading this, we are looking for enthusiastic people to join our team. We are growing fast and [would love for you to join us](https://posthog.com/careers).