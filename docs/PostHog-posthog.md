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

## PostHog: Open Source Product Analytics for Growth

**PostHog is your all-in-one, open-source platform to understand and grow your product by combining product analytics, session replay, feature flags, and more.** ([View the source on GitHub](https://github.com/PostHog/posthog))

**Key Features:**

*   📊 **Product Analytics:** Track user behavior with autocapture and event-based analytics, visualize data, and use SQL for advanced analysis.
*   🌐 **Web Analytics:** Monitor website traffic, user sessions, conversion rates, web vitals, and revenue.
*   🎬 **Session Replay:** Watch real user sessions to understand how users interact with your website or app and diagnose issues.
*   🚩 **Feature Flags:** Safely release features to specific users or cohorts with feature flags.
*   🧪 **Experiments:** Test and measure the impact of changes with A/B testing, with a no-code option.
*   🐛 **Error Tracking:** Monitor and resolve errors to improve product stability.
*   ✉️ **Surveys:** Gather user feedback with no-code survey templates or custom-built surveys.
*   🗄️ **Data Warehouse:** Integrate data from external tools like Stripe, HubSpot, and your data warehouse to analyze alongside your product data.
*   ⚙️ **Data Pipelines:** Transform and route your data to various tools and warehouses with data pipelines.
*   🤖 **LLM Analytics:** Track LLM-powered apps including traces, generations, latency, and cost.

**Get Started for Free:**
Enjoy a generous free tier with PostHog Cloud. Sign up at [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup).

## Table of Contents

*   [PostHog: Open Source Product Analytics for Growth](#posthog-open-source-product-analytics-for-growth)
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

## Getting started with PostHog

### PostHog Cloud (Recommended)

The easiest and most reliable way to get started with PostHog is by signing up for free to [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup). The free tier includes 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 1500 survey responses per month. Paid plans are based on usage.

### Self-hosting the open-source hobby deploy (Advanced)

For self-hosting, deploy a hobby instance with Docker on Linux using this command (recommended with 4GB memory):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Self-hosted deployments typically support up to 100k events per month. For larger volumes, consider migrating to [PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud).

**Important:** Open-source deployments don't receive customer support or guarantees. Refer to our [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer) for details.

## Setting up PostHog

After setting up your PostHog instance, install the [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), use one of [our SDKs](https://posthog.com/docs/getting-started/install?tab=sdks), or utilize [our API](https://posthog.com/docs/getting-started/install?tab=api).

SDKs and libraries are available for popular languages and frameworks, including:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Additional documentation and guides cover [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

Once PostHog is installed, consult our [product docs](https://posthog.com/docs/product-os) for setting up [product analytics](https://posthog.com/docs/product-analytics/capture-events), [web analytics](https://posthog.com/docs/web-analytics/getting-started), [session replays](https://posthog.com/docs/session-replay/how-to-watch-recordings), [feature flags](https://posthog.com/docs/feature-flags/creating-feature-flags), [experiments](https://posthog.com/docs/experiments/creating-an-experiment), [error tracking](https://posthog.com/docs/error-tracking/installation#setting-up-exception-autocapture), [surveys](https://posthog.com/docs/surveys/installation), [data warehouse](https://posthog.com/docs/cdp/sources), and other features.

## Learning more about PostHog

We open source our [company handbook](https://posthog.com/handbook), offering insights into our [strategy](https://posthog.com/handbook/why-does-posthog-exist), [ways of working](https://posthog.com/handbook/company/culture), and [processes](https://posthog.com/handbook/team-structure).

For guidance on maximizing PostHog's potential, check out our [winning with PostHog guide](https://posthog.com/docs/new-to-posthog/getting-hogpilled), which covers [measuring activation](https://posthog.com/docs/new-to-posthog/activation), [tracking retention](https://posthog.com/docs/new-to-posthog/retention), and [capturing revenue](https://posthog.com/docs/new-to-posthog/revenue).

## Contributing

We welcome contributions! Here's how you can participate:

*   Vote on features or get early access to betas on our [roadmap](https://posthog.com/roadmap).
*   Submit a PR (see our instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally)).
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or a [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md).

## Open-source vs. paid

This repository is available under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory (which has its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)).

For a fully Free and Open Source Software (FOSS) solution, explore our [posthog-foss](https://github.com/PostHog/posthog-foss) repository, which excludes proprietary code and features.

Pricing for our paid plans is transparent and available on [our pricing page](https://posthog.com/pricing).

## We’re hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

If you've made it this far, you might be a great fit for our team!

We're growing rapidly, and we encourage you to explore opportunities with us at [posthog.com/careers](https://posthog.com/careers).