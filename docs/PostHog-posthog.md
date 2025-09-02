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

## PostHog: The Open-Source Product Analytics Platform for Growth

[PostHog](https://github.com/PostHog/posthog) is a powerful, open-source platform designed to help you build successful products by understanding and optimizing user behavior.  It's an all-in-one platform for product analytics, session replays, feature flags, and more.

**Key Features:**

*   **Product Analytics:** Track user behavior with event-based analytics, autocapture, and SQL analysis.
*   **Web Analytics:**  Monitor web traffic, user sessions, conversion rates, and web vitals.
*   **Session Replays:**  Watch real user sessions to understand user interactions and diagnose issues.
*   **Feature Flags:**  Safely release new features to specific user segments.
*   **Experiments:** A/B test changes and measure their impact on key metrics.
*   **Error Tracking:** Identify and resolve issues by tracking errors and receiving alerts.
*   **Surveys:** Gather user feedback with no-code surveys.
*   **Data Warehouse & Pipelines:** Integrate data from external sources and transform it for deeper insights.
*   **LLM Analytics:** Analyze your LLM-powered app by capturing traces, generations, latency, and cost.

PostHog offers a generous free tier and open-source deployment options, making it accessible to teams of all sizes.

## Table of Contents

-   [PostHog: The Open-Source Product Analytics Platform for Growth](#posthog-the-open-source-product-analytics-platform-for-growth)
-   [Key Features](#key-features)
-   [Getting Started with PostHog](#getting-started-with-posthog)
    -   [PostHog Cloud (Recommended)](#posthog-cloud-recommended)
    -   [Self-hosting the open-source hobby deploy (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
-   [Setting up PostHog](#setting-up-posthog)
-   [Learning More about PostHog](#learning-more-about-posthog)
-   [Contributing](#contributing)
-   [Open-source vs. paid](#open-source-vs-paid)
-   [We’re hiring!](#were-hiring)

## Getting Started with PostHog

### PostHog Cloud (Recommended)

The easiest way to get started is by signing up for a free PostHog Cloud account:  [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup). The free tier includes 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 1500 survey responses monthly.

### Self-hosting the open-source hobby deploy (Advanced)

For self-hosting, you can deploy a hobby instance on Linux using Docker (recommended 4GB memory):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Open-source deployments are suitable for approximately 100k events per month. For higher volumes, consider PostHog Cloud.

**Important:** Open-source deployments do not receive customer support or guarantees.  Consult the [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer) for more information.

## Setting up PostHog

Integrate PostHog into your product using our:

*   [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet)
*   [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks)
*   [API](https://posthog.com/docs/getting-started/install?tab=api)

SDKs and libraries are available for:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Documentation and guides are also available for: [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

After installation, explore our [product docs](https://posthog.com/docs/product-os) for guidance on features such as [product analytics](https://posthog.com/docs/product-analytics/capture-events), [web analytics](https://posthog.com/docs/web-analytics/getting-started), [session replays](https://posthog.com/docs/session-replay/how-to-watch-recordings), [feature flags](https://posthog.com/docs/feature-flags/creating-feature-flags), [experiments](https://posthog.com/docs/experiments/creating-an-experiment), [error tracking](https://posthog.com/docs/error-tracking/installation#setting-up-exception-autocapture), [surveys](https://posthog.com/docs/surveys/installation), and [data warehouse](https://posthog.com/docs/cdp/sources).

## Learning More about PostHog

Our commitment to transparency extends to our documentation; we've open-sourced our [company handbook](https://posthog.com/handbook), which includes detailed information on our [strategy](https://posthog.com/handbook/why-does-posthog-exist), [ways of working](https://posthog.com/handbook/company/culture), and [processes](https://posthog.com/handbook/team-structure).

To discover how to maximize your PostHog experience, read our guide: [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled), which covers the fundamentals of [measuring activation](https://posthog.com/docs/new-to-posthog/activation), [tracking retention](https://posthog.com/docs/new-to-posthog/retention), and [capturing revenue](https://posthog.com/docs/new-to-posthog/revenue).

## Contributing

We appreciate all contributions!

*   Vote on features in our [roadmap](https://posthog.com/roadmap).
*   Submit a Pull Request (PR) - see our instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally).
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md).

## Open-source vs. paid

The `posthog` repository is available under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), excluding the `ee` directory (which has its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)).

For a completely free and open-source solution, explore the [posthog-foss](https://github.com/PostHog/posthog-foss) repository, which excludes proprietary code and features.

Find transparent pricing details for our paid plans on [our pricing page](https://posthog.com/pricing).

## We’re hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

Thanks for reading! We're looking for talented individuals to join our growing team.  Check out our [careers page](https://posthog.com/careers) and apply today!