<p align="center">
  <img alt="PostHog Logo" src="https://user-images.githubusercontent.com/65415371/205059737-c8a4f836-4889-4654-902e-f302b16a0.png">
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

## PostHog: The Open-Source Platform Powering Product Success

**PostHog is an all-in-one, open-source product analytics platform that empowers product teams to build successful products.**  

**Key Features:**

*   **Product Analytics:** Understand user behavior with autocapture and event-based analytics, utilizing visualization and SQL.
*   **Web Analytics:** Monitor web traffic, user sessions, and key metrics like conversions, web vitals, and revenue.
*   **Session Replays:** Watch real user sessions to diagnose issues and gain insights into user interactions.
*   **Feature Flags:** Safely release features to specific user segments using feature flags.
*   **Experiments:** Test changes and measure their impact on goals using no-code experiment setup.
*   **Error Tracking:** Track and resolve errors, receiving alerts to improve product quality.
*   **Surveys:** Gather user feedback with no-code survey templates and a custom survey builder.
*   **Data Warehouse:** Integrate data from external tools like Stripe and HubSpot, and query it alongside product data.
*   **Data Pipelines:** Transform your data, route it to various destinations, and export large volumes to your data warehouse.
*   **LLM Analytics:** Capture key metrics for your LLM-powered applications.

**Get Started:** Free to use with a generous monthly free tier: [PostHog Pricing](https://posthog.com/pricing)

**Sign Up:**
*   [PostHog Cloud US](https://us.posthog.com/signup)
*   [PostHog Cloud EU](https://eu.posthog.com/signup)

## Table of Contents

*   [PostHog: The Open-Source Platform Powering Product Success](#posthog-the-open-source-platform-powering-product-success)
*   [Key Features](#key-features)
*   [Table of Contents](#table-of-contents)
*   [Getting Started with PostHog](#getting-started-with-posthog)
    *   [PostHog Cloud (Recommended)](#posthog-cloud-recommended)
    *   [Self-hosting (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
*   [Setting up PostHog](#setting-up-posthog)
*   [Learning More about PostHog](#learning-more-about-posthog)
*   [Contributing](#contributing)
*   [Open-source vs. Paid](#open-source-vs-paid)
*   [We’re Hiring!](#were-hiring)

## Getting Started with PostHog

### PostHog Cloud (Recommended)

The fastest and most reliable way to get started is by signing up for free at [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup). Enjoy a free tier including 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 1500 survey responses per month.

### Self-hosting (Advanced)

For those wanting to self-host, deploy a hobby instance in one line using Docker (4GB memory recommended):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Open source deployments are designed to handle approximately 100k events per month. For higher event volumes, consider migrating to [PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud).

*Disclaimer: No customer support or guarantees for open-source deployments.* See our [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer).

## Setting up PostHog

Integrate PostHog into your project by using our [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), one of [our SDKs](https://posthog.com/docs/getting-started/install?tab=sdks), or through [our API](https://posthog.com/docs/getting-started/install?tab=api).

**SDKs and Libraries:**

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Also available: [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

Once installed, learn more in our [product docs](https://posthog.com/docs/product-os) for [product analytics](https://posthog.com/docs/product-analytics/capture-events), [web analytics](https://posthog.com/docs/web-analytics/getting-started), [session replays](https://posthog.com/docs/session-replay/how-to-watch-recordings), [feature flags](https://posthog.com/docs/feature-flags/creating-feature-flags), [experiments](https://posthog.com/docs/experiments/creating-an-experiment), [error tracking](https://posthog.com/docs/error-tracking/installation#setting-up-exception-autocapture), [surveys](https://posthog.com/docs/surveys/installation), [data warehouse](https://posthog.com/docs/cdp/sources), and more.

## Learning More about PostHog

Explore PostHog's open-source resources: [company handbook](https://posthog.com/handbook), covering [strategy](https://posthog.com/handbook/why-does-posthog-exist), [culture](https://posthog.com/handbook/company/culture), and [processes](https://posthog.com/handbook/team-structure).

For a quick start, consult our guide on [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled) to master [measuring activation](https://posthog.com/docs/new-to-posthog/activation), [tracking retention](https://posthog.com/docs/new-to-posthog/retention), and [capturing revenue](https://posthog.com/docs/new-to-posthog/revenue).

## Contributing

Join our community! We welcome contributions of all sizes:

*   Vote on features or get early access in our [roadmap](https://posthog.com/roadmap)
*   Submit a Pull Request ([developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally))
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md)

## Open-source vs. Paid

The core code is available under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory (which has its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)) if applicable.

Need 100% FOSS? Check out our [posthog-foss](https://github.com/PostHog/posthog-foss) repository.

Pricing is available at [our pricing page](https://posthog.com/pricing).

## We’re Hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

Enjoyed reading this README? Consider joining our growing team! Learn more at [PostHog Careers](https://posthog.com/careers).

[**Go back to the original repository**](https://github.com/PostHog/posthog)