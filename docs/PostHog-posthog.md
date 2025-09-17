<p align="center">
  <img alt="PostHog Logo" src="https://user-images.githubusercontent.com/65415371/205059737-c8a4f836-4889-4654-902e-f302b160.png">
</p>

<p align="center">
  <a href='https://posthog.com/contributors'><img alt="GitHub contributors" src="https://img.shields.io/github/contributors/posthog/posthog"/></a>
  <a href='http://makeapullrequest.com'><img alt='PRs Welcome' src='https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=shields'/></a>
  <img alt="Docker Pulls" src="https://img.shields.io/docker/pulls/posthog/posthog"/>
  <a href="https://github.com/PostHog/posthog/commits/master"><img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/posthog/posthog"/> </a>
  <a href="https://github.com/PostHog/posthog/issues?q=is%3Aissue%20state%3Aclosed"><img alt="GitHub closed issues" src="https://img.shields.io/github/issues-closed/posthog/posthog"/> </a>
</p>

<p align="center">
  <a href="https://posthog.com/docs">Docs</a> - <a href="https://posthog.com/community">Community</a> - <a href="https://posthog.com/roadmap">Roadmap</a> - <a href="https://posthog.com/why">Why PostHog?</a> - <a href="https://posthog.com/changelog">Changelog</a> - <a href="https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md">Bug reports</a>
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=2jQco8hEvTI">
    <img src="https://res.cloudinary.com/dmukukwp6/image/upload/demo_thumb_68d0d8d56d" alt="PostHog Demonstration">
  </a>
</p>

## PostHog: Open-Source Product Analytics for Modern Businesses

**PostHog is a powerful, open-source platform providing a complete suite of tools to help you build and grow successful products.** [Explore the code on GitHub](https://github.com/PostHog/posthog).

**Key Features:**

*   **Product Analytics:** Understand user behavior with event-based analytics, visualizations, and SQL querying.
*   **Web Analytics:** Monitor website traffic, user sessions, conversion rates, web vitals, and revenue with an intuitive dashboard.
*   **Session Replays:**  Watch recordings of real user sessions to diagnose issues and understand user interactions.
*   **Feature Flags:**  Safely roll out new features to specific user segments.
*   **Experiments:** Test product changes and measure their impact using A/B testing.
*   **Error Tracking:** Identify and resolve errors quickly to improve product stability.
*   **Surveys:** Collect user feedback using pre-built templates or custom surveys.
*   **Data Warehouse:**  Integrate data from external tools like Stripe, Hubspot, and your data warehouse.
*   **Data Pipelines:** Transform your incoming data and send it to various tools.
*   **LLM Analytics:** Track LLM-powered application metrics like traces, generations, latency, and cost.

**Get Started:**

*   **PostHog Cloud (Recommended):** Sign up for a free account with generous monthly free tiers: [US](https://us.posthog.com/signup) or [EU](https://eu.posthog.com/signup).
*   **Self-Hosting (Advanced):** Deploy a hobby instance with Docker:  ` /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"`

## Table of Contents

*   [PostHog: Open-Source Product Analytics for Modern Businesses](#posthog-open-source-product-analytics-for-modern-businesses)
*   [Key Features](#key-features)
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

The easiest way to get started is by signing up for a free PostHog Cloud account: [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup). Free tier includes 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 1500 survey responses monthly.

### Self-hosting the open-source hobby deploy (Advanced)

Self-host a hobby instance using the one-line Docker deploy:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

**Important:** Open-source deployments are designed for approximately 100k events per month. For higher volumes, consider migrating to [PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud). Support is not provided for self-hosted deployments.

## Setting up PostHog

Integrate PostHog into your project with:

*   [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet)
*   SDKs:

    | Frontend                                              | Mobile                                                          | Backend                                             |
    | ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
    | [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
    | [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
    | [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
    | [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

*   [API](https://posthog.com/docs/getting-started/install?tab=api)

## Learning more about PostHog

*   [Company Handbook](https://posthog.com/handbook): Our strategy, ways of working, and processes.
*   [Winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled): Learn the basics of activation, retention, and revenue tracking.

## Contributing

Contribute and help improve PostHog!

*   [Roadmap](https://posthog.com/roadmap) - Vote on features and get early access
*   [Developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally)
*   [Feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md)
*   [Bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md)

## Open-source vs. paid

This repository is available under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE) (excluding the `ee` directory, which has its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)).
For a completely free and open-source version, see [posthog-foss](https://github.com/PostHog/posthog-foss).

See [our pricing page](https://posthog.com/pricing) for information on our paid plans.

## We’re hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

Join the PostHog team!  Apply at [https://posthog.com/careers](https://posthog.com/careers).