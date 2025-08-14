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

## **PostHog: The Open-Source Product Analytics Platform to Build Successful Products**

PostHog is an all-in-one open-source platform packed with powerful tools to help you understand your users, improve your product, and drive growth.  [Discover PostHog on GitHub](https://github.com/PostHog/posthog).

**Key Features:**

*   **Product Analytics:**  Deep dive into user behavior with event-based analytics, data visualization, and SQL querying.
*   **Web Analytics:**  Monitor website traffic, user sessions, conversions, and web vitals with an intuitive, GA-like dashboard.
*   **Session Replays:**  Watch recordings of real user interactions to diagnose issues and understand user experience.
*   **Feature Flags:**  Safely roll out new features and conduct A/B testing.
*   **Experiments:**  Test changes and measure their statistical impact on goal metrics, no-code required.
*   **Error Tracking:**  Identify, track, and resolve errors to enhance product stability.
*   **Surveys:**  Collect user feedback with customizable surveys.
*   **Data Warehouse & Pipelines:** Sync data from external tools, run custom filters and transformations, and send data to 25+ tools or any webhook in real-time or batch export large amounts to your warehouse.
*   **LLM Observability:** Capture traces, generations, latency, and cost for your LLM-powered app.

PostHog offers a generous monthly free tier, with paid plans for increased usage. Get started with [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup).

### **Table of Contents**

*   [PostHog: The Open-Source Product Analytics Platform to Build Successful Products](#posthog-the-open-source-product-analytics-platform-to-build-successful-products)
*   [Key Features](#key-features)
*   [Getting Started with PostHog](#getting-started-with-posthog)
    *   [PostHog Cloud (Recommended)](#posthog-cloud-recommended)
    *   [Self-hosting the Open-Source Hobby Deploy (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
*   [Setting Up PostHog](#setting-up-posthog)
*   [Learning More About PostHog](#learning-more-about-posthog)
*   [Contributing](#contributing)
*   [Open-Source vs. Paid](#open-source-vs-paid)
*   [We're Hiring!](#were-hiring)

## Getting Started with PostHog

### PostHog Cloud (Recommended)

The quickest and most reliable way to experience PostHog is by signing up for free to [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup). Enjoy a generous free tier, with paid plans based on usage.

### Self-hosting the Open-Source Hobby Deploy (Advanced)

For self-hosting, deploy a hobby instance using Docker on Linux:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Open source deployments are suitable for approximately 100k events per month. Refer to our [self-hosting documentation](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer) for more information.

## Setting Up PostHog

Integrate PostHog using our [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), one of [our SDKs](https://posthog.com/docs/getting-started/install?tab=sdks), or the [PostHog API](https://posthog.com/docs/getting-started/install?tab=api).

SDKs are available for various languages and frameworks:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Documentation and guides are also available for [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

After installation, explore our [product docs](https://posthog.com/docs/product-os) for detailed setup instructions for [product analytics](https://posthog.com/docs/product-analytics/capture-events), [web analytics](https://posthog.com/docs/web-analytics/getting-started), [session replays](https://posthog.com/docs/session-replay/how-to-watch-recordings), [feature flags](https://posthog.com/docs/feature-flags/creating-feature-flags), [experiments](https://posthog.com/docs/experiments/creating-an-experiment), [error tracking](https://posthog.com/docs/error-tracking/installation#setting-up-exception-autocapture), [surveys](https://posthog.com/docs/surveys/installation), [data warehouse](https://posthog.com/docs/cdp/sources), and more.

## Learning More About PostHog

Discover our open-source [company handbook](https://posthog.com/handbook) to understand our [strategy](https://posthog.com/handbook/why-does-posthog-exist), [culture](https://posthog.com/handbook/company/culture), and [processes](https://posthog.com/handbook/team-structure).

Learn how to maximize your PostHog experience with our guide to [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled), covering [measuring activation](https://posthog.com/docs/new-to-posthog/activation), [tracking retention](https://posthog.com/docs/new-to-posthog/retention), and [capturing revenue](https://posthog.com/docs/new-to-posthog/revenue).

## Contributing

We welcome contributions of all sizes:

*   Vote on features and access beta functionality in our [roadmap](https://posthog.com/roadmap).
*   Submit a Pull Request (PR) – see instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally).
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md).

## Open-Source vs. Paid

This repository is licensed under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), excluding the `ee` directory (licensed [here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)).

For a 💯% FOSS experience, explore our [posthog-foss](https://github.com/PostHog/posthog-foss) repository, which is free of proprietary code.

Transparent pricing for our paid plan is available on [our pricing page](https://posthog.com/pricing).

## We’re Hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

Interested in joining the team? If so, [explore our careers](https://posthog.com/careers).