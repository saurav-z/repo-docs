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

## PostHog: The Open-Source Product Analytics Platform

**PostHog is an open-source product analytics platform that gives you the tools to build better products.**

[PostHog](https://posthog.com/) offers a comprehensive suite of tools, allowing you to understand and improve your product.

**Key Features:**

*   **Product Analytics:** Track user behavior and analyze data with event-based analytics.
*   **Web Analytics:** Monitor web traffic, user sessions, and key metrics with a GA-like dashboard.
*   **Session Replays:** Watch real user sessions to diagnose issues and understand user actions.
*   **Feature Flags:** Safely roll out features to specific users or cohorts.
*   **Experiments:** Test changes and measure their impact on important metrics.
*   **Error Tracking:** Identify, track, and resolve errors to improve product stability.
*   **Surveys:** Gather user feedback with no-code survey templates or custom builders.
*   **Data Warehouse:** Sync data from various sources and query it alongside your product data.
*   **Data Pipelines:** Transform and route your data to other tools in real-time or batch.
*   **LLM Analytics:** Capture insights into the performance of your LLM-powered applications.

Get started for free with a generous monthly free tier!  Sign up for [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup).

## Table of Contents

*   [PostHog: The Open-Source Product Analytics Platform](#posthog-the-open-source-product-analytics-platform)
*   [Key Features](#key-features)
*   [Getting Started with PostHog](#getting-started-with-posthog)
    *   [PostHog Cloud (Recommended)](#posthog-cloud-recommended)
    *   [Self-hosting the open-source hobby deploy (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
*   [Setting up PostHog](#setting-up-posthog)
*   [Learning More About PostHog](#learning-more-about-posthog)
*   [Contributing](#contributing)
*   [Open-Source vs. Paid](#open-source-vs-paid)
*   [We're Hiring!](#were-hiring)

## Getting Started with PostHog

### PostHog Cloud (Recommended)

The easiest and most reliable way to get started is by signing up for a free [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup) account.  You receive free usage for the first 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 1500 survey responses each month.

### Self-hosting the open-source hobby deploy (Advanced)

To self-host PostHog, you can deploy a hobby instance using Docker (recommended: 4GB memory):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Open source deployments are suitable for approximately 100k events per month. For larger volumes, consider migrating to [PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud).

Note: We do not offer customer support or provide guarantees for open source deployments. See our [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer) for more information.

## Setting up PostHog

Once you have a PostHog instance, install the [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), one of our [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks), or use our [API](https://posthog.com/docs/getting-started/install?tab=api).

We offer SDKs for various languages and frameworks:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Also, find documentation for [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

For further setup information, explore the [product docs](https://posthog.com/docs/product-os) for [product analytics](https://posthog.com/docs/product-analytics/capture-events), [web analytics](https://posthog.com/docs/web-analytics/getting-started), [session replays](https://posthog.com/docs/session-replay/how-to-watch-recordings), [feature flags](https://posthog.com/docs/feature-flags/creating-feature-flags), [experiments](https://posthog.com/docs/experiments/creating-an-experiment), [error tracking](https://posthog.com/docs/error-tracking/installation#setting-up-exception-autocapture), [surveys](https://posthog.com/docs/surveys/installation), [data warehouse](https://posthog.com/docs/cdp/sources), and more.

## Learning More About PostHog

Our [company handbook](https://posthog.com/handbook) is open-source and details our [strategy](https://posthog.com/handbook/why-does-posthog-exist), [ways of working](https://posthog.com/handbook/company/culture), and [processes](https://posthog.com/handbook/team-structure).

Learn how to get the most out of PostHog with our guide on [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled), which covers [measuring activation](https://posthog.com/docs/new-to-posthog/activation), [tracking retention](https://posthog.com/docs/new-to-posthog/retention), and [capturing revenue](https://posthog.com/docs/new-to-posthog/revenue).

## Contributing

We welcome contributions of all sizes:

*   Contribute by voting on features and getting early access to beta functionality via our [roadmap](https://posthog.com/roadmap)
*   Submit a Pull Request (PR) following the instructions for [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally)
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md)

## Open-Source vs. Paid

The source code for PostHog is available under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory, which has its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE), if applicable.

If you need a completely free and open-source version, check out our [posthog-foss](https://github.com/PostHog/posthog-foss) repository.

View the transparent pricing for our paid plan on [our pricing page](https://posthog.com/pricing).

## We’re Hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

If you’ve read this far, you might just be a great fit for our team.  Join us!  Find out more about our [careers](https://posthog.com/careers).
```
Key improvements:

*   **SEO-Optimized Title and Introduction:** Added "The Open-Source Product Analytics Platform" to the main title to improve SEO.  The hook clearly states what PostHog is.
*   **Clear Headings:**  Used H2 headings to break up the content.
*   **Bulleted Key Features:**  Made the features easily scannable.
*   **Concise Language:**  Condensed some sentences for better readability.
*   **Call to Action:**  Encouraged users to sign up.
*   **Improved Formatting:** Improved overall readability and organization.
*   **Added Keywords:** Incorporated keywords like "open-source," "product analytics," "web analytics," "session replay," "feature flags," and more throughout the document to aid SEO.
*   **Improved phrasing** More conversational and easier to scan.
*   **Removed Redundancy:** Eliminated some repeated sentences and shortened others for conciseness.
*   **Internal Linking:**  Links within the README for easy navigation.
*   **Consistent formatting.**

This revised README is more informative, user-friendly, and optimized for search engines. It provides a much better overview of what PostHog offers and encourages users to explore the platform.