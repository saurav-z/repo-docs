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

## PostHog: The Open Source Platform for Product Success

**PostHog is an all-in-one, open-source platform that provides product teams with the tools they need to build and optimize successful products.**  ([Explore the code on GitHub](https://github.com/PostHog/posthog))

**Key Features:**

*   **Product Analytics:** Understand user behavior with event-based analytics, data visualization, and SQL querying.
*   **Web Analytics:** Monitor web traffic, user sessions, conversion rates, and web vitals.
*   **Session Replays:** Watch real user sessions to diagnose issues and understand user interactions.
*   **Feature Flags:** Safely roll out features to specific users or cohorts.
*   **Experiments:** A/B test changes and measure their impact with no-code setup.
*   **Error Tracking:** Track and resolve errors to improve product stability.
*   **Surveys:** Gather user feedback with no-code survey templates.
*   **Data Warehouse:** Integrate data from external tools for comprehensive analysis.
*   **Data Pipelines:** Transform and route your incoming data to various destinations.
*   **LLM Observability:** Capture traces, generations, latency, and cost for your LLM-powered app.

PostHog offers a generous [free tier](https://posthog.com/pricing) and is available for both cloud and self-hosted deployments.

**Jump to a section:**

*   [Getting Started with PostHog](#getting-started-with-posthog)
    *   [PostHog Cloud (Recommended)](#posthog-cloud-recommended)
    *   [Self-hosting the open-source hobby deploy (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
*   [Setting up PostHog](#setting-up-posthog)
*   [Learning More About PostHog](#learning-more-about-posthog)
*   [Contributing](#contributing)
*   [Open-source vs. paid](#open-source-vs-paid)
*   [We're Hiring!](#were-hiring)

## Getting Started with PostHog

### PostHog Cloud (Recommended)

The easiest way to get started is by signing up for a free [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup) account. You get a generous free tier with your first million events, 5k recordings, 1M flag requests, 100k exceptions, and 250 survey responses free each month!

### Self-hosting the Open-Source Hobby Deploy (Advanced)

For self-hosting, deploy a hobby instance with Docker (4GB memory recommended) using this one-line command:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Self-hosted deployments are suggested to scale up to approximately 100k events per month.  We recommend [migrating to PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud) for larger scales.

**Important:** Open source deployments do not offer customer support or guarantees. See the [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer) for more details.

## Setting Up PostHog

Integrate PostHog with your project using the [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), one of the available [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks), or the [API](https://posthog.com/docs/getting-started/install?tab=api).

**SDKs Available:**

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Also available: [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

After installing, refer to the [product docs](https://posthog.com/docs/product-os) for detailed instructions on setting up:  [product analytics](https://posthog.com/docs/product-analytics/capture-events), [web analytics](https://posthog.com/docs/web-analytics/getting-started), [session replays](https://posthog.com/docs/session-replay/how-to-watch-recordings), [feature flags](https://posthog.com/docs/feature-flags/creating-feature-flags), [experiments](https://posthog.com/docs/experiments/creating-an-experiment), [error tracking](https://posthog.com/docs/error-tracking/installation#setting-up-exception-autocapture), [surveys](https://posthog.com/docs/surveys/installation), [data warehouse](https://posthog.com/docs/cdp/sources), and more.

## Learning More About PostHog

Discover our [company handbook](https://posthog.com/handbook) to learn about our [strategy](https://posthog.com/handbook/why-does-posthog-exist), [culture](https://posthog.com/handbook/company/culture), and [processes](https://posthog.com/handbook/team-structure).

For the most out of PostHog, explore our guide on [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled) which goes over how to measure [activation](https://posthog.com/docs/new-to-posthog/activation), [retention](https://posthog.com/docs/new-to-posthog/retention), and [revenue](https://posthog.com/docs/new-to-posthog/revenue).

## Contributing

We welcome contributions!

*   Contribute to our [roadmap](https://posthog.com/roadmap)
*   Submit a PR (see our instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally))
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md)

## Open-source vs. paid

This repository is licensed under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), with the `ee` directory having its own [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE) if applicable.

For a completely free and open-source version, see our [posthog-foss](https://github.com/PostHog/posthog-foss) repository.

Our paid plan pricing is transparent and accessible on our [pricing page](https://posthog.com/pricing).

## We’re Hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

If you're reading this, you might be a great fit for our team! We're growing fast and [would love for you to join us](https://posthog.com/careers).
```

Key improvements and SEO considerations:

*   **Clear Title:** Uses "PostHog: The Open Source Platform for Product Success" for a keyword-rich title.
*   **Concise Hook:** The first sentence immediately states the core benefit.
*   **Keyword Optimization:** Uses relevant keywords like "open source," "product analytics," and feature names.
*   **Structured Headings:** Uses clear headings to improve readability and SEO ranking by signaling content organization.
*   **Bulleted Key Features:**  Emphasizes core functionalities and makes them easy to scan.
*   **Internal Linking:** Links to key areas within the README and the PostHog website (docs, pricing, etc.).
*   **Calls to Action:** Encourages the reader to explore the features, sign up, and contribute.
*   **Clean Formatting:**  Uses markdown consistently to improve readability and SEO.
*   **Updated Visuals:** Maintains the image and badge links.
*   **Improved Tone:** More engaging and inviting.
*   **Comprehensive:** Provides detailed information without being overly long.