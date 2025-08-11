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

## PostHog: The Open-Source Platform Powering Product Success 🚀

[PostHog](https://posthog.com/) is an all-in-one, open-source product analytics platform, empowering product teams to build successful products by providing all the tools they need.

**Key Features:**

*   **Product Analytics:** Deep dive into user behavior with event-based analytics, autocapture, and SQL-based analysis.
*   **Web Analytics:** Monitor web traffic, user sessions, conversion rates, web vitals, and revenue with a GA-like dashboard.
*   **Session Replays:** Understand user interactions by watching recordings of real user sessions on your website or mobile app.
*   **Feature Flags:** Safely roll out new features to specific users or cohorts using feature flags.
*   **Experiments:** Test product changes and measure their impact with A/B testing, including a no-code interface.
*   **Error Tracking:** Track and resolve errors with error tracking, and receive alerts to improve your product.
*   **Surveys:** Engage your users with no-code survey templates or custom-built surveys.
*   **Data Warehouse:** Integrate data from external sources like Stripe and Hubspot, and query it alongside your product data.
*   **Data Pipelines:** Transform and route your data in real-time to 25+ tools.
*   **LLM Observability:** Monitor the performance of your LLM-powered applications.

**Get Started Today!**

PostHog offers a generous free tier, and is available via:

*   [PostHog Cloud US](https://us.posthog.com/signup)
*   [PostHog Cloud EU](https://eu.posthog.com/signup)

**Looking for the source code?**  Find it all on [GitHub](https://github.com/PostHog/posthog)!

### Table of Contents

-   [PostHog: The Open-Source Platform Powering Product Success 🚀](#posthog-the-open-source-platform-powering-product-success-)
    -   [Key Features](#key-features)
    -   [Get Started Today!](#get-started-today)
    -   [Table of Contents](#table-of-contents)
    -   [Getting Started with PostHog](#getting-started-with-posthog)
        -   [PostHog Cloud (Recommended)](#posthog-cloud-recommended)
        -   [Self-hosting the open-source hobby deploy (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
    -   [Setting up PostHog](#setting-up-posthog)
    -   [Learning more about PostHog](#learning-more-about-posthog)
    -   [Contributing](#contributing)
    -   [Open-source vs. paid](#open-source-vs-paid)
    -   [We’re hiring!](#were-hiring)

## Getting started with PostHog

### PostHog Cloud (Recommended)

The easiest way to get started with PostHog is by signing up for a free account on [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup). You receive a generous free tier with your first 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 250 survey responses free every month.

### Self-hosting the open-source hobby deploy (Advanced)

For self-hosting, deploy a hobby instance on Linux with Docker (4GB memory recommended):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Open source deployments should scale to approximately 100k events per month, after which we recommend [migrating to a PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud).

_We do not_ provide customer support or offer guarantees for open source deployments. See our [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer) for more info.

## Setting up PostHog

Integrate PostHog by installing our [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), using one of [our SDKs](https://posthog.com/docs/getting-started/install?tab=sdks), or through our [API](https://posthog.com/docs/getting-started/install?tab=api).

SDKs and libraries are available for popular languages and frameworks:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

We also have docs and guides for [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

Learn how to set up [product analytics](https://posthog.com/docs/product-analytics/capture-events), [web analytics](https://posthog.com/docs/web-analytics/getting-started), [session replays](https://posthog.com/docs/session-replay/how-to-watch-recordings), [feature flags](https://posthog.com/docs/feature-flags/creating-feature-flags), [experiments](https://posthog.com/docs/experiments/creating-an-experiment), [error tracking](https://posthog.com/docs/error-tracking/installation#setting-up-exception-autocapture), [surveys](https://posthog.com/docs/surveys/installation), [data warehouse](https://posthog.com/docs/cdp/sources), and other features in our [product docs](https://posthog.com/docs/product-os).

## Learning more about PostHog

Our [company handbook](https://posthog.com/handbook) is open-source, and details our [strategy](https://posthog.com/handbook/why-does-posthog-exist), [ways of working](https://posthog.com/handbook/company/culture), and [processes](https://posthog.com/handbook/team-structure).

Discover how to optimize PostHog by reading our guide to [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled), covering [measuring activation](https://posthog.com/docs/new-to-posthog/activation), [tracking retention](https://posthog.com/docs/new-to-posthog/retention), and [capturing revenue](https://posthog.com/docs/new-to-posthog/revenue).

## Contributing

We welcome all contributions:

-   Vote on features or get early access to beta functionality in our [roadmap](https://posthog.com/roadmap)
-   Open a PR (see our instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally))
-   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md)

## Open-source vs. paid

This repo is available under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory (which has its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)) if applicable.

For a completely FOSS experience, check out our [posthog-foss](https://github.com/PostHog/posthog-foss) repository.

Transparent pricing for paid plans is available on [our pricing page](https://posthog.com/pricing).

## We’re hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

Interested in joining our team? We're growing fast [and would love for you to join us](https://posthog.com/careers).
```

Key improvements and SEO considerations:

*   **Compelling Headline:**  A concise and benefit-driven headline uses a power word ("Powering") and includes a relevant emoji for visual appeal.
*   **SEO-Optimized Description:** Includes relevant keywords (product analytics, web analytics, session replay, feature flags, open-source) to improve search ranking.
*   **Clear Feature Highlights:** Uses bullet points for easy readability, making key features stand out.  Keywords are included.
*   **Call to Action:** Encourages immediate action with clear links to the sign-up pages.
*   **Structured Content:**  Uses clear headings and subheadings to organize information and improve readability. This also helps search engines understand the document structure.
*   **Internal Linking:**  Links to other key pages (docs, community, roadmap, etc.) to encourage users to explore the platform and improve SEO.
*   **Keyword Density:** Uses keywords strategically throughout the text, without keyword stuffing.
*   **Alt Text for Images:**  Includes descriptive alt text for images to improve accessibility and SEO.
*   **Concise Language:** Uses clear and concise language throughout the document.
*   **Hiring Section:**  Keeps the hiring call to action.
*   **Complete and Summarized:**  The summary is a condensed version of all the important content.