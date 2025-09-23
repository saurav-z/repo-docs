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

## PostHog: The Open-Source Product Analytics Platform for Growth

**PostHog is an all-in-one, open-source product analytics platform that empowers you to build successful products.**

**[Explore the PostHog Repository on GitHub](https://github.com/PostHog/posthog)**

### Key Features:

*   **Product Analytics:** Deep dive into user behavior with event-based analytics, including autocapture, custom event tracking, and SQL querying.
*   **Web Analytics:** Get a GA-like dashboard to monitor web traffic, user sessions, conversions, web vitals, and revenue.
*   **Session Replays:** Understand user interactions by watching real user sessions on your website or mobile app.
*   **Feature Flags:** Safely roll out new features to specific user segments with feature flags.
*   **Experiments:** Test changes and measure their impact with A/B testing, including no-code experiment setup.
*   **Error Tracking:** Identify, monitor, and resolve errors to improve product quality.
*   **Surveys:** Gather user feedback with built-in survey templates and a custom survey builder.
*   **Data Warehouse & CDP Integrations:** Sync data from tools like Stripe and HubSpot, and integrate with your data warehouse for unified insights.
*   **Data Pipelines:** Run custom filters and transformations on your data, sending it to 25+ tools or your warehouse in real-time or batch.
*   **LLM Analytics:** Track key metrics for your LLM-powered apps including traces, generations, latency and costs.

**Get started for free!** PostHog offers a generous [monthly free tier](https://posthog.com/pricing) for each product. Sign up for [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup).

### Table of Contents

*   [PostHog: The Open-Source Product Analytics Platform for Growth](#posthog-the-open-source-product-analytics-platform-for-growth)
    *   [Key Features:](#key-features)
    *   [Table of Contents](#table-of-contents)
*   [Getting Started with PostHog](#getting-started-with-posthog)
    *   [PostHog Cloud (Recommended)](#posthog-cloud-recommended)
    *   [Self-hosting the open-source hobby deploy (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
*   [Setting up PostHog](#setting-up-posthog)
*   [Learning more about PostHog](#learning-more-about-posthog)
*   [Contributing](#contributing)
*   [Open-source vs. paid](#open-source-vs-paid)
*   [We’re hiring!](#were-hiring)

### Getting started with PostHog

#### PostHog Cloud (Recommended)

The easiest and most reliable way to begin with PostHog is to sign up for free at [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup). Enjoy a generous free tier with 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 1500 survey responses each month.

#### Self-hosting the open-source hobby deploy (Advanced)

For self-hosting, deploy a hobby instance using Docker with this one-line command (Linux, 4GB memory recommended):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Note: Open-source deployments are designed to handle approximately 100k events monthly. Migration to [PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud) is recommended as your usage grows.

We do not provide customer support or guarantees for open source deployments. Consult the [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer) for more details.

### Setting up PostHog

To set up your PostHog instance, install the [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), one of our [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks), or the [API](https://posthog.com/docs/getting-started/install?tab=api).

SDKs and libraries are available for major languages and frameworks:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Find more resources for [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and other platforms.

Learn how to use PostHog with our product docs covering [product analytics](https://posthog.com/docs/product-analytics/capture-events), [web analytics](https://posthog.com/docs/web-analytics/getting-started), [session replays](https://posthog.com/docs/session-replay/how-to-watch-recordings), [feature flags](https://posthog.com/docs/feature-flags/creating-feature-flags), [experiments](https://posthog.com/docs/experiments/creating-an-experiment), [error tracking](https://posthog.com/docs/error-tracking/installation#setting-up-exception-autocapture), [surveys](https://posthog.com/docs/surveys/installation), [data warehouse](https://posthog.com/docs/cdp/sources), and more.

### Learning more about PostHog

Our commitment to openness extends to our [company handbook](https://posthog.com/handbook), where you'll find our [strategy](https://posthog.com/handbook/why-does-posthog-exist), [ways of working](https://posthog.com/handbook/company/culture), and [processes](https://posthog.com/handbook/team-structure).

For in-depth guidance, consult our guide to "winning with PostHog" outlining [measuring activation](https://posthog.com/docs/new-to-posthog/activation), [tracking retention](https://posthog.com/docs/new-to-posthog/retention), and [capturing revenue](https://posthog.com/docs/new-to-posthog/revenue).

### Contributing

We value contributions, big and small!

*   Vote on or gain early access to beta features in our [roadmap](https://posthog.com/roadmap).
*   Submit a PR (see our instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally)).
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md).

### Open-source vs. paid

This repository is available under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory (which has its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)) if applicable.

For a 100% FOSS experience, explore the [posthog-foss](https://github.com/PostHog/posthog-foss) repository, which is free of proprietary code and features.

Find transparent pricing details for our paid plans on [our pricing page](https://posthog.com/pricing).

### We’re hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

Hello! If you've read this far, you're a dedicated README reader.

You could be a great fit for our growing team. We're looking for talented individuals to [join us](https://posthog.com/careers).
```
Key improvements and SEO-focused changes:

*   **Concise Hook:**  The one-sentence hook is clear, benefit-driven, and includes the core keyword: "PostHog: The Open-Source Product Analytics Platform for Growth".
*   **Keyword Optimization:**  The introduction uses keywords like "product analytics," "open-source," and "growth" strategically. The feature descriptions use relevant terms to improve search ranking for specific features.
*   **Clear Headings:**  Uses proper H1, H2, and H3 tags to structure the content logically for both readability and SEO.
*   **Bulleted Key Features:** Highlights the key functionalities of PostHog, making them easy to scan and understand.  Each bullet point is concise and benefit-oriented.
*   **Call to Action:** Includes a clear call to action to get started, linking to cloud signup options.
*   **Links:** The included links are relevant and target the key pages on the PostHog site and github.
*   **Readability:** Improved the overall flow and readability to enhance user engagement.
*   **Images:** Kept the original images and updated the alt text.