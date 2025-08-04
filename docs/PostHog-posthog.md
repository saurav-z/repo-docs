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

## PostHog: The Open-Source Product Analytics Platform for Building Successful Products

PostHog is a powerful, all-in-one, open-source platform empowering product teams to understand user behavior and build successful products.

**[Check out the original repo](https://github.com/PostHog/posthog).**

### Key Features:

*   **Product Analytics:** Deep dive into user behavior with event-based analytics, visualizations, and SQL querying.
*   **Web Analytics:** Monitor website traffic, user sessions, conversions, web vitals, and revenue.
*   **Session Replays:** Watch recordings of real user sessions to diagnose issues and understand user interactions.
*   **Feature Flags:** Safely release new features to specific user segments.
*   **Experiments:** A/B test changes and measure their impact on key metrics with no-code setup.
*   **Error Tracking:** Identify, track, and resolve errors to improve product stability.
*   **Surveys:** Gather user feedback with no-code survey templates.
*   **Data Warehouse:** Integrate data from various sources and query it alongside your product data.
*   **Data Pipelines:** Transform and route your data to 25+ tools or your data warehouse.
*   **LLM Observability:** Monitor traces, generations, latency, and cost for LLM-powered applications.

PostHog offers a generous [free tier](https://posthog.com/pricing), and is available on [PostHog Cloud US](https://us.posthog.com/signup) and [PostHog Cloud EU](https://eu.posthog.com/signup).

## Table of Contents

*   [PostHog: The Open-Source Product Analytics Platform for Building Successful Products](#posthog-the-open-source-product-analytics-platform-for-building-successful-products)
*   [Key Features:](#key-features)
*   [Getting Started with PostHog](#getting-started-with-posthog)
    *   [PostHog Cloud (Recommended)](#posthog-cloud-recommended)
    *   [Self-hosting the open-source hobby deploy (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
*   [Setting Up PostHog](#setting-up-posthog)
*   [Learning More About PostHog](#learning-more-about-posthog)
*   [Contributing](#contributing)
*   [Open-source vs. Paid](#open-source-vs-paid)
*   [We’re Hiring!](#were-hiring)

## Getting Started with PostHog

### PostHog Cloud (Recommended)

The easiest and most reliable way to begin using PostHog is to sign up for free on [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup). The free tier includes 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 250 survey responses per month.

### Self-hosting the open-source hobby deploy (Advanced)

For self-hosting, you can deploy a hobby instance using Docker:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Open source deployments are recommended for up to ~100k events/month. See [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer).

## Setting Up PostHog

Set up PostHog after you get a running instance by installing the [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), one of our [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks), or by using our [API](https://posthog.com/docs/getting-started/install?tab=api).

SDKs and libraries available for:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Also available for: [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

See our [product docs](https://posthog.com/docs/product-os) to set up [product analytics](https://posthog.com/docs/product-analytics/capture-events), [web analytics](https://posthog.com/docs/web-analytics/getting-started), [session replays](https://posthog.com/docs/session-replay/how-to-watch-recordings), [feature flags](https://posthog.com/docs/feature-flags/creating-feature-flags), [experiments](https://posthog.com/docs/experiments/creating-an-experiment), [error tracking](https://posthog.com/docs/error-tracking/installation#setting-up-exception-autocapture), [surveys](https://posthog.com/docs/surveys/installation), [data warehouse](https://posthog.com/docs/cdp/sources), and more.

## Learning More About PostHog

Explore our [company handbook](https://posthog.com/handbook) for insights into our [strategy](https://posthog.com/handbook/why-does-posthog-exist), [culture](https://posthog.com/handbook/company/culture), and [processes](https://posthog.com/handbook/team-structure).

Learn how to maximize PostHog with our guide to [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled), covering [measuring activation](https://posthog.com/docs/new-to-posthog/activation), [tracking retention](https://posthog.com/docs/new-to-posthog/retention), and [capturing revenue](https://posthog.com/docs/new-to-posthog/revenue).

## Contributing

We welcome contributions of all sizes:

*   Vote on features or get early access to beta functionality in our [roadmap](https://posthog.com/roadmap).
*   Open a PR (see instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally)).
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md).

## Open-source vs. Paid

This repo is available under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory (which has its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)) if applicable.

For a fully open-source solution, check out [posthog-foss](https://github.com/PostHog/posthog-foss).

Transparent pricing is available on [our pricing page](https://posthog.com/pricing).

## We’re Hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

If you are still reading, consider joining our team! We're growing fast and [we'd love for you to join us](https://posthog.com/careers).
```
Key improvements and SEO considerations:

*   **Clear Headline:** Used a concise and keyword-rich headline: "PostHog: The Open-Source Product Analytics Platform for Building Successful Products." This immediately tells users what PostHog is and what it does.
*   **One-Sentence Hook:** Created a compelling one-sentence introduction that summarizes PostHog's core value.
*   **SEO-Friendly Content:** The content is now structured with relevant keywords.
*   **Bulleted Key Features:** Uses bullet points to highlight the main features, making them easy to scan and understand. This is more engaging for readers and helps with SEO.
*   **Internal Linking:** Links to relevant sections within the document and to external resources like documentation, pricing, etc.
*   **Concise and Actionable Language:** The language is clear, direct, and encourages user action (e.g., "Get started," "Sign up," "Explore").
*   **Optimized Table of Contents:** The table of contents is updated to reflect the improved structure.
*   **Call to Action:** Encourages readers to visit PostHog's website and to explore the platform.
*   **Hiring Section:** The "We're Hiring" section uses a relevant image and a clear call to action, which is great for attracting potential candidates.
*   **Markdown Formatting:**  Maintains the Markdown format for easy readability and potential rendering on other platforms.
*   **Keyword Density:** Strategically includes keywords like "open-source," "product analytics," "web analytics," "session replay," "feature flags," "experiments," etc. These keywords are naturally integrated within the text.
*   **Readability:** The document is formatted for easy readability.