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

## PostHog: The Open-Source Product Analytics Powerhouse

**PostHog is an open-source product analytics platform that empowers you to build better products by understanding your users' behavior.** ([View the source code on GitHub](https://github.com/PostHog/posthog))

**Key Features:**

*   **Product Analytics:** Analyze user behavior with event-based analytics, visualizations, and SQL.
*   **Web Analytics:** Monitor website traffic, user sessions, conversion rates, and web vitals.
*   **Session Replays:** Watch real user sessions to diagnose issues and understand user interactions.
*   **Feature Flags:** Safely roll out features and conduct A/B tests.
*   **Experiments:** Test changes and measure their impact with no-code experimentation.
*   **Error Tracking:** Track errors, receive alerts, and resolve issues to improve your product.
*   **Surveys:** Gather user feedback with no-code survey templates or a custom builder.
*   **Data Warehouse:** Integrate data from external tools and query it alongside product data.
*   **Data Pipelines:** Transform and route your data to various tools and warehouses.
*   **LLM Observability:** Monitor traces, generations, latency, and costs for your LLM-powered apps.

**Get Started:**

PostHog offers a generous free tier for all its features! Sign up for [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup) to start building better products today!

## Table of Contents

*   [PostHog: The Open-Source Product Analytics Powerhouse](#posthog-the-open-source-product-analytics-powerhouse)
*   [Key Features](#key-features)
*   [Get Started](#get-started)
*   [Table of Contents](#table-of-contents)
*   [Getting Started with PostHog](#getting-started-with-posthog)
    *   [PostHog Cloud (Recommended)](#posthog-cloud-recommended)
    *   [Self-hosting the open-source hobby deploy (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
*   [Setting up PostHog](#setting-up-posthog)
*   [Learning More About PostHog](#learning-more-about-posthog)
*   [Contributing](#contributing)
*   [Open-source vs. Paid](#open-source-vs-paid)
*   [We’re Hiring!](#were-hiring)

## Getting Started with PostHog

### PostHog Cloud (Recommended)

The easiest way to get started is to sign up for free to [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup). Enjoy a free tier with 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 250 survey responses every month.

### Self-hosting the open-source hobby deploy (Advanced)

Self-host PostHog with Docker (4GB memory recommended) in one line on Linux:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Open source deployments are ideal for up to approximately 100k events per month. For higher volumes, we recommend migrating to [PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud).

We do not offer customer support for self-hosted instances. Refer to the [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer).

## Setting Up PostHog

Integrate PostHog using the [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), our [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks), or the [API](https://posthog.com/docs/getting-started/install?tab=api).

SDKs are available for:

*   Frontend: [JavaScript](https://posthog.com/docs/libraries/js), [Next.js](https://posthog.com/docs/libraries/next-js), [React](https://posthog.com/docs/libraries/react), [Vue](https://posthog.com/docs/libraries/vue-js)
*   Mobile: [React Native](https://posthog.com/docs/libraries/react-native), [Android](https://posthog.com/docs/libraries/android), [iOS](https://posthog.com/docs/libraries/ios), [Flutter](https://posthog.com/docs/libraries/flutter)
*   Backend: [Python](https://posthog.com/docs/libraries/python), [Node](https://posthog.com/docs/libraries/node), [PHP](https://posthog.com/docs/libraries/php), [Ruby](https://posthog.com/docs/libraries/ruby)

Explore docs for [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

After installation, consult our [product docs](https://posthog.com/docs/product-os) for detailed setup guides on [product analytics](https://posthog.com/docs/product-analytics/capture-events), [web analytics](https://posthog.com/docs/web-analytics/getting-started), [session replays](https://posthog.com/docs/session-replay/how-to-watch-recordings), [feature flags](https://posthog.com/docs/feature-flags/creating-feature-flags), [experiments](https://posthog.com/docs/experiments/creating-an-experiment), [error tracking](https://posthog.com/docs/error-tracking/installation#setting-up-exception-autocapture), [surveys](https://posthog.com/docs/surveys/installation), [data warehouse](https://posthog.com/docs/cdp/sources), and more.

## Learning More About PostHog

Dive deeper into PostHog by exploring our open-source [company handbook](https://posthog.com/handbook), which covers our [strategy](https://posthog.com/handbook/why-does-posthog-exist), [culture](https://posthog.com/handbook/company/culture), and [processes](https://posthog.com/handbook/team-structure).

For valuable insights, read our guide to [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled), covering [measuring activation](https://posthog.com/docs/new-to-posthog/activation), [tracking retention](https://posthog.com/docs/new-to-posthog/retention), and [capturing revenue](https://posthog.com/docs/new-to-posthog/revenue).

## Contributing

We welcome all contributions!

*   Vote on features or get early access to beta functionality in our [roadmap](https://posthog.com/roadmap).
*   Submit a PR (see our instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally)).
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md).

## Open-source vs. paid

This repository is available under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory (which has its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)) if applicable.

For a fully FOSS version, check out our [posthog-foss](https://github.com/PostHog/posthog-foss) repository.

Transparent pricing is available on [our pricing page](https://posthog.com/pricing).

## We’re Hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

If you've made it this far, we'd love for you to consider joining our team! We're growing fast [and would love for you to join us](https://posthog.com/careers).
```

Key improvements and SEO considerations:

*   **Clear, concise title:** "PostHog: The Open-Source Product Analytics Powerhouse" is more compelling than the original title.
*   **One-sentence hook:**  The opening sentence immediately grabs the reader's attention and explains the core value proposition.
*   **Keyword optimization:** The description uses relevant keywords such as "open-source," "product analytics," "user behavior," etc., naturally throughout the text.
*   **Subheadings:**  Organized content with clear subheadings makes the README easier to scan and understand, which is good for both users and search engines.
*   **Bulleted key features:**  Clearly highlights the main benefits of PostHog.
*   **Calls to action:** Includes a direct call to action to encourage users to sign up.
*   **Internal linking:** Links to different sections of the README helps improve navigation.
*   **External links:**  Uses relevant links to official documentation, related pages, and resources.
*   **Alt text for images:** Adds descriptive alt text to the images to improve accessibility and SEO.
*   **Concise language:** Uses clear, direct language to convey information effectively.
*   **Job Posting Emphasis:** Enhanced the "We're Hiring!" section to increase impact.
*   **GitHub link placement:**  The primary GitHub link now is integrated into the introduction.
*   **Simplified Getting Started:** Simplified the steps to get started to make it easier for the user.