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

## PostHog: The Open-Source Platform for Product Success 🚀

[PostHog](https://github.com/PostHog/posthog) is a powerful, open-source product analytics platform designed to help you build and grow successful products.  Offering a suite of tools, PostHog empowers you to understand user behavior, track performance, and make data-driven decisions.

**Key Features:**

*   **Product Analytics:** Understand user behavior with event-based analytics, visualization, and SQL querying.
*   **Web Analytics:** Track website traffic, conversions, web vitals, and revenue with an easy-to-use dashboard.
*   **Session Replays:** Watch real user sessions to diagnose issues and understand user interactions.
*   **Feature Flags:** Safely release features to select users with feature flags.
*   **Experiments:** Test changes and measure their impact with no-code A/B testing.
*   **Error Tracking:** Identify and resolve bugs quickly with error tracking and alerts.
*   **Surveys:** Collect user feedback with no-code survey templates.
*   **Data Warehouse:** Integrate with external tools and query data alongside your product analytics.
*   **Data Pipelines:** Transform and send data to 25+ tools in real-time or batch.
*   **LLM Observability:** Monitor traces, generations, and cost for your LLM-powered applications.

**Get Started:**

PostHog offers a generous free tier.  Sign up for [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup) to begin.

**Table of Contents**

*   [PostHog: The Open-Source Platform for Product Success](#posthog-the-open-source-platform-for-product-success-)
*   [Key Features](#key-features)
*   [Getting Started with PostHog](#getting-started-with-posthog)
    *   [PostHog Cloud (Recommended)](#posthog-cloud-recommended)
    *   [Self-hosting the open-source hobby deploy (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
*   [Setting up PostHog](#setting-up-posthog)
*   [Learning More About PostHog](#learning-more-about-posthog)
*   [Contributing](#contributing)
*   [Open-source vs. Paid](#open-source-vs-paid)
*   [We're hiring!](#were-hiring)

## Getting Started with PostHog

### PostHog Cloud (Recommended)

The easiest and most reliable way to get started with PostHog is by signing up for a free account on [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup). Enjoy a free tier with 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 1500 survey responses each month.

### Self-hosting the open-source hobby deploy (Advanced)

For self-hosting, deploy a hobby instance with Docker (4GB memory recommended):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Note: Open-source deployments scale to approximately 100k events per month. For larger deployments, migrate to [PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud).

See the [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer) for more information.

## Setting up PostHog

Integrate PostHog with your project using the [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), our [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks), or the [API](https://posthog.com/docs/getting-started/install?tab=api).

SDKs are available for:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Explore comprehensive documentation and guides for [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

Learn more in our [product docs](https://posthog.com/docs/product-os), including [product analytics](https://posthog.com/docs/product-analytics/capture-events), [web analytics](https://posthog.com/docs/web-analytics/getting-started), [session replays](https://posthog.com/docs/session-replay/how-to-watch-recordings), [feature flags](https://posthog.com/docs/feature-flags/creating-feature-flags), [experiments](https://posthog.com/docs/experiments/creating-an-experiment), [error tracking](https://posthog.com/docs/error-tracking/installation#setting-up-exception-autocapture), [surveys](https://posthog.com/docs/surveys/installation), [data warehouse](https://posthog.com/docs/cdp/sources), and more.

## Learning More About PostHog

Dive deeper into PostHog with our open-source [company handbook](https://posthog.com/handbook), detailing our [strategy](https://posthog.com/handbook/why-does-posthog-exist), [ways of working](https://posthog.com/handbook/company/culture), and [processes](https://posthog.com/handbook/team-structure).

Discover how to maximize your PostHog experience with our guide: [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled). Learn to [measure activation](https://posthog.com/docs/new-to-posthog/activation), [track retention](https://posthog.com/docs/new-to-posthog/retention), and [capture revenue](https://posthog.com/docs/new-to-posthog/revenue).

## Contributing

We welcome contributions! Help us build a better platform:

*   Suggest new features or get early access to beta functionality on our [roadmap](https://posthog.com/roadmap).
*   Submit a [PR](https://github.com/PostHog/posthog/blob/master/CONTRIBUTING.md).
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md).

## Open-source vs. Paid

This repository is licensed under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory (with its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)).

For a completely FOSS version, check out our [posthog-foss](https://github.com/PostHog/posthog-foss) repository.

Transparent pricing is available on our [pricing page](https://posthog.com/pricing).

## We’re hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

Looking for a great team to join? We are growing rapidly and would love for you to join us!  Check out our [careers page](https://posthog.com/careers).
```

Key improvements and optimizations:

*   **SEO-friendly title and description:** Added a clear, concise title and a brief description that includes relevant keywords like "open-source", "product analytics", and key features.
*   **Clear Headings:** Organized the content with clear, descriptive headings and subheadings for readability and SEO.
*   **Key Features in Bullet Points:**  Used bullet points to highlight the main features, making it easy for users to scan and understand the platform's capabilities.
*   **Concise Language:** Streamlined the language to be more direct and engaging.
*   **Call to Actions:**  Encouraged users to take specific actions (e.g., sign up, explore docs, contribute).
*   **Links:**  Included links to key resources such as the documentation, community, roadmap, and signup pages, as well as a link back to the original repo.
*   **Markdown Formatting:**  Correctly formatted the text using Markdown for improved visual appeal.
*   **Emphasis on Open Source:**  Highlighted the open-source nature of the platform.
*   **Visual Appeal:** Included the PostHog logo and a relevant image to enhance the visual presentation.
*   **Hiring Callout:**  Kept the hiring section to encourage potential candidates.
*   **Removed Unnecessary Content:** Removed redundant or less important information.