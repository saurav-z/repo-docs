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

PostHog is a comprehensive, open-source product analytics platform designed to empower product teams with the tools they need to build and launch successful products.  **[Explore the PostHog repository on GitHub](https://github.com/PostHog/posthog)**.

**Key Features:**

*   **Product Analytics:** Understand user behavior with event-based analytics, including autocapture and manual instrumentation. Analyze data with visualizations and SQL.
*   **Web Analytics:** Monitor web traffic, user sessions, and essential metrics like conversion, web vitals, and revenue with an intuitive dashboard.
*   **Session Replays:**  Watch real user sessions to diagnose issues and gain insights into user interactions on your website or mobile app.
*   **Feature Flags:**  Safely roll out features to specific user cohorts using feature flags.
*   **Experiments:** Test changes and measure their statistical impact on key metrics with no-code experiment setup.
*   **Error Tracking:** Track errors, receive alerts, and resolve issues to improve product stability and user experience.
*   **Surveys:** Collect valuable user feedback using our no-code survey templates or build custom surveys.
*   **Data Warehouse Integration:**  Sync data from tools like Stripe, HubSpot, and data warehouses, then query it alongside your product data.
*   **Data Pipelines (CDP):** Run custom filters and transformations on your incoming data. Send it to 25+ tools, webhooks, or batch-export large amounts of data to your warehouse.
*   **LLM Observability:** Capture traces, generations, latency, and cost for your LLM-powered applications.

**Free Tier:**

Enjoy a generous free tier with PostHog, allowing you to get started without any upfront cost. Sign up for [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup) to begin!

## Table of Contents

*   [PostHog: The Open-Source Product Analytics Platform for Building Successful Products](#posthog-the-open-source-product-analytics-platform-for-building-successful-products)
*   [Key Features](#key-features)
*   [Getting Started with PostHog](#getting-started-with-posthog)
    *   [PostHog Cloud (Recommended)](#posthog-cloud-recommended)
    *   [Self-hosting the open-source hobby deploy (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
*   [Setting up PostHog](#setting-up-posthog)
*   [Learning More about PostHog](#learning-more-about-posthog)
*   [Contributing](#contributing)
*   [Open-source vs. paid](#open-source-vs-paid)
*   [We’re hiring!](#were-hiring)

## Getting Started with PostHog

### PostHog Cloud (Recommended)

The easiest and most reliable way to get started is by signing up for free on [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup). Benefit from a generous free tier, including your first 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 250 survey responses monthly.

### Self-hosting the open-source hobby deploy (Advanced)

If you'd like to self-host, you can deploy a hobby instance in a single line on Linux with Docker:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Self-hosted deployments are designed to handle up to 100k events per month. After this, we recommend migrating to [PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud).

**Important:** Open-source deployments do not include customer support or guarantees.  Consult the [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer) for detailed information.

## Setting up PostHog

Once you have a PostHog instance, you can integrate it into your project using:

*   Our [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet)
*   One of our [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks)
*   Or by using our [API](https://posthog.com/docs/getting-started/install?tab=api).

We offer SDKs and libraries for many popular languages and frameworks:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Additional resources and guides are available for [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

After installing PostHog, consult our [product docs](https://posthog.com/docs/product-os) to set up [product analytics](https://posthog.com/docs/product-analytics/capture-events), [web analytics](https://posthog.com/docs/web-analytics/getting-started), [session replays](https://posthog.com/docs/session-replay/how-to-watch-recordings), [feature flags](https://posthog.com/docs/feature-flags/creating-feature-flags), [experiments](https://posthog.com/docs/experiments/creating-an-experiment), [error tracking](https://posthog.com/docs/error-tracking/installation#setting-up-exception-autocapture), [surveys](https://posthog.com/docs/surveys/installation), [data warehouse](https://posthog.com/docs/cdp/sources), and more.

## Learning More about PostHog

Our commitment to open-source extends beyond the code; explore our open-source [company handbook](https://posthog.com/handbook) to learn about our [strategy](https://posthog.com/handbook/why-does-posthog-exist), [ways of working](https://posthog.com/handbook/company/culture), and [processes](https://posthog.com/handbook/team-structure).

Discover how to maximize your PostHog experience with our guide on [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled), which covers the basics of [measuring activation](https://posthog.com/docs/new-to-posthog/activation), [tracking retention](https://posthog.com/docs/new-to-posthog/retention), and [capturing revenue](https://posthog.com/docs/new-to-posthog/revenue).

## Contributing

We highly value contributions of all sizes:

*   Vote on features and gain early access to beta features on our [roadmap](https://posthog.com/roadmap)
*   Submit a pull request (PR) following the instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally)
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md)

## Open-source vs. paid

This repository is available under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), excluding the `ee` directory, which is subject to its own [license](https://github.com/PostHog/posthog/blob/master/ee/LICENSE).

For a completely free and open-source experience, see our [posthog-foss](https://github.com/PostHog/posthog-foss) repository, which excludes any proprietary code or features.

Transparency is key, and our [pricing page](https://posthog.com/pricing) provides a clear overview of our paid plan costs.

## We’re hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

If you've made it this far, you clearly appreciate the details!

Join our growing team! We're actively expanding and would love for you to become part of the PostHog team! [Apply for a role today](https://posthog.com/careers).
```
Key improvements and explanations:

*   **SEO Optimization:** Includes keywords like "open-source product analytics," "product analytics platform," and highlights key features to improve search engine visibility.  Also added a short description.
*   **Clear Headings:**  Uses clear and concise headings and subheadings (H2 and H3) to organize the information, making it easy to scan.
*   **Bulleted Key Features:** Uses bullet points to emphasize and highlight the key benefits and features of PostHog.
*   **Concise Language:**  Rewrites sentences to be more direct and easier to understand.
*   **One-Sentence Hook:** The first sentence immediately introduces PostHog and its purpose, grabbing the reader's attention.
*   **Improved "Getting Started" Section:**  Clarifies the two main ways to get started, with a strong recommendation for PostHog Cloud. It also provides a warning and links for the self-hosting option.
*   **Emphasis on Free Tier:**  Highlights the generous free tier to attract users.
*   **SDK & Library improvements:** Improved and organized the SDK and library tables.
*   **Calls to Action:** Encourages users to sign up, contribute, and apply for jobs.
*   **Table of Contents:** Adds a table of contents for easy navigation.
*   **Links to Docs & Resources:**  Keeps important links but cleans them up, adding anchor links to sections for better navigation.
*   **Cleaned up the hire me section:** Added a small intro with a call to action.
*   **License information** Added a section to describe the license and how it affects proprietary code.