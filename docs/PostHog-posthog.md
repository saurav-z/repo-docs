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


## PostHog: The Open-Source Product Platform for Growth & Success 🚀

**PostHog** is the all-in-one open-source product platform helping developers build successful products, providing everything you need for product analytics, session replays, feature flags, and more.  [Explore the original repo on GitHub](https://github.com/PostHog/posthog).

### Key Features:

*   **Product Analytics:** Understand user behavior with event-based analytics, visualization, and SQL capabilities.
*   **Web Analytics:**  Monitor website traffic, user sessions, conversion, web vitals, and revenue with a GA-like dashboard.
*   **Session Replays:** Watch real user sessions to diagnose issues and understand user behavior.
*   **Feature Flags:**  Safely roll out features with feature flags, targeting specific users or cohorts.
*   **Experiments:** A/B test changes and measure impact on goal metrics, also with no-code support.
*   **Error Tracking:** Track errors, receive alerts, and resolve issues to improve your product.
*   **Surveys:** Gather valuable user feedback using no-code survey templates or a custom survey builder.
*   **Data Warehouse:** Sync data from various sources like Stripe, Hubspot, and your data warehouse.
*   **Data Pipelines (CDP):** Transform and send data to 25+ tools or webhooks in real-time or batch.
*   **LLM Observability:** Capture traces, generations, latency, and cost for your LLM-powered applications.

Get started with a **generous monthly free tier** on [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup).

### Table of Contents

-   [PostHog: The Open-Source Product Platform for Growth & Success 🚀](#posthog-the-open-source-product-platform-for-growth--success-)
    -   [Key Features:](#key-features)
    -   [Table of Contents](#table-of-contents)
    -   [Getting started with PostHog](#getting-started-with-posthog)
        -   [PostHog Cloud (Recommended)](#posthog-cloud-recommended)
        -   [Self-hosting the open-source hobby deploy (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
    -   [Setting up PostHog](#setting-up-posthog)
    -   [Learning more about PostHog](#learning-more-about-posthog)
    -   [Contributing](#contributing)
    -   [Open-source vs. paid](#open-source-vs-paid)
    -   [We’re hiring!](#were-hiring)

### Getting Started with PostHog

#### PostHog Cloud (Recommended)

The fastest and easiest way to get started is by signing up for a free account on [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup). Enjoy a generous free tier that includes 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 1500 survey responses per month.

#### Self-hosting the open-source hobby deploy (Advanced)

For self-hosting, deploy a hobby instance on Linux with Docker (4GB memory recommended) using this command:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Self-hosted deployments are suitable for around 100k events per month. For higher volumes, migrating to [PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud) is recommended.

Note: Open-source deployments do not come with customer support or guarantees.  Refer to the [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer) for more details.

### Setting Up PostHog

Integrate PostHog into your project by installing the [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), one of our [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks), or through our [API](https://posthog.com/docs/getting-started/install?tab=api).

We offer SDKs and libraries for a variety of languages and frameworks:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Additional documentation and guides are available for [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

Once PostHog is installed, explore the [product docs](https://posthog.com/docs/product-os) to learn more about [product analytics](https://posthog.com/docs/product-analytics/capture-events), [web analytics](https://posthog.com/docs/web-analytics/getting-started), [session replays](https://posthog.com/docs/session-replay/how-to-watch-recordings), [feature flags](https://posthog.com/docs/feature-flags/creating-feature-flags), [experiments](https://posthog.com/docs/experiments/creating-an-experiment), [error tracking](https://posthog.com/docs/error-tracking/installation#setting-up-exception-autocapture), [surveys](https://posthog.com/docs/surveys/installation), [data warehouse](https://posthog.com/docs/cdp/sources), and more.

### Learning More About PostHog

Our commitment to openness extends to our [company handbook](https://posthog.com/handbook), where you can explore our [strategy](https://posthog.com/handbook/why-does-posthog-exist), [ways of working](https://posthog.com/handbook/company/culture), and [processes](https://posthog.com/handbook/team-structure).

To get the most out of PostHog, check out our guide,  [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled), which covers [measuring activation](https://posthog.com/docs/new-to-posthog/activation), [tracking retention](https://posthog.com/docs/new-to-posthog/retention), and [capturing revenue](https://posthog.com/docs/new-to-posthog/revenue).

### Contributing

We welcome your contributions!

*   Vote on features or get early access to beta functionality on our [roadmap](https://posthog.com/roadmap).
*   Submit a PR (see instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally)).
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md).

### Open-source vs. paid

This repository uses the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory (which uses its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)).

Need 100% FOSS? Check out our [posthog-foss](https://github.com/PostHog/posthog-foss) repository, which is free of proprietary code and features.

Find out about our transparent pricing on [our pricing page](https://posthog.com/pricing).

### We’re hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

If you've read this far, you might be a great fit for our team.  We're growing rapidly [and would love for you to join us](https://posthog.com/careers)!
```

Key improvements and optimizations:

*   **SEO-friendly Heading:** Added a clear, concise, and keyword-rich heading:  "PostHog: The Open-Source Product Platform for Growth & Success"
*   **One-Sentence Hook:** Created a compelling opening sentence to immediately grab the reader's attention.
*   **Keyword Integration:** Naturally incorporated relevant keywords like "open-source," "product platform," "product analytics," "session replays," and "feature flags."
*   **Bullet Points:** Reorganized the "Key Features" section for easy readability and scannability.
*   **Clear Structure:** Used headings and subheadings to improve organization and readability, mirroring the original's structure but enhancing it.
*   **Actionable Links:** Kept the important links and added descriptive text for better SEO and user experience.  Specifically called out the link back to the original repo.
*   **Concise Language:** Streamlined the language to make it more engaging and easier to understand.
*   **Call to Action:** Included a clear call to action to sign up and a link to the careers page.
*   **Alt Text:** Improved the `alt` text for the images.
*   **Added description:** More information in the description.