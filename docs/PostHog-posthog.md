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

## PostHog: The Open-Source Platform for Building Successful Products

**PostHog is your all-in-one, open-source platform empowering product teams to build and grow successful products.**  [Explore the original repository](https://github.com/PostHog/posthog).

**Key Features:**

*   **Product Analytics:** Understand user behavior with event-based analytics, including autocapture, data visualization, and SQL querying.
*   **Web Analytics:**  Monitor website traffic, track conversions, and analyze key metrics like web vitals and revenue, similar to Google Analytics.
*   **Session Replays:**  Visualize user interactions with session recordings to diagnose issues and enhance user experience.
*   **Feature Flags:** Safely roll out new features and test them with feature flags, targeting specific user segments.
*   **Experiments (A/B Testing):**  Run A/B tests and measure the statistical impact of your changes on key metrics.
*   **Error Tracking:** Identify and resolve errors quickly to improve your product's stability.
*   **Surveys:** Gather valuable user feedback with no-code survey templates and custom survey creation.
*   **Data Warehouse & CDP:** Integrate data from external sources for unified analysis and data transformation.
*   **LLM Observability:** Monitor the performance of your LLM-powered applications.

Best of all, PostHog offers a [generous free tier](https://posthog.com/pricing) so you can get started quickly. Sign up for [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup) today!

---

## Table of Contents

-   [PostHog: The Open-Source Platform for Building Successful Products](#posthog-the-open-source-platform-for-building-successful-products)
-   [Table of Contents](#table-of-contents)
-   [Getting Started with PostHog](#getting-started-with-posthog)
    -   [PostHog Cloud (Recommended)](#posthog-cloud-recommended)
    -   [Self-hosting the open-source hobby deploy (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
-   [Setting up PostHog](#setting-up-posthog)
-   [Learning more about PostHog](#learning-more-about-posthog)
-   [Contributing](#contributing)
-   [Open-source vs. paid](#open-source-vs-paid)
-   [We’re hiring!](#were-hiring)

---

## Getting Started with PostHog

### PostHog Cloud (Recommended)

The easiest and most reliable way to get started is with a free account on [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup). You get 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 250 survey responses free every month.

### Self-hosting the open-source hobby deploy (Advanced)

Deploy a self-hosted instance using Docker with this command (requires 4GB memory):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Self-hosted deployments are recommended for up to 100k events/month.  For larger volumes, consider [migrating to PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud).

_Note: Self-hosted deployments are not officially supported. See the [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer) for details._

## Setting up PostHog

Integrate PostHog by using our [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), our [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks), or the [PostHog API](https://posthog.com/docs/getting-started/install?tab=api).

SDKs and libraries are available for:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Find additional documentation and guides for: [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

After installation, consult our [product docs](https://posthog.com/docs/product-os) for detailed setup instructions for [product analytics](https://posthog.com/docs/product-analytics/capture-events), [web analytics](https://posthog.com/docs/web-analytics/getting-started), [session replays](https://posthog.com/docs/session-replay/how-to-watch-recordings), [feature flags](https://posthog.com/docs/feature-flags/creating-feature-flags), [experiments](https://posthog.com/docs/experiments/creating-an-experiment), [error tracking](https://posthog.com/docs/error-tracking/installation#setting-up-exception-autocapture), [surveys](https://posthog.com/docs/surveys/installation), [data warehouse](https://posthog.com/docs/cdp/sources), and more.

## Learning more about PostHog

Explore our open-source [company handbook](https://posthog.com/handbook) to learn about our [strategy](https://posthog.com/handbook/why-does-posthog-exist), [ways of working](https://posthog.com/handbook/company/culture), and [processes](https://posthog.com/handbook/team-structure).

For practical guidance, see our guide to [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled), which covers [measuring activation](https://posthog.com/docs/new-to-posthog/activation), [tracking retention](https://posthog.com/docs/new-to-posthog/retention), and [capturing revenue](https://posthog.com/docs/new-to-posthog/revenue).

## Contributing

We welcome contributions of all sizes:

*   Vote on features and get early access on our [roadmap](https://posthog.com/roadmap).
*   Submit a Pull Request (PR) - see instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally).
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md).

## Open-source vs. paid

This repository uses the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except the `ee` directory (see [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)).

For a completely open-source version, check out our [posthog-foss](https://github.com/PostHog/posthog-foss) repository.

View our transparent pricing at [our pricing page](https://posthog.com/pricing).

## We’re hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

If you're a dedicated README reader, you might make a great addition to our team! [Join us](https://posthog.com/careers) and help build the future of product development.
```

Key improvements and SEO optimizations:

*   **Clear Title & Introduction:**  Uses "PostHog: The Open-Source Platform..." for better keyword targeting.  The opening sentence is a hook.
*   **Keyword Integration:**  Strategically includes keywords like "open-source," "product analytics," "web analytics," "session replay," "feature flags," "A/B testing," "error tracking," and "data warehouse" throughout the description and key features.
*   **Feature Bullets:**  Uses clear, concise bullet points for readability and to highlight core functionality.
*   **Targeted Headings:** Uses descriptive headings (e.g., "Getting Started with PostHog Cloud").
*   **Internal Links:**  Includes links within the document to help users navigate the content better.
*   **External Links with Keywords:**  Links to the PostHog website and documentation are strategically placed with relevant anchor text.
*   **Alt Text for Images:** Improved alt text for image accessibility and SEO.
*   **Action-Oriented Language:**  Encourages users to "Explore," "Sign up," and "Join us."
*   **Focus on Benefits:**  Emphasizes the *benefits* of using PostHog (e.g., "empowering product teams to build and grow successful products").
*   **Concise Summarization:**  Condenses the original content without losing essential information.
*   **Call to Action:** Prominently features calls to action.
*   **Clear distinction of Free and Paid:** Clarifies both free and paid offerings.