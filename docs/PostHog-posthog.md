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

## PostHog: The Open-Source Product Analytics Powerhouse 🚀

[PostHog](https://posthog.com/) is a comprehensive, open-source platform designed to empower you to build successful products by providing a complete suite of tools in one place. **This repository contains the source code for PostHog.**

**Key Features:**

*   **Product Analytics:** Understand user behavior with event-based analytics, autocapture, data visualization, and SQL querying.
*   **Web Analytics:** Monitor web traffic, track user sessions, and analyze key metrics like conversion, web vitals, and revenue with a GA-like dashboard.
*   **Session Replays:**  Watch real user sessions to diagnose issues and understand user interaction.
*   **Feature Flags:**  Safely roll out new features to specific users or groups.
*   **Experiments:** Test changes and measure their impact on key metrics using no-code experimentation.
*   **Error Tracking:** Monitor errors, receive alerts, and resolve issues to improve your product.
*   **Surveys:** Gather user feedback with customizable surveys.
*   **Data Warehouse & Pipelines:** Sync data from external tools and run custom filters/transformations on your data.
*   **LLM Analytics:** Capture key metrics like traces, generations, latency, and cost for your LLM-powered applications.

**Get Started Today!**

PostHog is free to use with a generous monthly free tier!  Sign up for [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup).

**Table of Contents**

*   [PostHog: The Open-Source Product Analytics Powerhouse](#posthog-the-open-source-product-analytics-powerhouse)
*   [Key Features](#key-features)
*   [Getting Started with PostHog](#getting-started-with-posthog)
    *   [PostHog Cloud (Recommended)](#posthog-cloud-recommended)
    *   [Self-hosting the Open-Source Hobby Deploy (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
*   [Setting Up PostHog](#setting-up-posthog)
*   [Learning More About PostHog](#learning-more-about-posthog)
*   [Contributing](#contributing)
*   [Open-Source vs. Paid](#open-source-vs-paid)
*   [We're Hiring!](#were-hiring)

## Getting Started with PostHog

### PostHog Cloud (Recommended)

The fastest and easiest way to get started is by signing up for a free account on [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup). Your first 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 1500 survey responses are free every month.

### Self-hosting the Open-Source Hobby Deploy (Advanced)

For self-hosting, deploy a hobby instance with Docker on Linux:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Open-source deployments are suitable for approximately 100k events per month. Consider migrating to [PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud) for larger needs.

**Important:**  We do not provide customer support or guarantees for self-hosted deployments. Refer to our [self-hosting documentation](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer).

## Setting Up PostHog

After setting up your PostHog instance, integrate it into your product using our [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), one of our [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks), or our [API](https://posthog.com/docs/getting-started/install?tab=api).

SDKs are available for:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Explore our docs for more on [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

Once you've installed PostHog, consult our [product documentation](https://posthog.com/docs/product-os) to learn more about [product analytics](https://posthog.com/docs/product-analytics/capture-events), [web analytics](https://posthog.com/docs/web-analytics/getting-started), [session replays](https://posthog.com/docs/session-replay/how-to-watch-recordings), [feature flags](https://posthog.com/docs/feature-flags/creating-feature-flags), [experiments](https://posthog.com/docs/experiments/creating-an-experiment), [error tracking](https://posthog.com/docs/error-tracking/installation#setting-up-exception-autocapture), [surveys](https://posthog.com/docs/surveys/installation), [data warehouse](https://posthog.com/docs/cdp/sources), and beyond.

## Learning More About PostHog

Explore our open-source resources:

*   [Company Handbook](https://posthog.com/handbook): Delve into our [strategy](https://posthog.com/handbook/why-does-posthog-exist), [culture](https://posthog.com/handbook/company/culture), and [processes](https://posthog.com/handbook/team-structure).
*   [Winning with PostHog Guide](https://posthog.com/docs/new-to-posthog/getting-hogpilled): Learn to measure activation, retention, and revenue.

## Contributing

We welcome contributions of all sizes!

*   Vote on features or get early access on the [roadmap](https://posthog.com/roadmap).
*   Submit a Pull Request (see instructions on [local development](https://posthog.com/handbook/engineering/developing-locally)).
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md).

## Open-Source vs. Paid

This repository is licensed under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), with exceptions for the `ee` directory (see its [license](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)).

For 100% FOSS, check out our [posthog-foss](https://github.com/PostHog/posthog-foss) repository.

View our [pricing](https://posthog.com/pricing) for transparent paid plan details.

## We’re Hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

Interested in joining our team?  We're growing and [would love for you to join us](https://posthog.com/careers)!
```

Key improvements and explanations:

*   **SEO Optimization:**  Included relevant keywords like "open-source product analytics," "web analytics," "session replay," "feature flags," and more.  Used these keywords naturally in headings and sentences.
*   **Hook:**  Added a concise, attention-grabbing one-sentence hook at the beginning.
*   **Clear Structure with Headings:**  Organized the README with clear headings and subheadings for easy navigation.
*   **Bulleted Key Features:**  Used bullet points to highlight the main features, making them easy to scan.
*   **Concise Summary:** The overview is more concise and focused on the core value proposition.
*   **Call to Action:** Encouraged the user to sign up.
*   **Emphasis on Open Source:** Explicitly highlighted the open-source nature of PostHog throughout.
*   **Clearer "Getting Started" Section:** Simplified the initial setup steps.
*   **SDK Language Clarity** added a few visual breaks to make the SDK language more readable.
*   **Updated the Hiring section:** Added a more enticing image.
*   **Backlink to Original Repo:**  Explicitly stated that this is the source code repo at the beginning.
*   **Table of Contents:** Added a table of contents for easy navigation.
*   **Revised and summarized:**  Simplified language and removed repetitive information to improve readability.
*   **Clearer distinctions between open-source and paid**: Highlighted the license differences.