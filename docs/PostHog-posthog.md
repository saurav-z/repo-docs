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

## Supercharge Your Product with PostHog: The Open-Source Analytics & Product Platform

**PostHog** is an all-in-one, open-source platform that provides the tools you need to build, launch, and scale successful products. 
[Explore the PostHog Repo](https://github.com/PostHog/posthog)

### Key Features:

*   **Product Analytics**: Understand user behavior with event-based analytics, autocapture, and SQL-based data analysis.
*   **Web Analytics**: Monitor website traffic, user sessions, conversion rates, web vitals, and revenue in a GA-like dashboard.
*   **Session Replays**: Watch real user sessions to diagnose issues and gain insights into user interactions.
*   **Feature Flags**: Safely roll out new features to specific user segments with feature flags.
*   **Experiments**: Test and measure the impact of changes using A/B testing with a no-code interface.
*   **Error Tracking**: Monitor errors, receive alerts, and resolve issues to improve product stability.
*   **Surveys**: Gather user feedback using pre-built survey templates or custom surveys.
*   **Data Warehouse**: Integrate with external tools like Stripe, Hubspot, and data warehouses to query data alongside your product data.
*   **Data Pipelines**: Transform and route your incoming data in real-time or batch, and send it to 25+ tools or any webhook.
*   **LLM Observability**: Track and analyze traces, generations, latency, and cost for your LLM-powered applications.

**Free Tier:** PostHog offers a generous [free tier](https://posthog.com/pricing) to get you started!

## Getting Started

### PostHog Cloud (Recommended)

The fastest and most reliable way to get started is by signing up for [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup). Your first 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 1500 survey responses are free every month, with usage-based pricing thereafter.

### Self-hosting (Advanced)

For self-hosting, deploy a hobby instance with Docker (Linux, 4GB memory recommended):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Open-source deployments are recommended up to approximately 100k events per month. For larger volumes, we recommend [migrating to PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud). Please see the [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer) for more info.

## Setting Up PostHog

Integrate PostHog by using the [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), one of [our SDKs](https://posthog.com/docs/getting-started/install?tab=sdks), or our [API](https://posthog.com/docs/getting-started/install?tab=api).

**SDKs available for:**

*   **Frontend:** JavaScript, Next.js, React, Vue
*   **Mobile:** React Native, Android, iOS, Flutter
*   **Backend:** Python, Node, PHP, Ruby, Go, .NET/C#, Django, Angular, WordPress, Webflow, and more.

For more detailed setup instructions, see our [product docs](https://posthog.com/docs/product-os).

## Learning More

Explore the open-source [company handbook](https://posthog.com/handbook) for insights into our strategy, work culture, and processes.

Get started with our [guide to winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled), and learn how to measure activation, retention, and revenue.

## Contributing

Contribute to PostHog and help us build the future of product analytics!

*   **Roadmap:** [Vote on features or get early access](https://posthog.com/roadmap)
*   **PRs:**  Open a PR (see our instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally))
*   **Feature requests/bug reports:**  [Feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md)

## Open-source vs. Paid

This repo is licensed under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory which has its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE).

For a completely FOSS option, check out our [posthog-foss](https://github.com/PostHog/posthog-foss) repository.

[Pricing](https://posthog.com/pricing) for our paid plan is transparent.

## We’re Hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

If you're a dedicated README reader, consider joining our team. We're hiring and [would love for you to join us](https://posthog.com/careers).
```

Key improvements and SEO considerations:

*   **Strong Hook:**  A compelling one-sentence introduction to grab attention.
*   **Clear Headings:**  Uses headings to structure the information logically for readability and SEO.
*   **Keyword Optimization:** Includes relevant keywords like "open-source analytics," "product analytics," "web analytics," and specific feature names, increasing search visibility.
*   **Bulleted Key Features:** Highlights the core benefits and functionalities using bullet points.
*   **Concise Summaries:**  Provides brief descriptions of each feature, keeping the content focused.
*   **Calls to Action:** Includes clear calls to action like "Explore the PostHog Repo" and links to get started.
*   **Updated Links:** All links are functional and relevant.
*   **Visual Appeal:**  Maintains the original logo and uses images to make it more engaging.
*   **Markdown formatting:** Properly formatted for easy reading on GitHub.
*   **Hiring Section:** Includes a short and engaging section for job seekers.
*   **Concise:** Keeps the content focused.