# Security policy

Thanks for taking the time to help keep `fg-data-profiling` (formerly
`ydata-profiling`, formerly `pandas-profiling`) and its users secure.

## Supported versions

Security fixes are applied to the **latest minor release** on PyPI and
to the project's `develop` branch. Earlier minors are best-effort only.

| Version line  | Supported       |
|---------------|-----------------|
| `develop`     | yes             |
| latest 4.x    | yes             |
| older 4.x     | best-effort     |
| 3.x and older | no longer maintained |

The current supported Python range is declared in
[`pyproject.toml`](./pyproject.toml).

## Reporting a vulnerability

**Please do NOT open a public GitHub issue or pull request for security
problems.** Reports filed publicly give attackers a head-start before a
patched release is available.

You have two private channels:

1. **GitHub Security Advisory** — preferred. Use
   [Security Advisories → "Report a vulnerability"](https://github.com/Data-Centric-AI-Community/fg-data-profiling/security/advisories/new)
   on this repository. GitHub provides an end-to-end encrypted thread
   with the maintainers and lets us coordinate a fix + CVE without
   exposing the report.
2. **Email** — send a description to `opensource@ydata.ai`. Encrypt with
   the maintainers' public keys if possible. Include "fg-data-profiling
   security" in the subject line so the report is routed correctly.

Please include, at minimum:

- a clear description of the issue and its impact (what an attacker can
  do, on which versions);
- a minimal reproducer or proof-of-concept;
- the affected component (e.g. HTML report rendering, Spark backend,
  serialization, dependency surface);
- any suggested mitigation or patch you already have in mind.

## What to expect

- **Initial reply within 5 business days** acknowledging receipt and
  routing the report to the right maintainer.
- A coordinated-disclosure timeline — typically 30–90 days depending on
  the severity and complexity of the fix.
- A patched release on PyPI plus a public advisory once the fix lands.
  Reporters who wish to be credited will be named in the advisory and
  the release notes.
- If we determine the report is not a vulnerability (e.g. an unsupported
  configuration, a third-party dependency issue best filed upstream),
  we will explain that decision and, where useful, point you at the
  right place to file it.

## Out of scope

The following are not vulnerabilities in this project on their own:

- Bugs in unsupported versions (see the table above).
- Issues that require an attacker to already have local code execution
  or filesystem write access on the user's machine.
- Findings that depend on running the library on **untrusted dataframes
  or HTML output that the user has explicitly chosen to render in their
  browser**. The HTML report is meant to be rendered locally by trusted
  users; sharing a generated report with an untrusted party is the
  user's responsibility.
- Vulnerabilities in transitive dependencies — please file those upstream
  with the appropriate project. We are happy to review pinning changes
  here once a fixed version is published upstream.

## Hall of fame

Security researchers credited in past advisories will be listed here as
those advisories are published.

---

*This file responds to issue
[#1499](https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/1499).
GitHub recommends every public OSS repository ship a `SECURITY.md`
([documentation](https://docs.github.com/en/code-security/getting-started/adding-a-security-policy-to-your-repository))
so that vulnerability reports have a clear, private channel.*
