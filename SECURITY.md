# Security policy

## Scope

This repository is a research and demonstration project. The supported security scope is the checked-in source code, dependency manifests, build workflows, and static dashboard artifacts. The project is not a production aircraft system, authenticated prediction API, or approved place to store confidential fleet data.

## Reporting a vulnerability

Please do not publish credentials, personal data, private telemetry, or an exploitable vulnerability in a public issue. If GitHub private vulnerability reporting is available for this repository, use the repository's **Security → Report a vulnerability** flow. Otherwise, contact the maintainer through the public GitHub profile to request a private reporting channel before sharing technical details.

Reports should include the affected file or dependency, impact, reproduction steps that do not expose real data, and a suggested mitigation if known. The maintainer will acknowledge reports when able; this project does not currently promise a response or patching SLA.

## Maintainer guidance

- Revoke and rotate any credential committed by mistake; deleting it in a later commit does not invalidate it.
- Keep GitHub Actions permissions to the minimum required. The repository CI workflow requests read-only `contents` access and does not publish artifacts or releases.
- Review dependency and workflow updates before merging. Do not auto-upgrade packages as a substitute for reviewing compatibility.
- Keep operational or confidential sensor data out of commits, dashboard exports, and public issue attachments.
- Treat uploaded CSV/TXT as untrusted input. The Streamlit application parses files in memory and does not provide a hardened multi-tenant security boundary.
- Review exported JSON and reports for local paths or identifying information before committing them.

## Current known limitations

The Streamlit app accepts user-uploaded measurement files and is intended for trusted demonstration use. The static web dashboard has no prediction API or account system. Authentication, rate limiting, hosted-service configuration, and the security posture of external Vercel/Streamlit accounts are outside this repository's implementation.

The audit's text scan found no matches for common credential/private-key patterns, but it was not a full secret-scanning product. Python and npm advisory databases could not be queried successfully during the audit; see [`docs/limitations.md`](docs/limitations.md). No dependency changes were made based on an unavailable advisory scan.
