# Limitations and residual risks

These limitations define what the current repository's results can support. A successful benchmark run does not remove them.

## Data and external validity

- The only included measurements are simulated NASA C-MAPSS FD001-FD004 trajectories. There are no flight records, maintenance logs, independent fleets, or N-CMAPSS data in the repository.
- The four subsets share simulator provenance. Leave-one-subset-out scores measure a shift among related simulated conditions, not transfer to an unseen aircraft fleet.
- C-MAPSS has no maintenance action, cost, repair duration, downtime, overhaul date, or post-service outcome. Fractional-life thirds are not observed maintenance phases.
- The included train/test partitions and one fixed validation-engine split provide limited evidence about variation across engines, data sources, and time.
- Dataset and paper files are included alongside source code. The repository's MIT `LICENSE` is not a blanket license for those third-party materials. Check their source-specific terms before redistributing them; the repository does not establish a separate license for every bundled asset.

## Model and evaluation

- A 125-cycle cap and 30-cycle history are fixed defaults. Performance may change with different caps, histories, preprocessing, seeds, or engine splits.
- The primary results snapshot is one seeded run. The reliability snapshot repeats Ridge over three seeds, while LSTM and several challenger results use one seed; single-run challengers do not establish a stable ranking.
- Primary benchmark metrics aggregate overlapping windows, which are statistically dependent. Reliability reports terminal-engine results separately. Neither is a substitute for the other, and window-sum PHM08 penalties are not the official per-engine challenge score.
- Validation-calibrated split-conformal intervals have descriptive measured test coverage. Dependence, changing conditions, and domain shift can invalidate nominal coverage; an interval is not a per-engine probability statement or safety guarantee.
- Test-set evaluation is retrospective. Repeatedly exploring the same published test results can bias later research choices even when the scripted pipeline selects candidates using validation data.
- Sensor/PCA correlation is descriptive, not causal. Fractional-life assignment uses retrospective test lifetime information. The onset label is derived from capped RUL and is not a physical fault onset.
- Sensor corruption and reduced-history experiments are controlled synthetic interventions. Their rates and severity do not estimate real sensor failure prevalence.
- The maintenance what-if analysis uses analyst-selected urgency cutoffs and service capacities. It is a sensitivity exercise, not a validated maintenance policy.

## Application and deployment

- The React/Vite application is a static viewer of committed JSON, CSV, and image artifacts. It does not serve predictions or automatically refresh when ignored local experiment outputs change; regenerate the bundle and commit it to update the published snapshot.
- The Streamlit forecast tab uses Ridge artifacts for uploaded histories, even when an LSTM model was trained for comparison. The displayed interval is derived from validation residuals and can under-cover on uploaded data from another domain.
- A successful app build does not verify current Vercel or Streamlit settings, network routes, secrets, uptime, authentication, or the deployed revision.
- Streamlit's upload parser processes user-provided CSV/TXT in memory. The repository does not implement a multi-tenant trust boundary or confidential-data workflow; do not upload private operational records to a public deployment.
- Docker is a local Streamlit packaging path, not a hardened production image. The repository does not provide a production API, identity management, fleet database, rate limits, or an aircraft system integration.

## Optional tools and environment

- TensorFlow is required for LSTM training and unavailable in the repository's Python 3.14 deployment configuration. Ridge and static result views remain available without it; Python 3.12 is the documented full training environment.
- The C++ extension is optional, compiler-dependent, and limited to PCA/K-Means diagnostics. Its timing results are host-specific and do not imply faster RUL prediction.
- Julia is needed to regenerate the maintenance scenario table. The committed table lets hosted apps display it without Julia.
- ManimGx is needed only to render a new animation; the included video is a static artifact.
- The current tests exercise helpers and dashboard data behavior. They do not run every full multi-subset training and reliability study on every commit.

## Security and verification limits

- A repository text scan found no matches for common token/private-key patterns; this is not a complete secret-scanning guarantee.
- Python `pip-audit --local` timed out while retrieving advisories from PyPI, and the npm advisory endpoint reset its connection. Neither dependency advisory scan produced a result; no claim is made that the dependencies are vulnerability-free.
- No external credential, production setting, or hosted-service configuration was inspected. Review GitHub, Vercel, and Streamlit account configuration separately before relying on them.
- No physical hardware, flight system, or live maintenance workflow was available or tested.
