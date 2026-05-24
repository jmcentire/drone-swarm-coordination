# Retrieve GCP bench run (drone-swarm-bench-1)

VM launched 2026-05-23, project `baton-dev-jmc`, zone `us-central1-a`.
Auto-shuts on completion. Results stream to GCS.

## Bucket layout
`gs://baton-dev-jmc-drone-bench/run-<TIMESTAMP>/`
- `manifest.json` — git_sha, start time, instance metadata (written first)
- `S*.json` — per-scenario bench output
- `S*.log` — per-scenario stdout/stderr
- `bench_results_combined.json` — aggregated JSON written at end
- `startup.log` — full startup-script log
- `DONE.json` — sentinel written only on clean completion

## Check status (any of these)
```bash
# list runs and contents
gcloud storage ls -r gs://baton-dev-jmc-drone-bench/run-*/

# is the VM still running?
gcloud compute instances list

# tail the live startup log on the VM
gcloud compute ssh drone-swarm-bench-1 --zone=us-central1-a -- tail -n 200 /var/log/startup.log

# read serial console (no SSH needed)
gcloud compute instances get-serial-port-output drone-swarm-bench-1 --zone=us-central1-a | tail -n 300
```

## Pull results to laptop
```bash
RUN=$(gcloud storage ls gs://baton-dev-jmc-drone-bench/ | grep run- | sort | tail -1)
gcloud storage cp -r "${RUN}*" ~/Code/drone_swarm/distributed/gcp_results/
```

## Teardown if anything is hung (manual safety)
```bash
gcloud compute instances delete drone-swarm-bench-1 --zone=us-central1-a --quiet
```

## Cost ceiling
n2-highcpu-16 on-demand: ~$0.57/hr. Hard ceiling: $20.
Expected: $0.50-$1.50 for one full bench run (~30-90 min wall).
