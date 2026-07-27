# Slurm GSV metric jobs

`run_gsv_metric.sbatch` is a self-contained, resumable metric job. It requests
one node with 12 tasks, 40 GB RAM, and one GPU. On the default `ws-ia`
partition, the human-perception, SAM3, and SAM servers share the node's single
GPU. The job waits for all three services, runs one metric, and stops the
servers when it exits.

The default `ws-ia` partition advertises one GPU per node, so two one-GPU jobs
cannot be placed on the same node and their fixed localhost ports 8111, 8005,
and 8004 cannot collide. Do not point these jobs at a multi-GPU partition
without either requesting exclusive nodes or configuring job-specific ports.

Do not put the OpenRouter key in these files. Export it in the submission shell;
`sbatch --export=ALL` captures the environment at submission time, including
for delayed jobs.

Run one immediate smoke test using the first `boring` image:

```bash
cd /l/users/hosam.elgendy/VIDA-GEO
read -rsp "OpenRouter API key: " OPENROUTER_API_KEY; echo
export OPENROUTER_API_KEY
./slurm/submit_gsv_smoke.sh boring
```

The smoke test writes separately to `outputs/gsv_smoke/` and does not alter the
full benchmark CSV.

Submit the default pair (`beautiful` and `wealthy`) so they become eligible nine
hours later:

```bash
cd /l/users/hosam.elgendy/VIDA-GEO
export OPENROUTER_API_KEY='your-key'
./slurm/submit_two_gsv_metrics.sh
```

Choose another pair or delay:

```bash
./slurm/submit_two_gsv_metrics.sh boring depressing now+9hours
```

Submit only one:

```bash
./slurm/submit_gsv_metric.sh beautiful now+9hours
```

Submit only the next pair. By default this submits `boring` and `depressing`,
which allows the promoted boring smoke result to resume as the first completed
boring input:

```bash
./slurm/submit_remaining_gsv_metrics.sh
```

After those finish, submit the second pair:

```bash
./slurm/submit_remaining_gsv_metrics.sh beautiful wealthy now+9hours
```

This wrapper never submits more than two jobs at once.

The final target is 100 total results per metric, including reused prior runs.
Accordingly, Slurm runs 91 new images for `safety` and `depressing` (plus 9
legacy results each), and 90 new images for the other four metrics (plus 10
legacy results each). The replacement images under
`benchmark_images/extra_images` are retained as a reserve and are not part of
these default runs.

Optional cluster settings can be exported before submission:

```bash
export VIDA_SLURM_PARTITION='partition-name'
export VIDA_SLURM_ACCOUNT='account-name'
export VIDA_SLURM_QOS='qos-name'
export VIDA_SLURM_CONSTRAINT='constraint-name'
```

Each metric writes to the shared GSV output root but uses its own
`benchmark_results_<metric>.csv`. Job and server logs are under `slurm_logs/`.
Rerunning or resubmitting the same metric skips completed job summaries.
