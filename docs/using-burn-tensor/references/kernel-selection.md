# Kernel Selection

## Goal

Shape code so Burn's backend and autotuning machinery can choose strong kernels for the actual hardware and workload.

## Shape guidance

Kernel choice is hardware- and shape-dependent. As a user, the biggest lever is often the tensor shape itself.

Prefer when practical:

- dimensions that are multiples of 8
- dimensions that are multiples of 32
- powers of two for hot inner dimensions

Be cautious with:

- shapes like `[1000, 1000]`
- long propagation of uneven dimensions through many layers

If a suboptimal shape is unavoidable, it can be better to pay the cost once, transform into a more backend-friendly shape, and keep later layers regular.

## Autotune and cold start

Burn may benchmark multiple kernels on first use to choose an implementation for the current hardware and shape pattern.

Therefore:

- expect first-run latency on new machines or new shape patterns
- do not benchmark only the very first run and call it representative
- consider bundling autotune caches in deployment environments where cold start matters

## Practical review questions

- Are hot inner dimensions awkward when they could be regularized?
- Is a cold-start benchmark being mistaken for steady-state performance?
- Would a one-time reshape or padding step unlock better downstream kernels?

## Common mistake

### "Autotune means shape does not matter"

Autotune helps choose among available kernels, but poor shapes can still limit vectorization and throughput.
