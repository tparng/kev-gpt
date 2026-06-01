"""Stage 3 — the fabric throughput core (transposed wider-word banked GEMV) on
the road to ~100k tok/s. Breaks the PE=64 one-bank-per-lane ceiling by storing
weights so one wide URAM word feeds LANES lanes sharing the activation x[k]."""
