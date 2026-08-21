# Prospective train-only classical comparator protocol (21 August 2026)

Status: **FROZEN PRE-RUN PROTOCOL RECORD. At the protocol-freeze timestamp, no
real comparator lock had been fitted and no validation or locked-retrospective
result existed. Subsequent execution state is recorded by the content-addressed
fit lock and the separate execution ledger; this pre-run record is not rewritten
after fitting.**

Protocol ID: `classical_train_only_comparators_v1`

## Purpose and separation from the neural screen

This protocol adds three task-matched classical comparators for the canonical
three-class problem: C0 isolated/disconnected pores, C1 connected pores, and C2
mineral matrix. The comparators are external references. They do not join the
five-candidate neural validation screen, cannot select the neural winner, and
cannot change the neural formulation, architecture, or checkpoint.

These are **new comparators**, not claimed reproductions of the eight pipelines
reported by Alwan et al. (2025). The published study masked the mineral matrix
and evaluated a two-pore-class formulation. Its code was not released and its
method table does not specify enough parameters for an exact independent
reimplementation.

The recovered local `scripts/traditional_algorithms_comparison.py` is ignored
and retained as historical source material only; it is not part of the public
script boundary. It cannot supply publication results because:

1. it reads masks encoded as `0/1/255` but evaluates them as `0/1/2`, forcing
   mineral precision, recall, and IoU to zero;
2. it selects the first eight files returned by an unsorted directory traversal
   rather than the canonical source-group split;
3. its saved numbers do not match the companion paper's table;
4. several comments say thresholds were "tuned" without recording the data or
   selection rule; and
5. the recovered hybrid script calls method names that do not exist and uses a
   different three-method ensemble from the published hybrid.

Archived classical metrics and rankings are therefore excluded from all claims.

### Recovery boundary for the companion study

The [official companion article](https://doi.org/10.5194/adgeo-67-79-2025)
names Watershed, Distance Transform, Local Contrast, Canny edge,
Morphological Gradient, Gabor, global Otsu, and Refined Morphology methods. It
reports a local-contrast sweep from -0.5 to 0.2 in steps of 0.1; describes the
refined rule as round 5--100 pixel-squared objects with at least 80% grey
perimeter; and defines its hybrid as a two-of-three vote over Distance
Transform, Watershed, and Local Contrast. However, its published analyses
collapse the phase interpretation to isolated versus connected pore and ignore
matrix pixels. Source code is unavailable, and the article does not specify
the complete implementation state needed to recreate predictions exactly.

The local historical script is a separate heuristic implementation. Its exact
recovered operations are:

| Local method | Recovered operations and C0 rule |
| --- | --- |
| Watershed | gate `<100`; opening disk 2; closing disk 3; distance peaks `min_distance=10`, `threshold_abs=5`; region area `<500` |
| Refined morphology | gate `<100`; opening/closing disks 1/2, 3/4, and 5/6; union; component area `<300` (the computed `<70` mask is unused) |
| Distance transform | gate `<100`; opening disk 2; component area `<400` and maximum distance `>8` |
| Edge based | Canny 30/100 plus Sobel 85th and Laplacian 90th percentiles; fill/intersect gate; opening disk 2; closing disk 3; area `<250` |
| Local contrast | adaptive-mean windows 15 and 31 with `C=51`; 21-pixel local standard deviation; 75th contrast and 30th intensity percentiles; opening disk 2; closing disk 4; area `<350` and solidity `>0.8` |
| Morphological gradient | disk radii 2, 4, and 6; 80th percentile; fill/union gate; opening disk 2; closing disk 3; remove `<50`; area `<400`, eccentricity `<0.8`, extent `>0.6` |
| Otsu | four-class multi-Otsu two-darkest union single Otsu; opening disk 2; closing disk 4; remove `<100`; area `<450`, aspect ratio `<3` |
| Gabor | orientations 0/90 degrees and frequencies 0.1/0.3; energy 40th, standard-deviation 70th, mean 30th percentiles; gates `<100`/`<120`; opening disk 1; closing disk 3; remove `<80`; area `<380` and bounding-box texture below the global 35th percentile |

These details document why the archive cannot be used as evidence; they do not
promote the script to a comparator. In addition to the label and split defects,
its reconstructed hybrid calls non-existent method names and combines Refined
Morphology, Otsu, and Local Contrast rather than the published trio.

## Data access boundary

Comparator fitting and parameter choice use only the 74 filenames matching the
four `train_series` declarations in `config/confirmatory_splits.json`. The
manifest remains whole and hashed, while the frozen training filename/byte and
mask-byte attestations authenticate the exact discovered set. The fitter does
not resolve validation or test paths on disk, construct those datasets, or open
their image/mask bytes. It deliberately does not open the shared COCO annotation
index because that JSON container also holds non-training polygon records.

The four filename-derived training series are the grouped-CV units:

- `pdo1_12`;
- `pdo1_7`;
- `pdo4_1_140721`; and
- `pdo4_2_151020`.

The data owner must still confirm whether these identifiers are independent
mosaics or specimens. The last two series contain very few retained training
tiles, so train-only fold estimates are method-development diagnostics, not
uncertainty estimates or publication performance.

Before fitting, the script must match the frozen split, training-image, and
training-mask SHA-256 attestations. It must not claim to have matched the
annotation index, which it does not read. The generated lock must state zero
shared-annotation-index, validation, and held-out reads. Any required attestation
mismatch is a hard failure. The same train-only filename, grouping, image-byte,
mask-byte, and split-manifest attestations are recomputed after fitting and must
still match before the immutable lock directory can be published.

## Inference boundary shared by B0, B1, and B2

Every comparator receives one native `2048 x 2048` clean uint8 greyscale tile.
An RGB PNG is accepted only if all three channels are exactly identical. This
rejects yellow-ring overlays instead of silently converting them to greyscale.
No ring image, ring mask, target mask, filename series, tile coordinate, or
neural prediction is an inference feature.

The recovered operational pore gate is frozen as:

```text
pore = raw_uint8_intensity < 100
```

Every pixel outside this gate is C2. B0, B1, and B2 distinguish only C0 from C1
inside it. This makes their C2/pore-union result definition-driven rather than a
learned generalisation result, exactly as for the neural C2 conditional
candidates. The manuscript must state this limitation and must not select or
rank methods using overall accuracy.

## B0: small connected components

1. Compute the fixed pore gate.
2. Label 8-connected gate components on the full tile.
3. Assign a component to C0 when its area is strictly below a selected cutoff;
   otherwise assign it to C1.
4. Assign all pixels outside the gate to C2.

The prespecified cutoff grid is `25, 50, 100, 200, 300, 500, 1000` pixels. Each
cutoff is scored separately on each held-out training series. Selection uses the
mean, across the four training series, of

```text
S_pore = 2 * IoU_C0 * IoU_C1 / (IoU_C0 + IoU_C1 + 1e-8).
```

An exact tie is resolved by the larger minimum of mean C0 and C1 IoU, then the
smaller cutoff. C2 IoU and overall accuracy cannot select the cutoff.

B0 is intentionally weak and interpretable. A component can contain both C0
and C1 pixels, so one uniform label per component is not expected to solve the
task.

## B1: marker-watershed regions

1. Compute the same fixed pore gate without erosion, dilation, opening, or
   closing, preserving its pore-union boundary.
2. Compute the Euclidean distance transform inside the gate.
3. Detect marker maxima with minimum distance 10 pixels, absolute distance 5
   pixels, and border exclusion disabled.
4. Add one distance-maximum marker to every 8-connected gate component that did
   not receive a marker.
5. Apply 8-connected marker watershed to negative distance, constrained to the
   fixed gate.
6. Assign a watershed region to C0 when its area is strictly below the selected
   cutoff; otherwise assign it to C1.

B1 uses the same prespecified cutoff grid and train-group selection rule as B0.
All gate pixels must receive a watershed region; an unassigned pixel is a hard
failure.

## B2: pore-gated ExtraTrees

B2 is the single supervised classical comparator. It uses one prespecified
`ExtraTreesClassifier` rather than a post-result model search:

- 128 trees;
- Gini criterion;
- maximum depth 24;
- minimum leaf size 8;
- `sqrt` features per split;
- no bootstrap;
- balanced class weights;
- random state `20260821`; and
- deterministic training-pixel sampling seed `20260821`.

Up to 4096 in-gate C0 pixels and 4096 in-gate C1 pixels are sampled per training
tile without replacement. The sample seed is derived from the fixed global
seed, image ID, and class ID by SHA-256. No target-derived spatial feature is
computed.

The explicitly frozen clean-image feature vector is:

1. raw greyscale scaled to `[0,1]`;
2. Gaussian intensity at sigma 1;
3. Gaussian intensity at sigma 3;
4. 7-pixel local mean;
5. 7-pixel local standard deviation;
6. 31-pixel local mean;
7. 31-pixel local standard deviation;
8. Sobel gradient magnitude;
9. Laplacian of Gaussian at sigma 1;
10. log-scaled distance inside the fixed pore gate;
11. 15-pixel local pore-gate fraction; and
12. log-scaled 8-connected pore-gate component area.

Leave-one-training-series-out models are fitted for a train-only stability
diagnostic. Fold scoring uses a deterministic stratified pixel sample with
inverse sampling weights and includes mineral pixels that fall inside the fixed
gate. This diagnostic does not choose among RF/ExtraTrees variants: only the
single configuration above is eligible. The final model is then fitted once on
the sampled pixels from all 74 training tiles. The lock records both the exact
numeric-NPZ byte digest and a deterministic digest of the complete ordered
fitted-tree state. The latter is unchanged by container compression or
reserialization. The NPZ contains numeric arrays only and is loaded with
`allow_pickle=False`; no Python object graph is deserialized.

## Freeze artifact and later scoring

`scripts/fit_classical_comparators.py --campaign-id <execution-id>` uses the
campaign only as execution provenance. After fitting, it derives a
campaign-independent scientific identity from the full train-only protocol,
data, source, runtime, selection records, and fitted-tree semantic digest. It
then exclusively creates the canonical directory
`results/classical_comparators/locked/classical-fit-<identity-prefix>/` and
writes:

- `B2_extra_trees.npz`; and
- `classical_comparator_lock.json`.

The JSON records non-null source hashes, dependency versions, all 74 canonical
training filenames and their stable sampling IDs, data attestations, grouped
training diagnostics, selected B0/B1 cutoffs, the full B2 configuration and
feature list, exact model-byte and semantic-state digests, and explicit zero-read counters for the shared
annotation index, validation data, and held-out data. It also contains a
machine-readable predictor/callable schema for the later independent one-pass
evaluator. Absolute private paths are not emitted.

The lock records its public-path-safe execution campaign but excludes that
label and its timestamp from the scientific identity. It also binds the immutable
held-out execution schema, the canonical 21-tile `pdo2_24` scope, native tile
shape, split/input/target attestations, whole-tile bootstrap seed and replicate
count, and an exact numeric-artifact digest. Its source-hash map covers the comparator,
fitter, separate evaluator, shared metric/diagnostic implementation, AIRE
fit/evaluation wrappers, data contract, and this protocol. Missing or null
source hashes are a hard failure.

The fitter records exact Python, NumPy, Pillow, SciPy, scikit-image,
scikit-learn, OpenCV, and Matplotlib versions. The held-out preflight
requires an exact version match before loading the B2 artifact or reserving an
evaluation. This makes a changed `pore-seg` environment a recorded hard failure
rather than an implicit model re-serialization or dependency upgrade.

The canonical training filename list contains exactly 74 entries and has
SHA-256 `62c797c4486d6f6007610d2bda8e2c8522ea6bf3b8862c3ea2c3d0fbd8ff7d87`.
The evaluator checks the full source-controlled list as well as this digest,
the training-image aggregate, and the training-mask aggregate; a self-consistent
but alternate filename list is therefore rejected.

On AIRE, `scripts/aire_fit_classical_comparators.slurm` is the sole supported
fit/freeze wrapper. It requires one active, non-array CPU allocation on the
`nodes` partition, runs the fitter exactly once, and fixes the manifest, clean
image, lossless mask, and derived output paths. It does not accept an annotation
index. If no explicit `PORE_CLASSICAL_CAMPAIGN_ID` is supplied, the safe
execution identifier is `classical-fit-<SLURM_JOB_ID>`. The printed
`classical-fit-...` content ID, not the execution campaign, is the only
identifier accepted later by the evaluator. Refitting under another caller
campaign, changing a timestamp, reformatting JSON, or reserializing an identical
model cannot create a second canonical scientific fit.

Only after this lock exists and passes focused tests may the independent locked
evaluator load each comparator and score it unchanged. Comparator development
must not inspect validation or held-out outputs. In addition, the classical
evaluator requires a content-addressed neural-freeze manifest built from the
verified 15-cell selected-method lock and all six successful validation-only
selected-retraining outcomes/checkpoints: three primary and three plain U-Net
cells at seeds 42, 123, and 2025. The manifest builder opens no corpus bytes and
requires zero held-out construction/evaluation in every checkpoint. A neural
held-out result is deliberately not required. Build it only after both
selected-retraining arrays finish:

```bash
python3 scripts/build_neural_freeze_manifest.py \
  --primary-campaign-id <PRIMARY_RETRAIN_JOB_ID> \
  --plain-campaign-id <PLAIN_RETRAIN_JOB_ID>
```

The builder derives the sole output path under
`results/neural_freeze/locked/neural-freeze-<identity-prefix>/` and rejects
partial, substituted, non-canonical, traversing, or symlinked artifacts. Retain
the printed manifest ID for the held-out wrapper. The eventual one-pass report
must use the same native-tile C0 IoU, C1 IoU, balanced pore IoU, precision,
recall, Dice, and structural diagnostics as the neural models. Overall accuracy
is not a comparator outcome and cannot support a ranking or superiority claim.

If B1 cannot run with the declared scikit-image dependency or B2 cannot complete
within the bounded CPU/memory envelope, the failure is recorded. Neither method
may be replaced by a new post-result algorithm or a narrowed parameter grid.

## Exactly-once held-out evaluator

The held-out wrapper is a distinct later job. It must not be submitted until
the neural method selection and selected-winner retraining campaigns have been
frozen. The only supported classical held-out command is:

```bash
export PORE_CLASSICAL_FIT_ID=<classical-fit-identity>
export PORE_NEURAL_FREEZE_ID=<neural-freeze-identity>
sbatch scripts/aire_locked_classical_evaluation.slurm
```

The wrapper requires an active, non-array Slurm allocation on the fixed `nodes`
partition. It accepts no data, model, lock, checkpoint, output, threshold,
cutoff, device, comparator, fit path, or neural-freeze-path override. It accepts
only the content-addressed classical-fit and neural-freeze IDs, requires their
canonical lock/model/manifest paths, and independently rebuilds the neural
manifest from the selected-method lock and six retraining artifacts before
reservation. One invocation loads the single frozen
B2 numeric NPZ artifact and evaluates B0, B1, and B2 together on all 21 canonical
`pdo2_24` tiles at `2048 x 2048` resolution.

Before discovering a held-out filename or opening any held-out byte, the
evaluator authenticates the canonical content-addressed fit path and scientific
identity, its execution-campaign provenance, the canonical
neural-freeze manifest and its selected-method/six-checkpoint hashes, train-only data
attestations and filename identities, exact source hashes, model-byte and
semantic tree-state hashes, ExtraTrees schema, split-manifest metadata, fixed held-out corpus attestations,
and the complete evaluator contract. The shared annotation index is deliberately
not an input and is not opened. The protocol therefore makes no claim that this
evaluator matched that index. Validation identifiers exist in hashed split
metadata, but no validation image or target path is constructed or opened.

Only after preflight succeeds does the evaluator atomically reserve the fixed
pair directory
`results/classical_evaluation/locked/<classical-fit-id>/<neural-freeze-id>/`.
The evaluator has no output-root or lock-path override. A new caller campaign,
copied lock/model, JSON reformat, timestamp edit, symlink, or identical model
reserialization resolves to the same fit identity or fails canonical-path
verification, so it cannot reserve another pass for the same frozen pair.
Separate exclusive byte-read claims are created for held-out inputs and targets
before their first bytes are read. Any crash, hash mismatch, partial read, or
output failure consumes the attempt; changing an execution campaign is not a
retry mechanism.

The evaluator checks exact test filename equality between inputs and targets,
the count of 21, the `pdo2_24` prefix, raw-byte aggregate hashes, clean greyscale
input, lossless `0/1/255` target encoding, and native shape. Each byte payload is
cached from its single attestation read and consumed once. Hard-label outputs
are restricted to canonical C0/C1/C2 values.

The prospective report contains aggregate and per-tile JSON/CSV evidence for
C0 and C1 IoU, Dice, precision and recall, their balanced harmonic IoU, pore
union IoU, confusion matrices, fixed-gate/reference discrepancies, and the same
2-D area/component diagnostics used by the neural evaluator where applicable.
Uncertainty is the fixed deterministic percentile bootstrap over whole native
tiles. Publication outputs include vector and raster confusion figures, a C0/C1
IoU interval figure, and a lexicographically preselected qualitative tile.
Overall accuracy, calibrated probability curves, post-result tuning, winner
ranking, and effects on neural selection are explicitly excluded.

## Verification and blocker matrix

| Comparator | Current verification | Remaining gate before freeze |
| --- | --- | --- |
| B0 | strict gate, strict area inequality, full-gate/C2 composition, and canonical-label tests pass | run the declared train-group selection in the complete environment |
| B1 | deterministic full-gate/fallback-marker test passes in the declared AIRE environment with scikit-image 0.25.2 | run the declared train-only fit in its canonical campaign |
| B2 | feature shape/finiteness, deterministic sampling, gate composition, and invalid-output tests pass | complete grouped training diagnostics and model serialization in the declared AIRE environment |

The real 74-training-tile preflight currently matches the frozen input and mask
attestations and yields training-series counts of 32, 39, 1, and 2. A synthetic
contract test also succeeds when validation/test identifiers exist in manifest
metadata but their image and mask files do not exist, demonstrating that only
training bytes are required. The focused comparator/evaluator/two-wrapper suite
plus the neural-freeze/screen-selection authentication tests report 83 passes
and one local scikit-image dependency skip. A fresh source-only, no-corpus
public-snapshot copy in the declared AIRE `pore-seg` environment reports 252
passes and two intentional corpus-absence skips. The public-snapshot audit and
synthetic public smoke test pass. The full CI job installs the declared PyTorch,
OpenCV, and scikit-image dependencies before running the same suite.
