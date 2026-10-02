"""Scoring methods: how SoftSeg decides which assignment of transcripts is best.

``evaluate_overlapping_regions`` resolves an ambiguous region by generating
candidate hard assignments and picking the highest-scoring one. *How* a
candidate is scored is the business of a **scoring method**, and this is where
they live.

A scoring method has two halves:

1. **A reference, built once** -- :meth:`ScoringMethod.prepare`. For the
   methods here that is the per-cell-type average expression matrix, but it is
   whatever a method needs computed up front: a classifier, a neighbourhood
   graph, a set of marker genes. Any preparation a method needs belongs here
   rather than in the resolve loop, which runs it per candidate.
2. **A score per candidate** -- :meth:`ScoringMethod.score`, called once for
   every candidate assignment of every ambiguous region. This is the hot path:
   a single FOV asks for it tens of thousands of times.

Between the two sits :meth:`ScoringMethod.region_context`, which is what lets a
method afford to look at a whole cell. A region's candidates differ only in
where the *ambiguous* transcripts go; the transcripts already settled in each
cell are the same for every candidate of that region. ``region_context`` folds
those into whatever form the method can add cheaply -- a running total, a gene vector --
once per region instead of once per candidate.

The methods here are one family, **scoring by cell type**
(:class:`CellTypeScoring`): they share that matrix and differ in how a cell's
transcripts are compared against its type's row.

- ``"additive matrix"`` -- the sum, over every transcript placed, of its gene's
  value in the receiving cell's type. Scored over the placed transcripts alone.
- ``"mse difference"`` -- the negative squared error between a cell's gene
  tally and its type's profile, over the whole cell: what is already settled
  in it plus what the candidate adds.

- ``"ml classifier"`` -- trains a small PyTorch network in ``prepare``
  to predict a cell's type from its gene counts, and scores the log-probability
  it gives each cell's own type, over the whole cell. Needs torch.

Another reading of the same matrix is another child of
:class:`CellTypeScoring` that says what one cell is worth (``score_cell``); a
child with a reference of its own, like the classifier, also overrides
``prepare`` and declares ``matrix_reference = False``.

Methods register themselves by name, and the name plus the parameters it was
given are recorded in ``sdata.attrs["scoring"][assigned_col]["scoring"]``,
alongside the evaluation's own arguments and the blur it read, so a column in a
store says how it was produced (``SoftAssigner.scoring_params``).

Writing another one
-------------------

Subclass :class:`ScoringMethod`, declare a ``name`` (and a ``region_context`` or
``use_qc_filter`` where the defaults do not suit), take whatever parameters the
method needs, and register it.
Nothing else has to change: the registry is how the pipeline finds it::

    @register_scoring_method
    class MarkerCountScoring(ScoringMethod):
        # Score by how many of the receiving cell type's markers land in it.

        name = "marker count"
        use_qc_filter = False      # a gene list; noise does not shape it

        def __init__(self, assigner, markers=None, weight=1.0):
            super().__init__(assigner)
            self.markers = markers or {}      # a parameter set of its own --
            self.weight = float(weight)       # nothing like the matrix method's
            self.by_type = None

        @property
        def params(self):
            # what lands in sdata.attrs["scoring"][col]["scoring"]; plain
            # values only
            return {"markers": self.markers, "weight": self.weight}

        def prepare(self, adata=None, save=True):
            # the reference, built once. A real method fits its classifier or
            # reads its reference here; this one just indexes the lists.
            self.by_type = {t: frozenset(g) for t, g in self.markers.items()}
            self.prepared = True
            return self

        def score(self, assignment, context=None):
            gene_of = self.assigner.tr_to_gene
            total = 0.0
            for cell, transcripts in assignment.items():
                cell_type = self.assigner.cell_to_type.get(cell, "other")
                wanted = self.by_type.get(cell_type, frozenset())
                total += self.weight * sum(
                    1 for tr in transcripts if gene_of.get(tr) in wanted
                )
            return total

Then::

    asgn.set_scoring_method("marker count", markers={"T cell": ["CD3E", "CD8A"]})
    asgn.prepare_scoring()
    asgn.evaluate_all_overlapping_regions(assigned_col="by_markers")

A method that names ``region_context = "conf_trs"`` is handed
``{cell: [transcript id, ...]}`` of what is already in each cell, alongside the
candidate, and decides for itself what that is worth.
"""

from __future__ import annotations

from itertools import chain

import numpy as np
import pandas as pd

# "not passed", as distinct from an explicit None, which means "no filter here".
# Defined in this module rather than in SoftAssigner so a scoring method can
# reach it without importing the assigner it is scoring for.
UNSET = object()

# The registry. Keys are what a user passes to `SoftAssigner.set_scoring_method`
# and what lands in `attrs["scoring"][col]["scoring"]["method"]`, so they are stable
# names rather than class names.
SCORING_METHODS: dict = {}


def register_scoring_method(cls):
    """Add a `ScoringMethod` subclass to the registry, keyed by its `name`."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} needs a name to be registered under.")
    if cls.name in SCORING_METHODS and SCORING_METHODS[cls.name] is not cls:
        raise ValueError(
            f"{cls.name!r} is already registered to "
            f"{SCORING_METHODS[cls.name].__name__}."
        )
    SCORING_METHODS[cls.name] = cls
    return cls


def get_scoring_method(name, assigner, **params):
    """Build the registered scoring method called `name` over `assigner`."""
    if name not in SCORING_METHODS:
        raise KeyError(
            f"No scoring method named {name!r}. Available: "
            f"{sorted(SCORING_METHODS)}."
        )
    return SCORING_METHODS[name](assigner, **params)


class ScoringMethod:
    """Base class for a way of scoring a candidate assignment.

    A subclass declares `name`, takes its parameters in `__init__`, computes
    whatever reference it needs in `prepare`, and scores a candidate in `score`.

    The three pieces of state the resolve loop relies on:

    `name`
        The registry key, recorded in the sdata alongside the column it scored.
    `params`
        Everything that defines this scoring, as a dict of plain values. It goes
        into `sdata.attrs["scoring"][assigned_col]["scoring"]`, so it has to
        survive a round-trip through zarr's JSON: numbers, strings, bools, None,
        and lists or dicts of those. `table_name` and `source`, which say where
        a reference was read from rather than how it scores, are left out of
        the record (`SoftAssigner.UNRECORDED_SCORING_PARAMS`).
    `region_context`
        Which region context this method is scored against, by name -- a key of
        `SoftAssigner.REGION_CONTEXTS`, which is where the contexts themselves
        live. The default "new_trs" hands `score` the candidate's transcripts
        and nothing else; "conf_trs" adds what is already settled in each cell.
        **A declaration, not a run-time argument**: a method that measures a
        cell's profile needs all of it and a method that scores what is being
        moved does not, and neither answer changes between runs of one.
    `use_qc_filter`
        The same kind of declaration, for whether the table this method is
        prepared from should be QC-filtered first. The filtering is the
        assigner's (`SoftAssigner.scoring_table`); this only says whether to
        ask for it.
    """

    #: Registry key. A subclass must set this.
    name = None

    #: Which region context `score` is handed, by name -- a key of
    #: `SoftAssigner.REGION_CONTEXTS`. The default, "new_trs", is the
    #: transcripts the candidate is placing and nothing else; "conf_trs" adds
    #: everything already settled in the cell. A subclass declares this only
    #: when it wants something other than the default.
    region_context = "new_trs"

    #: Whether the cell-by-gene table this method is prepared from should have
    #: `SoftAssigner`'s QC bounds applied to it first. The filtering itself lives
    #: on the assigner (`filter_for_scoring`, `qc_bounds`), because it is a
    #: property of the dataset; whether a method wants it is a property of the
    #: method, and like `region_context` it does not vary between runs of one. A
    #: method that reads per-cell profiles wants the noisy cells gone; one that
    #: only needs a gene list does not care.
    use_qc_filter = False

    def __init__(self, assigner):
        self.assigner = assigner
        self.prepared = False

    def __repr__(self):
        ready = "prepared" if self.prepared else "not prepared"
        return f"{type(self).__name__}({self.name!r}, {ready}, {self.params})"

    @property
    def params(self) -> dict:
        """The parameters that define this scoring, for the attrs record."""
        return {}

    def prepare(self, **kwargs):
        """Compute the reference this method scores against.

        Called once before any region is resolved. Everything a method needs
        computed up front belongs here -- it is the one place that is allowed to
        be slow, since `score` runs tens of thousands of times per FOV.

        Must set `self.prepared = True` when it finishes.
        """
        raise NotImplementedError

    def require_prepared(self):
        """Raise unless `prepare` has been run."""
        if not self.prepared:
            raise RuntimeError(
                f"The {self.name!r} scoring method has not been prepared. Run "
                "SoftAssigner.get_scoring_matrix() (or set_scoring_method(...) "
                "then prepare_scoring()) before evaluating overlapping regions."
            )

    def score(self, assignment, context=None):
        """Score one candidate assignment. Higher is better.

        assignment (dict): `{cell: [transcript id, ...]}`, the ambiguous
            transcripts this candidate places. A cell id of `"other"` is the
            notional cell a one-way comparison is made against.
        context (dict): `{cell: [transcript id, ...]}` already settled in each
            cell, built once per region by the context this method named in
            `region_context`. Empty under the default "new_trs". A region's
            candidates differ
            only in where the *ambiguous* transcripts go, so this is the same
            for every candidate of the region and is built once for all of them.
        """
        raise NotImplementedError


class CellTypeScoring(ScoringMethod):
    """Score by cell type: how expected a cell's transcripts are for its type.

    The family of methods that share one reference -- a `{gene: {cell type:
    value}}` matrix holding the average expression of each gene over the cells
    of each type, from an annotated cell-by-gene table -- and differ only in how
    a cell's transcripts are compared against its type's row. This class builds
    that reference and walks the cells of a candidate; a child says what one
    cell is worth, in `score_cell`, and which region context it wants.

    Not registered itself: it has a reference but no way of scoring against it.
    The registered children are `AdditiveMatrixScoring` ("additive matrix") and
    `MseDifferenceScoring` ("mse difference"); a new reading of the same matrix
    is a third child, with nothing else to change.

    Parameters
    ----------
    cats
        Where the cell types are: ``{"obs column": ["type", ...]}`` in full, or
        just the column name, or None to find the one column that can hold them.
    table_name
        Which table in the sdata to read, when it holds more than one. Ignored
        when an `adata` is handed to `prepare` directly.
    normed
        Weight each type's row by how rare the type is, so a common type does not
        dominate by volume alone.

    The QC bounds that shape the table are not parameters here: `use_qc_filter`
    says this family wants them applied and `SoftAssigner.scoring_table` applies
    them, from what the sdata records or what the caller passed it. They belong
    to the dataset -- what counts as too few transcripts depends on the panel --
    rather than to the method.
    """

    #: The reference is an average profile per cell type, so a cell carrying
    #: mostly noise drags its type's row toward noise. Filter first.
    use_qc_filter = True

    #: The row of everything without a type of its own: the notional "other"
    #: cell a one-cell region is compared against, and any untyped cell. It is
    #: the average over every cell of the table, typed or not. An untyped cell
    #: only reaches scoring under `use_other_cells`; otherwise a region holding
    #: one is skipped before it is scored at all.
    OTHER = "other"

    def __init__(self, assigner, cats=None, table_name=None, normed=False):
        super().__init__(assigner)
        self.cats = cats
        self.table_name = table_name
        self.normed = bool(normed)

        # filled by prepare()
        self.score_mat = None       # {gene: {cell type: value}} -- the old shape
        self.genes = None           # gene name -> column in `matrix`
        self.types = None           # cell type -> row in `matrix`
        self.matrix = None          # (n types, n genes) float array
        self.other_row = None       # the row of "other", and of untyped cells
        self.source = None          # what was scored, for the record

    #: Whether `prepare` builds nothing beyond the shared matrix, so that a
    #: prepared reference can be handed to a sibling by `as_method`. A child that
    #: trains or fits something of its own says False.
    matrix_reference = True

    #: What `prepare` leaves behind, and all that a sibling needs to score
    #: against the same reference -- see `as_method`.
    REFERENCE_STATE = ("cats", "score_mat", "genes", "types", "matrix",
                       "other_row", "source", "prepared")

    @property
    def params(self) -> dict:
        """What this scoring was, in values that survive a trip through zarr."""
        cats = self.cats
        if isinstance(cats, dict):
            cats = {str(k): [str(v) for v in vals] for k, vals in cats.items()}
        elif cats is not None:
            cats = str(cats)
        return {
            "cats": cats,
            "table_name": self.table_name,
            "source": self.source,
            "normed": self.normed,
        }

    def as_method(self, name):
        """This reference, scored the way the cell-type method `name` scores.

        Every child of this class builds the same matrix from the same
        parameters, so switching between them needs no second `prepare`. The
        result is a new object; this one is not changed.
        """
        family = sorted(
            n for n, c in SCORING_METHODS.items()
            if issubclass(c, CellTypeScoring) and c.matrix_reference
        )
        if name not in family or not self.matrix_reference:
            raise KeyError(
                f"Cannot score {self.name!r}'s reference as {name!r}: only the "
                f"methods whose reference is the matrix alone can share one "
                f"({family})."
            )
        cls = SCORING_METHODS[name]
        if cls is type(self):
            return self
        other = cls(self.assigner, cats=self.cats, table_name=self.table_name,
                    normed=self.normed)
        for attr in self.REFERENCE_STATE:
            setattr(other, attr, getattr(self, attr))
        return other

    # ------------------------------------------------------------------ #
    # 1. the reference                                                    #
    # ------------------------------------------------------------------ #

    def prepare(self, adata=None, save=True):
        """Build the per-cell-type average expression matrix.

        adata: the annotated cell-by-gene table, already resolved and filtered
            -- `SoftAssigner.scoring_table` does both, and `prepare_scoring`
            calls it. Passing None asks for the same thing here, for the rare
            caller that reaches a method directly.
        save: whether the `{cell: type}` mapping this also builds is written back
            to the store as the `cell_to_type` table.
        """
        asgn = self.assigner
        if adata is None:
            self.source = self.table_name or "<default cxg table>"
            adata = asgn.scoring_table(table_name=self.table_name)
        else:
            self.source = self.table_name or "<adata passed in>"

        cats = self.cats
        if cats is None or isinstance(cats, str):
            cats = asgn.infer_cats(adata, column=cats)
        # a spelling of "no type" is not a type, even when named as one: those
        # cells are untyped, and scored against the OTHER row
        missing = getattr(asgn, "MISSING_TYPES", frozenset())
        cats = {
            key: [v for v in values if str(v) not in missing]
            for key, values in cats.items()
        }
        self.cats = cats

        all_avg = np.average(adata.X, axis=0)

        # Each type's rows are taken by index off the matrix. Subsetting the
        # AnnData and deep-copying the result, as this used to, duplicates the
        # whole expression matrix once per cell type to compute one mean of it.
        X = adata.X
        avgs = {}
        counts = {}
        for key, values in cats.items():
            column = adata.obs[key].to_numpy()
            for cat in values:
                rows = np.flatnonzero(column == cat)
                if not len(rows):
                    # the QC bounds can empty a type out; averaging nothing
                    # would put nan across its whole row
                    asgn.logger.info(
                        f"no cells left of type {cat!r} after filtering, so it "
                        "is left out of the scoring matrix."
                    )
                    continue
                avgs[cat] = np.average(X[rows], axis=0)
                counts[cat] = len(rows)

        avgs[self.OTHER] = all_avg
        counts[self.OTHER] = len(adata)

        df_avg = pd.DataFrame.from_dict(avgs, orient="index", columns=adata.var.index)
        if self.normed:
            norm_total = sum(counts.values())
            df_avg = df_avg.multiply(
                [(norm_total - n) / norm_total * 100 for n in counts.values()],
                axis=0,
            )
        df_avg = df_avg.fillna(0)

        # the dict form the old code used, kept because `score_dataset` and
        # anything a user wrote against `assigner.score_mat` reads it
        self.score_mat = df_avg.to_dict()

        # and the array form everything in here scores through: one float lookup
        # by (type index, gene index) instead of two dict lookups by name
        self.types = {t: i for i, t in enumerate(df_avg.index)}
        self.genes = {g: i for i, g in enumerate(df_avg.columns)}
        self.matrix = df_avg.to_numpy(dtype=float)
        self.other_row = self.types[self.OTHER]

        self._build_cell_to_type(adata, cats, avgs, save=save)
        self.prepared = True
        return self

    def _build_cell_to_type(self, adata, cats, avgs, save=True):
        """The `{cell: type}` mapping, built while the types are in hand.

        A convenience, not the only way the assigner gets cell types -- see
        `SoftAssigner.cell_to_type` -- but this method has already read the
        column and filtered the table, so it would be wasteful to make someone
        ask for them separately. Note that it types only the cells that survived
        that filtering, which is what makes a filtered-out cell untyped;
        `SoftAssigner.set_cell_types_from_table` reads the whole column instead.

        Written a type at a time rather than a cell at a time: `iterrows` builds
        a Series per cell, which on a real table is most of the cost. A cell
        matching more than one entry still ends up with the last one, since the
        loops run in the same order. The assigner normalises what it is handed,
        so this does not have to.
        """
        mapping = {}
        names = adata.obs_names.to_numpy()
        for key, cell_types in cats.items():
            column = adata.obs[key].to_numpy()
            for cell_type in cell_types:
                if cell_type not in avgs or cell_type == self.OTHER:
                    continue  # filtered out entirely, or not a type
                for name in names[column == cell_type]:
                    mapping[name] = cell_type
        self.assigner.cell_to_type = mapping
        # kept in the sdata so a later assigner over the same store can read the
        # cell types back rather than having to be handed them again
        self.assigner.save_cell_to_type(save=save)

    # ------------------------------------------------------------------ #
    # 2. scoring                                                          #
    # ------------------------------------------------------------------ #

    def _type_row(self, cell):
        """The matrix row a cell is scored against.

        Its type's row; the OTHER row for the notional "other" cell, for a cell
        with no type, and for one whose type the reference has no row for (one
        the QC emptied out).
        """
        cell_type = self.assigner.cell_to_type.get(cell)
        if cell_type is None:
            return self.other_row
        return self.types.get(cell_type, self.other_row)

    def score_cell(self, row, columns):
        """What one cell is worth. Higher is better.

        row (int): the matrix row of the cell's type.
        columns (list of int): the matrix column of each transcript's gene, one
            entry per transcript the cell is being scored over -- what the
            candidate places there, plus whatever the region context says is
            settled. Transcripts of a gene the matrix does not have (blanks, or
            anything the panel has that the table does not) are already left out.
        """
        raise NotImplementedError

    def score(self, assignment, context=None):
        """Sum `score_cell` over every cell this candidate involves.

        `context` is `{cell: [transcript id, ...]}` of what is already settled
        in each cell, from the region context this method names. A cell the
        candidate gives nothing but that holds settled transcripts still counts:
        what is in it is the same for every candidate of the region, but a
        method that compares a whole profile is not indifferent to it, and
        leaving it out would be asking a different question.
        """
        self.require_prepared()
        gene_of = self.assigner.tr_to_gene
        index = self.genes

        cells = assignment.keys()
        if context:
            held = [c for c, trs in context.items() if trs and c not in assignment]
            if held:
                cells = list(cells) + held

        total = 0.0
        for cell in cells:
            settled = context.get(cell, ()) if context else ()
            placed = assignment.get(cell, ())
            columns = [
                column
                for column in (index.get(gene_of.get(tr))
                               for tr in chain(settled, placed))
                if column is not None
            ]
            total += self.score_cell(self._type_row(cell), columns)
        return total


@register_scoring_method
class AdditiveMatrixScoring(CellTypeScoring):
    """Sum, over every transcript placed, of its gene's average in the cell's type.

    The method the package started with. An assignment scores better when it
    puts transcripts where that cell type usually has them, one transcript at a
    time.

    A cell is scored over the transcripts the candidate places in it, not over
    everything it already holds -- the default "new_trs" region context. Every
    candidate of a region places the same transcripts somewhere, so what is
    already settled would add the same to all of them and change nothing.

    Takes the parameters of `CellTypeScoring`: `cats`, `table_name`, `normed`.
    """

    name = "additive matrix"

    def score_cell(self, row, columns):
        if not columns:
            return 0.0
        # summed transcript by transcript in order, as it always has been, so
        # the floating-point total is the same one
        values = self.matrix[row]
        running = 0.0
        for column in columns:
            running += values[column]
        return running


@register_scoring_method
class MseDifferenceScoring(CellTypeScoring):
    """Negative squared error between a cell's gene tally and its type's profile.

    Asks that the cell's whole profile match its type's, rather than that each
    transcript be individually likely there. The profile in question is the
    *whole* cell, so this names the "conf_trs" region context: a cell is scored
    over everything already settled in it plus what the candidate adds, not over
    the added transcripts alone -- a handful of transcripts against an average
    cell's worth of expression would be all error, whichever cell they went to.

    The reference is the same matrix `AdditiveMatrixScoring` builds, from the
    same parameters (`cats`, `table_name`, `normed`); only the comparison
    differs. `as_method` switches one prepared reference between the two.
    """

    name = "mse difference"
    region_context = "conf_trs"

    def score_cell(self, row, columns):
        tally = np.bincount(columns, minlength=self.matrix.shape[1]).astype(float)
        # one vectorised pass over the panel, rather than a python loop over
        # every gene in the matrix for every cell of every candidate
        diff = self.matrix[row] - tally
        # smaller error is better and everything here treats higher as better.
        # Negating each cell and summing is the negation of the summed error,
        # exactly: what must not happen is the sign flipping between cells, as
        # it once did, so that a two-cell region scored `cell1 - cell2`.
        return -float(diff @ diff)


@register_scoring_method
class MLClassifierScoring(CellTypeScoring):
    """Log-probability that a classifier names each cell's own type.

    `prepare` trains a small PyTorch network on the QC-filtered cell-by-gene
    table to predict a cell's type from its gene counts. A candidate then
    scores, for every cell it involves, the log-probability the network gives
    that cell's *own* type from the transcripts the cell would hold. Summed, the
    score is the log of the joint probability that every cell would be
    classified correctly -- so an assignment is better when it leaves each cell
    looking more like what it was typed as.

    The network learns from whole cells, which is why this names the "conf_trs"
    region context: a cell is scored over everything already settled in it plus
    what the candidate adds, the same kind of input it was trained on.

    A cell with no type -- untyped, typed as something the QC emptied out, or
    the notional "other" cell -- has no correct label to score. It contributes
    the log-probability of whichever type the network finds likeliest, i.e. how
    clearly its transcripts look like *some* cell. Only reached under
    `use_other_cells`, as for every method.

    Input to the network is library-size normalised and log-transformed, the
    usual treatment of counts: `log1p(counts / total * scale)`, where `scale` is
    the median total of the training cells. A cell being scored is often a
    partial one, so training thins each cell's counts at random (`augment`) to
    show the network smaller, noisier versions of whole cells.

    After training the weights are copied out to numpy and the forward pass is
    done there: `score` runs tens of thousands of times per FOV, one cell at a
    time, and per-call overhead in torch would dominate a network this small.
    Results for a (type, transcript multiset) already seen are cached, since a
    region's candidates repeat the same cell contents often.

    Parameters
    ----------
    cats, table_name
        As for every cell-type method: where the types are, and which table.
    hidden
        Width of the hidden layer.
    epochs, batch_size, lr, weight_decay
        Training settings, for Adam.
    augment
        Thin each training cell's counts by a random fraction (binomially) each
        epoch, so the network sees partial cells.
    balance
        Weight the loss by inverse class frequency, so rare types are learned
        rather than ignored.
    val_fraction
        Share of cells held out to report accuracy on; 0 trains on all of them.
    seed
        For the split, the initialisation and the augmentation.
    device
        Where to train: "cpu", "cuda", or None for cuda when available. Scoring
        is always on the CPU, in numpy.

    Requires torch, which is an optional dependency (`pip install
    softseg[classifier]`), imported only when this method is prepared.
    """

    name = "ml classifier"
    region_context = "conf_trs"
    matrix_reference = False

    #: Cached (type row, transcript multiset) -> score entries, before the cache
    #: is emptied. Enough for a FOV's worth of repeats.
    CACHE_SIZE = 200_000

    def __init__(self, assigner, cats=None, table_name=None, hidden=64,
                 epochs=30, batch_size=256, lr=1e-3, weight_decay=1e-4,
                 augment=True, balance=True, val_fraction=0.1, seed=0,
                 device=None):
        super().__init__(assigner, cats=cats, table_name=table_name)
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.augment = bool(augment)
        self.balance = bool(balance)
        self.val_fraction = float(val_fraction)
        self.seed = int(seed)
        self.device = device

        # filled by prepare()
        self.classes = None       # class index -> cell type
        self.class_of_row = None  # matrix row -> class index, -1 for none
        self.scale = None         # the normalisation's target total
        self.weights = None       # (W1, b1, W2, b2), numpy, for scoring
        self.history = None       # training summary, for looking at
        self._cache = {}

    @property
    def params(self) -> dict:
        out = super().params
        del out["normed"]  # the matrix's setting; nothing here reads it
        out.update(
            hidden=self.hidden, epochs=self.epochs, batch_size=self.batch_size,
            lr=self.lr, weight_decay=self.weight_decay, augment=self.augment,
            balance=self.balance, val_fraction=self.val_fraction,
            seed=self.seed,
        )
        return out

    # ------------------------------------------------------------------ #
    # 1. the reference: a trained network                                 #
    # ------------------------------------------------------------------ #

    def prepare(self, adata=None, save=True):
        """Train the classifier on the annotated, QC-filtered table.

        The shared cell-type preparation runs first -- it settles `cats`, the
        gene order and the `{cell: type}` mapping, which is what the labels are
        read from, so the network learns exactly the types the evaluation
        steps will ask it about.
        """
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "The 'ml classifier' scoring method needs torch: "
                "pip install softseg[classifier]"
            ) from e

        asgn = self.assigner
        if adata is None:
            adata = asgn.scoring_table(table_name=self.table_name)
            source = self.table_name or "<default cxg table>"
        else:
            source = self.table_name or "<adata passed in>"
        super().prepare(adata=adata, save=save)
        self.source = source

        mapping = asgn.cell_to_type
        names = adata.obs_names.astype(str)
        labels = np.array([mapping.get(n) for n in names], dtype=object)
        keep = np.array([lab is not None for lab in labels])
        self.classes = sorted(set(labels[keep]))
        if len(self.classes) < 2:
            raise ValueError(
                f"A classifier needs at least two cell types to tell apart; the "
                f"table has {self.classes}."
            )
        class_index = {t: i for i, t in enumerate(self.classes)}

        X = adata.X[keep]
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        y = np.array([class_index[t] for t in labels[keep]], dtype=np.int64)

        totals = X.sum(axis=1)
        self.scale = float(np.median(totals[totals > 0])) if (totals > 0).any() else 1.0

        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(y))
        n_val = int(round(len(y) * self.val_fraction))
        val, train = order[:n_val], order[n_val:]

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        model = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], self.hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, len(self.classes)),
        ).to(device)

        weight = None
        if self.balance:
            freq = np.bincount(y[train], minlength=len(self.classes)).astype(float)
            weight = torch.tensor(
                np.where(freq > 0, freq.sum() / np.maximum(freq, 1) / len(freq), 0),
                dtype=torch.float32, device=device,
            )
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        optim = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        counts = torch.tensor(X, device=device)
        target = torch.tensor(y, device=device)
        gen = torch.Generator(device=device).manual_seed(self.seed)
        train_t = torch.tensor(train, device=device)

        losses = []
        model.train()
        for _ in range(self.epochs):
            perm = train_t[torch.randperm(len(train_t), generator=gen, device=device)]
            epoch_loss = 0.0
            for start in range(0, len(perm), self.batch_size):
                idx = perm[start:start + self.batch_size]
                batch = counts[idx]
                if self.augment:
                    # keep each transcript with a per-cell probability in
                    # [0.3, 1]: a partial cell, as one being scored often is
                    keep_p = torch.empty(
                        (len(idx), 1), device=device
                    ).uniform_(0.3, 1.0, generator=gen)
                    batch = torch.binomial(
                        batch, keep_p.expand_as(batch), generator=gen
                    )
                loss = loss_fn(model(self._normalise_t(batch)), target[idx])
                optim.zero_grad()
                loss.backward()
                optim.step()
                epoch_loss += float(loss) * len(idx)
            losses.append(epoch_loss / max(len(perm), 1))

        model.eval()
        val_acc = None
        if len(val):
            with torch.no_grad():
                val_t = torch.tensor(val, device=device)
                pred = model(self._normalise_t(counts[val_t])).argmax(dim=1)
                val_acc = float((pred == target[val_t]).float().mean())

        first, last = model[0], model[2]
        self.weights = tuple(
            t.detach().cpu().numpy().astype(np.float64)
            for t in (first.weight, first.bias, last.weight, last.bias)
        )
        # matrix row -> class, so score_cell can go from the row it is handed
        # to the label it is scoring; rows with no class (the OTHER row, and any
        # type the training cells did not include) are -1
        self.class_of_row = np.full(len(self.types), -1, dtype=np.int64)
        for cell_type, row in self.types.items():
            if cell_type in class_index:
                self.class_of_row[row] = class_index[cell_type]
        self._cache = {}
        self.history = {
            "n_train": int(len(train)), "n_val": int(len(val)),
            "classes": list(self.classes), "loss": losses, "val_accuracy": val_acc,
            "device": device,
        }
        asgn.logger.info(
            f"ml classifier: trained on {len(train)} cells, "
            f"{len(self.classes)} types, final loss {losses[-1]:.3f}"
            + (f", held-out accuracy {val_acc:.3f}" if val_acc is not None else "")
        )
        self.prepared = True
        return self

    def _normalise_t(self, counts):
        """`log1p(counts / total * scale)`, on a batch of torch counts."""
        import torch

        total = counts.sum(dim=1, keepdim=True)
        return torch.log1p(counts * (self.scale / torch.clamp(total, min=1.0)))

    # ------------------------------------------------------------------ #
    # 2. scoring                                                          #
    # ------------------------------------------------------------------ #

    def log_probs(self, columns):
        """The network's log-probability of each class, for one cell's genes.

        columns: the matrix column of each transcript's gene, one per
            transcript, as `score_cell` is handed them.
        """
        W1, b1, W2, b2 = self.weights
        tally = np.bincount(columns, minlength=W1.shape[1]).astype(float)
        total = tally.sum()
        x = np.log1p(tally * (self.scale / total)) if total else tally
        h = W1 @ x + b1
        np.maximum(h, 0, out=h)
        logits = W2 @ h + b2
        top = logits.max()
        return logits - (top + np.log(np.exp(logits - top).sum()))

    def score_cell(self, row, columns):
        key = (row, tuple(sorted(columns)))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        logp = self.log_probs(columns)
        label = self.class_of_row[row]
        value = float(logp[label] if label >= 0 else logp.max())
        if len(self._cache) >= self.CACHE_SIZE:
            self._cache.clear()
        self._cache[key] = value
        return value


__all__ = [
    "SCORING_METHODS",
    "ScoringMethod",
    "CellTypeScoring",
    "AdditiveMatrixScoring",
    "MseDifferenceScoring",
    "MLClassifierScoring",
    "register_scoring_method",
    "get_scoring_method",
    "UNSET",
]
