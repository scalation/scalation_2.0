
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Mon Sep 14 15:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Term Library and Selection (CEM, Backward, ...) for `LLM_NARX_SR`
 *
 *  A `TermLibrary` is a set of candidate terms (one design column each) from
 *  which ONE equation is selected by a technique of `SubsetSearch` (`selectBy`:
 *  the cross-entropy method -- the search's selection stage --, backward
 *  elimination, forward, stepwise or beam), every set scored by
 *  `FormulaScorer.evalCols` -- the in-sample selection criterion of the scorer on
 *  the TRAINING rows (J, the mean FPE-form sMAPE over horizons), the same criterion every
 *  candidate of the search is ranked by, each set fitted by the same engine.  The
 *  techniques differ only in how sets are explored, never in what is scored;
 *  `Forecaster_D.selectFeatures` is the same selection on the base's own models
 *  (@see `termLibraryTest2`).  Two ways to build
 *  a library:  `TermLibrary.from` takes the union of the terms of scored
 *  candidates (what an LLM loop proposed: the LLM writes the hypothesis space),
 *  with `frequencyPrior` turning how often each term appeared into the CEM's
 *  starting inclusion probabilities; `TermLibrary.fromGrammar` expands the grammar
 *  itself (every variable x lag x function), the library an LLM did not write --
 *  the control that isolates what the LLM's library is worth.  A sin/cos wave
 *  pair is kept whole when only one member is selected.
 *  Terms with FREE NONLINEAR PARAMETERS (@see `NARX_Formula`): a library term enters
 *  the design matrix at its parameter's current value (the role start value, or the
 *  value of the fit of the candidate it came from) -- the subsets are scored without
 *  refitting the parameters --, and the selected equation gets its own parameters
 *  (`NARX_Formula.fresh`) and is fitted and scored by the engine like any proposal.
 *  Two copies of a term that differ only in start or fitted values are one term.
 *  The grammar library has the free-parameter forms when `nlTerms` (pow(x~k, ?),
 *  log(? + x~k), sat(x~k, ?), ema(x~k, ?)).
 *  @note    AI Assisted Code
 */

package scalation
package modeling
package forecasting
package nonlinear
package llm_narx_sr

import scala.collection.mutable.{ArrayBuffer => VEC, LinkedHashSet => LSET}

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TermLibrary` class holds the candidate terms of a selection problem.
 *  @param resp    the response variable
 *  @param terms   the candidate terms (distinct by label; one design column each)
 *  @param tFormT  the response/base-column transform the selected equation carries (null => none)
 */
class TermLibrary (val resp: FormulaVar, val terms: VEC [Term], val tFormT: TransformT = null):

    private val debug = debugf ("TermLibrary", false)                   // debug function

    val size: Int              = terms.size                             // number of candidate terms/columns
    val labels: Array [String] = Par.withMode (Par.LabelMode.Slot) { terms.map (_.label).toArray }   // column names (free parameters as ?)
    private val index          = labels.zipWithIndex.toMap              // label -> column

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the column of a term label, or -1 if the library lacks it.  The
     *  library's `labels` write free parameters as ?; a label with start values
     *  (?w) matches, a label with FITTED values (plain numbers) does not -- take it
     *  from the term in the `Par.LabelMode.Slot` mode instead (@see `frequencyPrior`).
     *  @param label  the term label (@see `Term.label`)
     */
    def indexOf (label: String): Int = index.getOrElse (NARX_Formula.slots (label), -1)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the columns with a wave pair completed: when one member of a
     *  sin/cos pair (same `WaveTerm.idx`) is in cols, its partner is added.
     *  @param cols  the selected columns
     */
    def completeWaves (cols: LSET [Int]): LSET [Int] =
        val idxs  = cols.toSeq.collect { case j if terms(j).isInstanceOf [WaveTerm] => terms(j).asInstanceOf [WaveTerm].idx }.toSet
        val freqs = cols.toSeq.collect { case j if terms(j).isInstanceOf [ParWaveTerm] => terms(j).asInstanceOf [ParWaveTerm].freq }
        val out   = LSET [Int] ()
        for j <- terms.indices do
            val partner = terms(j) match                                // the other member of a selected pair
                case w: WaveTerm    => idxs.contains (w.idx)
                case w: ParWaveTerm => freqs.exists (_ eq w.freq)      // a free-frequency pair shares its parameter
                case _              => false
            if cols.contains (j) || partner then out += j
        end for
        out
    end completeWaves

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the formula made of the given columns (library order, wave pairs
     *  completed, the library's transform).
     *  @param cols  the selected columns
     */
    def formulaOf (cols: LSET [Int]): ARX_Formula =
        val keep = completeWaves (cols)
        NARX_Formula (resp, NARX_Formula.fresh (terms.zipWithIndex.collect { case (t, j) if keep.contains (j) => t }), tFormT)
    end formulaOf

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the formula containing every term of the library (own parameters).
     */
    def full: ARX_Formula = NARX_Formula (resp, NARX_Formula.fresh (terms), tFormT)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the library without the terms whose data column is not finite on
     *  the given rows (e.g., log of a zero, a power of a negative value), so the
     *  library model can be built (@see `ARX_D.fromTerms`).
     *  @param dataset  the map of named data columns
     */
    def finite (dataset: Map [String, VectorD]): TermLibrary =
        val ok = terms.filter (t => t.vars.forall (dataset.contains) &&
                                    ! t.column (dataset).exists (v => v.isNaN || v.isInfinite))
        if ok.size < size then debug ("finite", s"dropped ${size - ok.size} non-finite terms")
        new TermLibrary (resp, ok, tFormT)
    end finite

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the unit of a column: the sin/cos pair of a wave term (the two
     *  columns enter and leave a set together), the column itself otherwise.
     *  @param j  the column
     */
    def unitOf (j: Int): Set [Int] =
        def sameWave (k: Int): Boolean = (terms(j), terms(k)) match       // members of one sin/cos pair
            case (a: WaveTerm, b: WaveTerm)       => a.idx == b.idx
            case (a: ParWaveTerm, b: ParWaveTerm) => a.freq eq b.freq
            case _                                => false
        terms.indices.filter (k => k == j || sameWave (k)).toSet
    end unitOf

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Select one equation from the library by a selection technique of
     *  `SubsetSearch` -- the cross-entropy method (@see `CEM_Selector`), backward
     *  elimination, forward selection, stepwise selection or beam search -- every
     *  set scored by the in-sample criterion of the scorer (@see
     *  `FormulaScorer.evalCols`: the training rows, the ridge penalty, the mean
     *  FPE-form sMAPE over horizons, J) with wave pairs completed first (so the set scored is
     *  the formula returned) and kept whole (`unitOf`).  The techniques differ only
     *  in how the sets are explored, never in what is scored.  Returns the selected
     *  equation scored as a `Candidate`, the final inclusion probability per column
     *  (CEM; null otherwise) and the number of column sets evaluated.
     *  @param tech     the selection technique
     *  @param scorer   the fit-and-score engine (its training rows, ridge and criterion)
     *  @param prior    the prior inclusion probability per column (CEM; null => 0.5 each)
     *  @param stream   the random number stream of the sampler (CEM)
     *  @param maxCols  the maximum number of columns in a set (-1 => no cap; a completed
     *                      wave pair may add one column beyond it)
     *  @param nSample  the number of column sets sampled per round (CEM; -1 => 5 x columns)
     *  @param maxIter  the maximum number of rounds (CEM)
     *  @param minCols  the smallest set kept (Backward)
     */
    def selectBy (tech: SelectionTech, scorer: FormulaScorer, prior: VectorD = null, stream: Int = 0,
                  maxCols: Int = -1, nSample: Int = CEM_Selector.N_SAMPLE, maxIter: Int = CEM_Selector.MAX_ITER,
                  minCols: Int = 1): (Candidate, VectorD, Int) =
        require (size > 0, "selectBy: the library is empty")
        given qk: Int = QoF.smapeC.ordinal                               // the QoF slot carrying the selection score
        val lib  = scorer.library (resp, terms, tFormT)
        val eval = (cols: LSET [Int]) =>
            val keep = completeWaves (cols)
            scorer.evalCols (lib, keep, freeCount (keep))
        val maxFree = FormulaScorer.hp("maxFree").toInt
        val freeCol = terms.indices.filter (j => NARX_Formula.params (Seq (terms(j))).exists (_.free))
        val prior_  =                                                    // CEM: a free-parameter column starts as likely as the cap allows
            if maxFree < 0 || freeCol.size <= maxFree then prior
            else
                val p  = if prior == null then VectorD.fill (size)(0.5) else prior.copy
                val pf = math.min (0.5, 0.8 * maxFree / freeCol.size)
                for j <- freeCol do p(j) = math.min (p(j), pf)
                p
        val path = tech match
            case SelectionTech.CEM      => SubsetSearch.cem (size, eval, LSET (), prior_, nSample, maxIter, maxCols, stream)
            case SelectionTech.Forward  => SubsetSearch.forward (size, eval, LSET (), unitOf, maxCols)
            case SelectionTech.Backward => SubsetSearch.backward (size, eval, LSET (), unitOf, minCols)
            case SelectionTech.Stepwise => SubsetSearch.stepwise (size, eval, LSET (), unitOf)
            case SelectionTech.Beam     => SubsetSearch.beam (size, eval, LSET (), unitOf)
        val f    = if ! path.found then full else formulaOf (withinCap (path.cols, maxFree))   // no set evaluated: the full library
        var cand = scorer.score (f)                                     // re-scored as a formula (same fit path)
        val top  = FormulaScorer.hp("refitTop").toInt
        if tech == SelectionTech.CEM && top > 0 && path.found && freeCount (path.cols) > 0 then
            val sets = path.steps.filter (_.mod_cols != null).map (s => withinCap (s.mod_cols, maxFree))
                           .reverse.distinctBy (_.toSet).take (top)
            for s <- sets if s.toSet != path.cols.toSet do
                val c = scorer.score (formulaOf (s))
                if ! c.failed && (cand.failed || c.score < cand.score) then cand = c
            end for
        if cand.failed && path.found && freeCount (path.cols) > 0 then   // a free parameter degenerate in context (on its bound,
            val plain = LSET.from (path.cols.toSeq.filterNot (j => NARX_Formula.params (Seq (terms(j))).exists (_.free)))
            if plain.nonEmpty then                                       //   collinear): the selected set without its free-parameter terms
                val c = scorer.score (formulaOf (plain))
                debug ("selectBy", s"$tech: the selected equation was rejected (${cand.reason}); without its free-parameter terms: ${c.formulaStr} @ ${c.score}")
                if ! c.failed then cand = c
        debug ("selectBy", s"$tech: ${path.nEval} sets evaluated => ${cand.formulaStr} @ ${cand.score}")
        (cand, path.probs, path.nEval)
    end selectBy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of free nonlinear parameters in a column set.
     *  @param cols  the columns
     */
    def freeCount (cols: LSET [Int]): Int =
        cols.toSeq.map (j => NARX_Formula.params (Seq (terms(j))).count (_.free)).sum
    end freeCount

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a column set brought within the free-parameter cap: while it holds
     *  more than `maxFree` free parameters, the free-parameter column added last
     *  (highest index) is dropped -- the selected equation must be scorable.
     *  @param cols     the columns
     *  @param maxFree  the cap (-1 => none)
     */
    def withinCap (cols: LSET [Int], maxFree: Int): LSET [Int] =
        if maxFree < 0 then return cols
        val out = LSET.from (cols)
        while freeCount (out) > maxFree do
            val j = out.toSeq.filter (j => NARX_Formula.params (Seq (terms(j))).exists (_.free)).max
            out -= j
        out
    end withinCap

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the library with every free-parameter term fitted alone against the
     *  response (the formula resp ~= term, @see `FormulaScorer.score`, fit rows), its
     *  parameters then holding the fitted values the selection stage builds the
     *  design matrix at; a term whose fit fails or is rejected (a parameter on its
     *  bound, @see `FormulaScorer.rejection`) is dropped with its reason.  Terms
     *  without free parameters pass unchanged.  One fit per free-parameter term.
     *  @param scorer  the fit-and-score engine
     */
    def fitted (scorer: FormulaScorer): TermLibrary =
        val keep = VEC [Term] ()
        var nDrop = 0
        for t <- terms do
            if NARX_Formula.params (Seq (t)).exists (_.free) then
                val c = scorer.score (NARX_Formula (resp, VEC (t), tFormT))
                if c.failed then
                    nDrop += 1
                    debug ("fitted", s"dropped ${t.label}: ${c.reason}")
                else keep += c.formula.preds.head                        // the term with its fitted parameter
            else keep += t
        end for
        if nDrop > 0 then println (s"TermLibrary.fitted: ${terms.size - nDrop} of ${terms.size} terms kept ($nDrop free-parameter terms dropped: degenerate fits)")
        new TermLibrary (resp, keep, tFormT)
    end fitted

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show the inclusion probabilities next to the term labels, highest first.
     *  @param probs  the inclusion probability per column
     *  @param k      the number of terms to show
     */
    def showProbs (probs: VectorD, k: Int = 10): String =
        (0 until size).sortBy (j => -probs(j)).take (k).map (j => f"${labels(j)} ${probs(j)}%.2f").mkString (", ")
    end showProbs

    override def toString: String = s"TermLibrary (${resp.name}, ${labels.mkString (", ")}" +
                                    (if tFormT == null then ")" else s", TFORM: $tFormT)")

end TermLibrary


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TermLibrary` companion object builds libraries from scored candidates
 *  or from the grammar, and priors from term frequencies.
 */
object TermLibrary:

    /** The functions and powers the grammar library applies to every lagged variable.
     */
    val LIB_FUNCTIONS = Seq ("log1p", "sq")
    val LIB_POWERS    = Seq (1.5)

    /** The free-parameter forms the grammar library applies to every lagged variable
     *  when `nlTerms` (each term gets its own parameter): a free exponent, a shifted
     *  log, a saturation, a hinge (a threshold, rev3.2) and an exponentially weighted
     *  moving average.
     */
    val LIB_NL_FORMS: Seq [Term => Term] = Seq (
        (t: Term) => pow (t, power ()),
        (t: Term) => log (shift () + t),
        (t: Term) => sat (t, half ()),
        (t: Term) => hinge (t, thresh ()),                              // a threshold effect of a driver
        (t: Term) => ema (t, decay ()))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build the union library of the terms of the given candidates (failed
     *  candidates excluded); the transform is the best candidate's.
     *  @param cands  the scored candidates, best first
     */
    def from (cands: Seq [Candidate]): TermLibrary =
        val ok = cands.filterNot (_.failed)
        require (ok.nonEmpty, "from: no fitted candidates to build a library from")
        val resp  = ok.head.formula.resp
        val terms = NARX_Formula.distinct (ok.flatMap (_.formula.preds))   // by structure: fitted values do not count
        new TermLibrary (resp, terms, ok.head.formula.tFormT)
    end from

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the prior inclusion probability per library column as the share of
     *  candidates whose formula contains the term, clamped to [eps, 1 - eps].
     *  @param cands  the scored candidates the frequencies are taken from
     *  @param lib    the library
     *  @param eps    the floor/ceiling of the probabilities
     */
    def frequencyPrior (cands: Seq [Candidate], lib: TermLibrary, eps: Double = CEM_Selector.EPS): VectorD =
        val ok = cands.filterNot (_.failed)
        val p  = new VectorD (lib.size)
        if ok.nonEmpty then
            for c <- ok; t <- c.formula.preds do
                val j = lib.indexOf (Par.withMode (Par.LabelMode.Slot) { t.label })   // by structure (@see `labels`)
                if j >= 0 then p(j) += 1.0 / ok.size
        end if
        p.map (v => math.min (1.0 - eps, math.max (eps, v)))
    end frequencyPrior

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Expand the grammar into a library: every variable (response first) at
     *  lags 1..maxLag, each such term under every function and power, and the
     *  seasonal wave pairs.
     *  @param resp    the response variable name
     *  @param exo     the exogenous variable names
     *  @param maxLag  the maximum lag
     *  @param funcs   the function names applied to each lagged variable (@see `ARX_Formula.functions`)
     *  @param powers  the exponents applied to each lagged variable (`pow`)
     *  @param freqs   the wave frequencies (null => no waves)
     *  @param nWaves  the number of wave pairs (capped by freqs.dim)
     *  @param tFormT  the response/base-column transform (null => none)
     *  @param nlTerms whether to add the free-parameter forms (`LIB_NL_FORMS`) of each lagged variable
     */
    def fromGrammar (resp: String, exo: Seq [String], maxLag: Int,
                     funcs: Seq [String] = LIB_FUNCTIONS, powers: Seq [Double] = LIB_POWERS,
                     freqs: VectorD = null, nWaves: Int = 0, tFormT: TransformT = null,
                     nlTerms: Boolean = false): TermLibrary =
        val terms = VEC [Term] ()
        for v <- resp +: exo; k <- 1 to maxLag do terms += FormulaVar (v, k)
        for v <- resp +: exo; k <- 1 to maxLag; fn <- funcs do terms += ARX_Formula.functions (fn)(FormulaVar (v, k))
        for v <- resp +: exo; k <- 1 to maxLag; p <- powers do terms += pow (FormulaVar (v, k), p)
        if nlTerms then
            for v <- resp +: exo; k <- 1 to maxLag; form <- LIB_NL_FORMS do terms += form (FormulaVar (v, k))
        if freqs != null && nWaves > 0 then terms ++= wave (freqs, nWaves)
        new TermLibrary (FormulaVar (resp), NARX_Formula.distinct (terms), tFormT)
    end fromGrammar

end TermLibrary


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `termLibraryTest` main function tests cross-entropy selection from a
 *  grammar library on a series with a planted structure
 *      y_t = 5 + 0.7 y_t-1 - 0.2 y_t-3 + 1.5 z_t-2 + e_t,
 *  scored at h = 1 (only h = 1 identifies the planted lags: a direct model at a
 *  later horizon needs other ones).  The selected equation must carry exactly
 *  the planted (variable, lag) pairs -- a monotone transform of a planted
 *  column, e.g., sq(y~3) for y~3, is the same structure up to the transform and
 *  is accepted -- and the frequency prior and `from` are checked on hand-made
 *  candidates, with exact recovery from the union library.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.termLibraryTest
 */
@main def termLibraryTest (): Unit =

    import scalation.random.Normal

    val n     = 400
    val noise = Normal (0.0, 1.0, 11)
    val zz    = Normal (0.0, 4.0, 12)
    val z     = VectorD (for _ <- 0 until n yield 10.0 + zz.gen)
    val y     = new VectorD (n)
    y(0) = 30.0; y(1) = 31.0; y(2) = 32.0
    for t <- 3 until n do y(t) = 5.0 + 0.7 * y(t-1) - 0.2 * y(t-3) + 1.5 * z(t-2) + noise.gen

    given dataset: Map [String, VectorD] = Map ("y" -> y, "z" -> z)

    val scorer = new FormulaScorer (hh = 1)
    val lib    = TermLibrary.fromGrammar ("y", Seq ("z"), maxLag = 4)

    def baseOf (t: Term): (String, Int) = t match                        // the (variable, lag) under any transform
        case FormulaVar (nm, l)          => (nm, l)
        case TransformedTerm (s, _, _)   => baseOf (s)
        case PowerTerm (s, _, _)         => baseOf (s)
        case other                       => (other.label, other.lag)
    banner (s"grammar library with ${lib.size} terms")
    println (lib)
    assert (lib.size == 8 + 16 + 8, s"unexpected library size ${lib.size}")

    val (cand, probs, nEval) = lib.selectBy (SelectionTech.CEM, scorer, stream = 0)
    banner ("selected equation")
    println (s"${cand.formulaStr}  score = ${cand.score}, sMAPE by horizon = ${cand.smapeH}")
    println (s"evaluated $nEval column sets; top probabilities: ${lib.showProbs (probs, 6)}")
    val chosen = cand.formula.preds.map (baseOf).toSet
    val truth  = Set (("y", 1), ("y", 3), ("z", 2))
    assert (truth.subsetOf (chosen), s"planted structure not selected: $chosen")
    assert (chosen.size <= 4, s"too many spurious terms selected: $chosen")

    banner ("library from candidates and frequency prior")
    val f1 = NARX_FormulaParser.parseFormula ("y ~= y~1 + z~2", Set ("y", "z"), maxLag = 4)
    val f2 = NARX_FormulaParser.parseFormula ("y ~= y~1 + y~3", Set ("y", "z"), maxLag = 4)
    val f3 = NARX_FormulaParser.parseFormula ("y ~= y~1 + log1p(z~1)", Set ("y", "z"), maxLag = 4)
    val cands = Seq (f1, f2, f3).map (scorer.score)
    val lib2  = TermLibrary.from (cands)
    val prior = TermLibrary.frequencyPrior (cands, lib2)
    println (s"union library = ${lib2.labels.mkString (", ")}")
    println (s"prior         = $prior")
    assert (lib2.size == 4, "union of the three formulas has 4 distinct terms")
    assert (math.abs (prior(lib2.indexOf ("y~1")) - 0.95) < 1e-12, "y~1 appears in all three: prior at the ceiling")
    assert (math.abs (prior(lib2.indexOf ("z~2")) - 1.0 / 3.0) < 1e-12, "z~2 appears once: prior 1/3")

    val (cand2, _, _) = lib2.selectBy (SelectionTech.CEM, scorer, prior, stream = 1)
    println (s"selected from the union: ${cand2.formulaStr} @ ${cand2.score}")
    assert (cand2.formula.preds.map (_.label).toSet == Set ("y~1", "y~3", "z~2"), "exact recovery from the union library")

    banner ("backward elimination on the same criterion")
    val (candB, _, nEvalB) = lib.selectBy (SelectionTech.Backward, scorer)
    println (s"backward from the grammar library: ${candB.formulaStr} @ ${candB.score} ($nEvalB sets evaluated)")
    val chosenB = candB.formula.preds.map (baseOf).toSet
    assert (truth.subsetOf (chosenB), s"planted structure not selected by backward elimination: $chosenB")
    assert (chosenB.size <= 4, s"too many spurious terms kept by backward elimination: $chosenB")
    val (candB2, _, _) = lib2.selectBy (SelectionTech.Backward, scorer)
    println (s"backward from the union: ${candB2.formulaStr} @ ${candB2.score}")
    assert (candB2.formula.preds.map (_.label).toSet == Set ("y~1", "y~3", "z~2"), "exact recovery from the union library")
    assert (math.abs (candB2.score - cand2.score) < 1e-12, "the same equation scores the same under both techniques")

    banner ("every technique through selectBy recovers the planted structure from the union")
    for tech <- SelectionTech.values do
        val (c, _, nEv) = lib2.selectBy (tech, scorer, prior, stream = 1)
        println (s"$tech: ${c.formulaStr} @ ${c.score} ($nEv sets evaluated)")
        assert (c.formula.preds.map (_.label).toSet == Set ("y~1", "y~3", "z~2"), s"$tech must recover the planted structure")
    println ("termLibraryTest: passed")

end termLibraryTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `termLibraryTest2` main function checks that the search's selector and the
 *  base's own feature selection are ONE selection: on a formula's term library over
 *  the COVID-19 dataset (icu, pRate), `selectBy (Backward)` must keep exactly the
 *  terms `Forecaster_D.selectFeatures (Backward, first = 0)` keeps on the same
 *  `fromTerms` model (no intercept column, the search's skip, the training rows),
 *  with the same score J -- same rows, same regression, same rule, same
 *  evaluations.  (The base's three selection modes against each other are
 *  `aRX_DTest8`; this package depends on the base, never the reverse.)
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.termLibraryTest2
 */
@main def termLibraryTest2 (): Unit =

    import Example_Covid.{BEST2_EXO, clip, loadData}
    import SelectionTech.Backward

    val (xe, y) = clip (loadData (BEST2_EXO))                          // icu_patients, positive_rate
    val hh      = 6                                                     // maximum forecasting horizon
    val lam     = RidgeRegression.hp("lambda")                          // remember the global lambda
    RidgeRegression.hp("lambda") = 1.0
    FormulaScorer.hp("lambda")   = 1.0
    FormulaScorer.hp("valFrac")  = 0.0                                  // the base selects in-sample on the training rows
    FeatureSelection.fullset_FS  = false                                // select on the training rows

    given dataset: Map [String, VectorD] = Map ("y" -> y, "icu" -> xe.col(0), "pRate" -> xe.col(1))
    val scorer = new FormulaScorer (hh)
    val f      = NARX_FormulaParser.parseFormula ("y ~= y~(3,1) + icu~(2,1) + log1p(pRate~1) + sq(icu~2)", scorer.vars, maxLag = 6)
    val lib    = new TermLibrary (f.resp, f.preds, f.tFormT)

    banner ("backward over all horizons: TermLibrary.selectBy vs Forecaster_D.selectFeatures on the same fromTerms model")
    val (candB, _, nEvalB) = lib.selectBy (Backward, scorer)
    val modF   = scorer.build (f)                                       // `fromTerms`: one column per term, no intercept, scaled as the search
    modF.setSkip (scorer.skip)                                          // the search's skip
    val (cJ, rJ) = modF.selectFeatures (Backward, first = 0)          // arrays of one element (one set for all horizons)
    val keptJ  = cJ(0).toSeq.sorted.map (j => f.preds(j).label)
    val keptB  = candB.formula.preds.map (_.label).toSeq
    val scoreJ = rJ(0).map (_(3)).min                                   // the best J along the path
    println (s"TermLibrary.selectBy (Backward):  ${keptB.mkString (", ")} @ ${candB.score} ($nEvalB sets)")
    println (s"ARX_D.selectFeatures (Backward): ${keptJ.mkString (", ")} @ $scoreJ")
    assert (keptJ == keptB, "the joint selection must keep the terms the search's selector keeps")
    assert (math.abs (scoreJ - candB.score) < 1E-9, "and score them the same")

    RidgeRegression.hp("lambda") = lam                                  // restore the global lambda
    println ("termLibraryTest2: passed")

end termLibraryTest2



//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `termLibraryTest3` main function tests libraries with FREE PARAMETERS: the
 *  union of candidates counts a term with a free parameter once whatever its start or
 *  fitted value (`NARX_Formula.distinct`), the frequency prior finds it, the grammar
 *  library with `nlTerms` has one free-parameter form per variable, lag and form, the
 *  library model is built at the parameters' current values without a fit, and the
 *  equation `selectBy` returns has its OWN parameters (none shared with the library)
 *  and is fitted (its parameters set).  On y_t = 5 + 0.6 y_t-1 + 4 log (z_t-1 + 0.3) + e_t
 *  the backward selection from a small library keeps the shifted log.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.termLibraryTest3
 */
@main def termLibraryTest3 (): Unit =

    import scalation.random.Normal

    val n     = 300
    val burn  = 50
    val noise = Normal (0.0, 0.3, 41)
    val zz    = Normal (0.0, 1.5, 42)
    val zAll  = VectorD (for _ <- 0 until n + burn yield 3.0 + zz.gen).map (v => math.max (v, 0.1))
    val yAll  = new VectorD (n + burn)
    yAll(0) = 20.0
    for t <- 1 until n + burn do yAll(t) = 5.0 + 0.6 * yAll(t-1) + 4.0 * math.log (zAll(t-1) + 0.3) + noise.gen
    val (y, z) = (yAll(burn until n + burn), zAll(burn until n + burn))

    given dataset: Map [String, VectorD] = Map ("y" -> y, "z" -> z)
    FormulaScorer.hp("lambda")   = 1e-6
    FormulaScorer.hp("boundTol") = -1.0                                 // the rejection rules off and the in-sample J -- this test is
    FormulaScorer.hp("maxVIF")   = 0.0                                  //   about the library mechanics (pow(z~1, ?) on a log-driven series ends on
    FormulaScorer.hp("valFrac")  = 0.0                                  //   its bound, and on a validation block z^0.5 can edge out log(0.3 + z))
    val scorer = new FormulaScorer (hh = 1)
    val vars   = scorer.vars

    banner ("union library: one term per structure")
    val f1 = NARX_FormulaParser.parseFormula ("y ~= y~1 + log(? + z~1)", vars, 4)
    val f2 = NARX_FormulaParser.parseFormula ("y ~= y~1 + log(?3 + z~1) + z~2", vars, 4)
    val f3 = NARX_FormulaParser.parseFormula ("y ~= y~2 + pow(z~1, ?)", vars, 4)
    val cands = Seq (f1, f2, f3).map (scorer.score)
    for c <- cands do println (s"  ${c.formulaStr} @ ${c.score}")
    val lib   = TermLibrary.from (cands)
    println (s"library: ${lib.labels.mkString (", ")}")
    assert (lib.size == 5, s"y~1, log(? + z~1), z~2, y~2, pow(z~1, ?): 5 structures, got ${lib.size}")
    assert (lib.labels.count (_ == "log(? + z~1)") == 1, "the fitted and the unfitted shifted log are one term")
    val prior = TermLibrary.frequencyPrior (cands, lib)
    println (s"prior:   $prior")
    assert (math.abs (prior (lib.indexOf ("log(? + z~1)")) - 2.0 / 3.0) < 1e-12, "the shifted log appears in two of three candidates")
    assert (lib.indexOf ("log(?2.000 + z~1)") == lib.indexOf ("log(? + z~1)"), "indexOf ignores the start values in a label")

    banner ("the library model: built at the current values, no fit")
    val nFits = scorer.evaluations
    val libMod = scorer.library (lib.resp, lib.terms, lib.tFormT)
    assert (! libMod.isInstanceOf [NARX_SR_D] && libMod.getX.dim2 == lib.size, "an ARX_D with one column per term")
    assert (scorer.evaluations == nFits, "building the library model fits nothing")

    banner ("backward selection: the kept equation has its own, fitted parameters")
    val (sel, _, nEv) = lib.selectBy (SelectionTech.Backward, scorer)
    println (s"selected: ${sel.formulaStr} @ ${sel.score} ($nEv sets)")
    assert (! sel.failed && sel.formula.preds.exists (_.label.startsWith ("log(")), "the shifted log must be kept")
    val libPars = NARX_Formula.params (lib.terms)
    val selPars = NARX_Formula (sel.formula).params
    assert (selPars.forall (p => ! libPars.exists (_ eq p)), "the selected equation must not share a parameter object with the library")
    assert (selPars.forall (p => ! p.label.startsWith ("?")), s"its parameters are fitted (plain labels): ${sel.formulaStr}")

    banner ("the grammar library with the free-parameter forms")
    val gl = TermLibrary.fromGrammar ("y", Seq ("z"), maxLag = 3, nlTerms = true)
    println (gl)
    assert (gl.size == 6 + 12 + 6 + 6 * TermLibrary.LIB_NL_FORMS.size, s"unexpected library size ${gl.size}")
    assert (NARX_Formula.params (gl.terms).size == 6 * TermLibrary.LIB_NL_FORMS.size, "one parameter per free-parameter term")

    banner ("the grammar library with its free-parameter terms fitted one by one")
    val n0  = scorer.evaluations
    val glf = gl.fitted (scorer)
    println (s"fitted library: ${glf.size} of ${gl.size} terms kept, ${scorer.evaluations - n0} fits")
    println (Par.withMode (Par.LabelMode.Value) { glf.terms.filter (t => NARX_Formula.params (Seq (t)).nonEmpty).map (_.label).mkString (", ") })
    assert (glf.size <= gl.size && glf.size >= 6 + 12 + 6, "the plain terms are all kept, degenerate free-parameter terms dropped")
    assert (scorer.evaluations - n0 <= 6 * TermLibrary.LIB_NL_FORMS.size, "one fit per free-parameter term at most")
    val fittedPars = NARX_Formula.params (glf.terms)
    assert (fittedPars.forall (p => ! p.label.startsWith ("?")), "every kept free parameter holds a fitted value")
    assert (glf.freeCount (LSET.range (0, glf.size)) == fittedPars.count (_.free), "freeCount counts the free parameters of a set")
    assert (fittedPars.nonEmpty, "some free-parameter terms survive their univariate fit")
    println ("termLibraryTest3: passed")

end termLibraryTest3
