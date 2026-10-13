
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Mon Sep 14 15:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: The LLM Proposer of `LLM_NARX_SR`
 *
 *  Prompts an LLM for formula strings and parses the reply.  What
 *  the LLM sees, by default (the contamination policy of the paper): the
 *  variables under BLIND names (y, x1, x2, ...), a generic dataset sentence
 *  ("a weekly series y with 2 covariates"), the grammar (@see `ARX_Formula.grammar`)
 *  with a note that the intercept is implicit and, when `nonlinear`, the free-parameter
 *  forms of `NARX_FormulaParser` (pow(x1~1, ?), log(? + x1~1), sat(x1~1, ?), ...: a ?
 *  is a parameter the engine fits on the training rows; a fitted equation is shown
 *  back with its values as start values, ?w), optionally the EDA profile of the
 *  training rows in words (@see `promptBlock` of an `EDA4TS`; over the whole
 *  pool of candidate variables, strongest driver first, when the system picks the
 *  exogenous variables), the scored population (each equation with its in-sample
 *  sMAPE per horizon on the training rows; when a trimmer ran, the proposal, the
 *  kept equation and the dropped terms -- so the LLM learns what selection
 *  removed), last round's rejects with their reasons and, when a cross-entropy
 *  stage ran, the stable core of terms it found.  It never sees data rows, the
 *  dataset's name, the real variable names or anything from the testing rows.
 *  `named = true` restores the domain sentence and the real names (the
 *  contamination ablation).
 *
 *  LLMs wrap output in prose or code fences despite instructions, so lines are
 *  kept by grammar prefix ("y ~=", plus a following "TFORM:" line) rather than
 *  trusting the format; a line that does not parse is returned as a reject with
 *  its reason, which the search counts as invalid and shows back next round.
 *  @note    AI Assisted Code
 */

package scalation
package modeling
package forecasting
package nonlinear
package llm_narx_sr

import scala.collection.mutable.{ArrayBuffer => VEC}

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LLMProposer` class prompts an LLM for formulas.
 *  @param llm     the completion client (wrap in `CachedClient`/`CountingClient` for experiments)
 *  @param cfg     the dataset configuration (variables, generic/domain sentence)
 *  @param vars    the legal variable names (the dataset's keys)
 *  @param maxLag  the maximum lag advertised in the grammar
 *  @param freqs   the wave frequencies (null or empty => omit wave)
 *  @param eda     the EDA profile whose prompt block is shown (null => none)
 *  @param named   whether the real variable names and the domain sentence are shown
 *  @param tag        a text folded into the prompt (e.g., "[run 3]") so a cache
 *                        keyed on the prompt gives each run its own transcript
 *  @param nonlinear  whether the grammar shown includes the free-parameter forms
 *                        (@see `NARX_FormulaParser.grammarLines`; defaults to hp("nlTerms") = 1)
 */
class LLMProposer (llm: LLMClient, cfg: DatasetConfig, vars: Set [String], maxLag: Int,
                   freqs: VectorD, eda: EDA4TS = null, named: Boolean = false, tag: String = "",
                   nonlinear: Boolean = FormulaScorer.hp("nlTerms").toInt == 1)
      extends Proposer:

    private val debug   = debugf ("LLMProposer", false)                 // debug function (prints raw replies)
    private val nFreqs  = if freqs == null then 0 else freqs.dim        // wave pairs available
    private val toShown = if named then Map [String, String] () else cfg.blindNames   // real -> shown name
    private val toReal  = toShown.map (_.swap)                          // shown -> real name
    private val ident   = "[A-Za-z_][A-Za-z0-9_]*".r                   // identifier pattern
    private var rejects: Seq [(String, String)] = Seq ()                // last round's (formulaStr, reason)
    private var core: Seq [String] = Seq ()                             // stable-core terms from a CEM stage
    private var failed: Seq [(String, String)] = Seq ()                 // last round's proposals that fitted badly or were rejected, with reasons
    private var notes: String = ""                                      // diagnostic notes on the current best (residual autocorrelation, importances)
    private var _calls  = 0                                             // prompts sent
    private var _last   = ""                                            // the last user prompt sent

    def source: Source = Source.LLM

    private val rn = shown (cfg.resp)                                   // names as the LLM sees them
    private val x1 = cfg.exo.headOption.map (shown).getOrElse (rn)
    private val x2 = cfg.exo.lift (1).map (shown).getOrElse (x1)

    /** The system prompt: task, variables, grammar, rules, examples in the shown names.
     */
    val system: String =
        val about  = if named then cfg.schema else LLMProposer.genericSchema (cfg.exo.size, cfg.freq)
        val waveEx = if nFreqs > 0 then s"\n  $rn ~= $rn~(2,1) + $x1~1 + wave(1)" else ""
        val waveRl = if nFreqs > 0 then s" wave(k) with 1 <= k <= $nFreqs adds k seasonal sin/cos pairs." else ""
        val nlEx   = if nonlinear then s"\n  $rn ~= $rn~(2,1) + pow($rn~1, ?) + log(? + $x1~1) + sat($x2~1, ?)" else ""
        val nlRl   = if nonlinear then "\n" + NARX_FormulaParser.grammarLines (x1) else ""
        s"""You propose formulas for a symbolic time-series model that forecasts $rn.
           |$about
           |Variables: ${(cfg.resp +: cfg.exo).map (shown).mkString (", ")}.  VAR~k is the value of VAR k steps back, 1 <= k <= $maxLag.
           |The model is linear in its terms with an implicit intercept: never write a constant term.
           |Write one formula per line, nothing else on the line, exactly in this style:
           |  $rn ~= $rn~1 + $rn~2 + $x1~1
           |  $rn ~= $rn~(3,1) + log1p($x1~2) + sq($x2~1)
           |  $rn ~= $rn~1 + pow($x1~1, 1.5) + $rn~1 * $x1~1$waveEx$nlEx
           |Allowed terms: VAR~k, VAR~(max,min) (all lags from max down to min), log1p(term), log(term), sq(term),
           |exp(term), pow(term, p) and powr(term, p) with 0.2 <= p <= 3.0, a product of exactly two single
           |lagged variables written like $rn~1 * $x1~2 (never a range, a function or wave inside a product).$waveRl$nlRl
           |Use the variable names exactly as given; never write VAR or term literally.
           |Non-negative data: log(0) is not finite, so prefer log1p.  Formulas with 3 to 8 terms do best
           |(each term costs a small penalty).${if nonlinear then s"  A free parameter (?) costs as much as a term and at most ${LLMProposer.maxFreeShown} are allowed per formula; a fit that ends on a parameter's bound is rejected." else ""}
           |Optionally end a formula with a second line:  TFORM: Log1p
           |Each proposal should test ONE idea -- one new term, one changed lag, one transform -- against the best
           |formulas shown, so that what helped can be told from what did not; the ideas should differ between proposals.
           |Formulas are judged by their forecast error on rows they were NOT fitted on (a validation block at the end
           |of the training rows, all horizons), so prefer parsimonious structures that generalize over close fits.
           |Propose formulas likely to have LOWER validation sMAPE than those shown.
           |""".stripMargin

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of prompts sent so far.
     */
    def calls: Int = _calls

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the last user prompt sent (for inspection and the paper's appendix).
     */
    def lastPrompt: String = _last

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Tell the proposer which terms a cross-entropy stage found stable (shown
     *  in the next prompts as the core to build on; @see `PipelineArm`).
     *  @param labels  the term labels of the stable core (real names)
     */
    def setCore (labels: Seq [String]): Unit = core = labels

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Keep the round's failed/rejected proposals (shown next round with their
     *  reasons, as the parse rejects are) and the diagnostic notes on the best.
     *  @param failed_  (formula string, reason) of the round's failed and rejected proposals
     *  @param notes_   the diagnostic notes ("" when none)
     */
    override def feedback (failed_ : Seq [(String, String)], notes_ : String): Unit =
        failed = failed_
        notes  = notes_
    end feedback

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Propose n formulas by prompting the LLM with the scored population, last
     *  round's rejects and failures, the diagnostic notes on the best, and the
     *  optional EDA block and core; parse the reply.
     *  Context lines are prefixed with "# " so a model that echoes them back is
     *  filtered out rather than mis-parsed.  A trimmed trace is shown as
     *  "proposed f -> kept fTrim (dropped: ...)" with the kept equation's score.
     *  @param top  the origin traces of the current best candidates, best first
     *  @param n    the number of proposals requested
     */
    def propose (top: Seq [Trace], n: Int): Seq [Either [String, ARX_Formula]] =
        def line (t: Trace): String =                                   // a fitted equation as a template: ?w slots
            val c = t.candTrim
            val head = f"# validation sMAPE = ${c.smapeVal}%7.2f  (in-sample ${c.smape}%7.2f)  by horizon = ${byH (c.smapeValH)}  terms = ${c.nTerms}%2d  :  "
            if t.trimmed then
                head + s"proposed ${toShownText (t.cand.proposalStr)}  ->  kept ${toShownText (c.proposalStr)}" +
                s"  (dropped: ${t.dropped.map (d => toShownText (d.label)).mkString (", ")})"
            else head + toShownText (c.proposalStr)
        end line
        val shownTop = if top.isEmpty then "# (none yet)" else top.map (line).mkString ("\n")
        val profile  = if eda == null then "" else toShownText (LLMProposer.promptBlock (eda)) + "\n"
        val coreTxt  = if core.isEmpty then ""
                       else s"Terms found stable by the selection stage (build on them): ${core.map (toShownText).mkString (", ")}\n"
        val bad      = if rejects.isEmpty && failed.isEmpty then ""
                       else "Rejected last round -- do NOT repeat these:\n" +
                            (rejects ++ failed).map ((s, r) => s"# ${toShownText (s).replace ('\n', ' ')}  -- ${toShownText (r)}").mkString ("\n") + "\n"
        val diag     = if notes.isEmpty then "" else s"# ${toShownText (notes)}\n"   // residual autocorrelation and importances of the best
        val user =
            s"""$tag
               |${profile}Best formulas so far (lower validation sMAPE is better; do NOT repeat these, they are EXAMPLES not proposals):
               |$shownTop
               |$diag$coreTxt${bad}Propose $n NEW, DIVERSE formulas, each testing one idea.  One per line, grammar only, no prose, no numbering.
               |""".stripMargin

        _last   = user
        val raw = llm.complete (system, user)
        _calls += 1
        debug ("propose", s"raw completion (top.size = ${top.size}, n = $n):\n$raw")

        val lines = VEC [String] ()
        for ln <- raw.linesIterator.map (_.trim.stripPrefix ("```").trim) if ln.nonEmpty do
            if ln.startsWith (s"$rn ~=") then lines += toRealText (ln)
            else if ln.toUpperCase.startsWith ("TFORM:") && lines.nonEmpty then
                lines(lines.size - 1) = lines.last + " " + ln              // attach to the preceding formula
            else debug ("propose", s"line dropped: $ln")
        end for
        val rej = VEC [(String, String)] ()
        val out = lines.toSeq.take (n).map { s =>
            NARX_FormulaParser.tryParseFormula (s, vars, maxLag, freqs) match
            case Right (f)     => Right (f)
            case Left (reason) =>
                val why = reason.stripPrefix ("parse: ").split (" in formula: ")(0)
                rej += ((s, why))
                Left (s"$s -- $why")
        }
        rejects = rej.toSeq                                             // remembered for the next prompt
        out
    end propose

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the shown name of a variable (blind unless `named`).
     *  @param v  the real variable name
     */
    private def shown (v: String): String = toShown.getOrElse (v, v)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Rename the real variables to the shown names in a text (identity when `named`).
     *  @param s  the text (a formula string, a reason or a profile line)
     */
    private def toShownText (s: String): String =
        if named then s else ident.replaceAllIn (s, m => toShown.getOrElse (m.matched, m.matched))
    end toShownText

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Rename the shown names back to the real variables in a text (identity when `named`).
     *  @param s  the text (a proposal line from the LLM)
     */
    private def toRealText (s: String): String =
        if named then s else ident.replaceAllIn (s, m => toReal.getOrElse (m.matched, m.matched))
    end toRealText

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Format an sMAPE per horizon vector for the prompt, e.g., [11.4, 12.4].
     *  @param sH  the sMAPE per horizon
     */
    private def byH (sH: VectorD): String = sH.indices.map (h => f"${sH(h)}%.1f").mkString ("[", ", ", "]")

end LLMProposer


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LLMProposer` companion object provides the generic dataset sentence of
 *  the blind (default) prompt and the EDA profile in words (`promptBlock`).
 */
object LLMProposer:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Render an EDA profile in words for a prompt (rev4.1: moved here from `EDA4TS`).
     *  Variables are shown under the given display names (the real names when
     *  absent; the proposer maps them to blind names y, x1, x2, ... through
     *  `toShownText`), nothing identifies the series, and only training-row
     *  statistics appear.
     *  @param eda    the EDA profile of the training rows
     *  @param names  the display name per variable (response and exogenous)
     */
    def promptBlock (eda: EDA4TS, names: Map [String, String] = Map ()): String =
        def nm (v: String): String = names.getOrElse (v, v)
        val yn  = nm (eda.respName)
        val sb  = new StringBuilder ()
        val pac = eda.pacfSpikes.take (3).map ((k, r) => f"$k (${r}%.2f)").mkString (", ")
        sb.append (s"Data profile (training rows only):\n")
        sb.append (f"- $yn: level ${eda.yMean}%.3g, spread ${eda.yStd}%.3g, range ${eda.yMin}%.3g to ${eda.yMax}%.3g" +
                   (if eda.zeroShare > 0.0 then f", ${eda.zeroShare * 100}%.0f%% zeros (log is unsafe, prefer log1p)" else "") + "\n")
        sb.append (s"- $yn autocorrelation at lags 1..${eda.maxLag}: " +
                   (1 to eda.maxLag).map (k => f"${eda.acf(k)}%.2f").mkString (", ") + "\n")
        sb.append (s"- $yn partial autocorrelation spikes (lag (value)): " + (if pac.isEmpty then "none" else pac) + "\n")
        if eda.freqs.dim > 0 then
            sb.append (s"- seasonality: strongest periods " +
                       (0 until eda.freqs.dim).map (i => f"${eda.periods(i)}%.1f steps (wave(${i+1}))").mkString (", ") + "\n")
        for (v, k, r) <- eda.bestLags do
            sb.append (f"- ${nm (v)}: strongest correlation with $yn at lag $k (r = $r%.2f" +
                       (if math.abs (r) <= eda.band then ", weak)" else if r > 0.0 then ", positive)" else ", negative)") + "\n")
        if ! eda.kpssP.isNaN then
            sb.append (f"- KPSS level-stationarity p-value ${eda.kpssP}%.2f" +
                       (if eda.kpssP < 0.05 then s" (non-stationary: lags of $yn matter)" else " (stationary)") + "\n")
        sb.toString
    end promptBlock

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the free-parameter cap as the prompt states it (hp("maxFree"); "any
     *  number" when there is no cap).
     */
    def maxFreeShown: String =                                          // the cap in words
        val k = FormulaScorer.hp("maxFree").toInt
        if k < 0 then "any number" else k.toString
    end maxFreeShown

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the generic dataset sentence: the sampling frequency and the number
     *  of covariates, nothing that identifies the series.
     *  @param nExo  the number of exogenous variables
     *  @param freq  the sampling frequency word (e.g., "weekly")
     */
    def genericSchema (nExo: Int, freq: String = "weekly"): String =
        val covs = nExo match
            case 0 => "no covariates"
            case 1 => "one covariate x1"
            case k => s"$k covariates x1 .. x$k"
        s"Dataset: a $freq series y with $covs; all values are non-negative counts or rates."
    end genericSchema

end LLMProposer


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lLMProposerTest` main function tests the `LLMProposer` class with a
 *  `MockLLMClient`: blind names in the prompt, real names after mapping back,
 *  code fences and prose dropped, a TFORM line attached, rejects returned with
 *  reasons and remembered for the next prompt, and the EDA block rendered in
 *  blind names.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.lLMProposerTest
 */
@main def lLMProposerTest (): Unit =

    val cfg = Datasets.COVID2
    given dataset: Map [String, VectorD] = cfg.load ()
    FormulaScorer.hp("boundTol") = -1.0                                 // the rejection rules off -- this test is about the
    FormulaScorer.hp("maxVIF")   = 0.0                                  //   proposer (sat(pRate~1, ?) ends on its bound on these data)
    val scorer = new FormulaScorer (hh = 6)
    val freqs  = VectorD (0.13043, 0.06521)
    val eda    = new EDA4TS (dataset ("y"), Map ("icu" -> dataset ("icu"), "pRate" -> dataset ("pRate")), scorer.trSize)

    val reply =
        """Here are my proposals:
          |```
          |y ~= y~1 + y~2 + log1p(x1~2)
          |TFORM: Log1p
          |y ~= y~(3,1) + x2~1 * y~1
          |y ~= y~1 + pow(x1~1, ?) + sat(x2~1, ?)
          |y ~= y~1 + wave(3)
          |y ~= y~1 + x9~1
          |```""".stripMargin
    val llm = new MockLLMClient (VEC (reply))
    val lp  = new LLMProposer (llm, cfg, scorer.vars, 6, freqs, eda, tag = "[run 0]")

    banner ("system prompt")
    println (lp.system)
    assert (lp.system.contains ("x1") && ! lp.system.contains ("icu"), "the blind prompt must not name icu")
    assert (lp.system.contains ("pow(x1~1, ?)") && lp.system.contains ("sat(x1~1, ?)"), "the free-parameter forms must be in the grammar")

    banner ("proposals")
    val c0  = scorer.score (NARX_FormulaParser.parseFormula ("y ~= y~1 + icu~2", scorer.vars, 6))
    val cF  = scorer.score (NARX_FormulaParser.parseFormula ("y ~= y~1 + icu~2 + pRate~3", scorer.vars, 6))
    val top = Seq (Trace (0, Source.Seed, c0, None, c0), Trace (1, Source.LLM, cF, Some (SelectionTech.CEM), c0))
    val out = lp.propose (top, 5)
    println (lp.lastPrompt)
    assert (lp.lastPrompt.contains ("proposed y ~= y~1 + x1~2 + x2~3  ->  kept y ~= y~1 + x1~2  (dropped: x2~3)"),
            "a trimmed trace must be shown as proposed -> kept with the dropped terms, in blind names")
    assert (! lp.lastPrompt.contains ("icu") && ! lp.lastPrompt.contains ("testing rows") && ! lp.lastPrompt.contains ("TnT"),
            "blind names, nothing about the testing rows")                 // the validation block IS mentioned
    assert (lp.lastPrompt.contains ("validation sMAPE"), "the shown scores are validation scores (rev3.2)")
    for p <- out do println (s"  $p")
    val ok  = out.collect { case Right (f) => f }
    val bad = out.collect { case Left (r) => r }
    assert (ok.size == 3 && bad.size == 2, s"expected 3 formulas and 2 rejects, got ${ok.size} and ${bad.size}")
    assert (ok(0).tFormT == TransformT.Log1p && ok(0).preds.exists (_.label == "log1p(icu~2)"), "TFORM attached and x1 mapped to icu")
    assert (ok(1).preds.exists (_.label == "pRate~1 * y~1"), "x2 mapped to pRate inside the product")
    assert (NARX_Formula (ok(2)).nFree == 2 && ok(2).preds.exists (_.label == "sat(pRate~1, ?)"),
            s"the free parameters must be parsed, with the names mapped back: ${ok(2).formulaStr}")
    val cN  = scorer.score (ok(2))
    println (s"nonlinear proposal fitted: ${cN.formulaStr} @ ${cN.score}  (shown back as ${cN.proposalStr})")
    assert (! cN.failed && cN.nFree == 2, "the nonlinear proposal must fit")
    val out2 = lp.propose (Seq (Trace (2, Source.LLM, cN, None, cN)), 2)
    assert (lp.lastPrompt.contains ("pow(x1~1, ?") && ! lp.lastPrompt.contains ("pow(x1~1, ?)"),
            s"a fitted equation is shown with its values as start values (?w): ${lp.lastPrompt}")
    assert (out2.nonEmpty, "the second prompt is answered")
    assert (bad.exists (_.contains ("wave(3)")) && bad.exists (_.contains ("x9")), "rejects carry the offending formula")
    assert (lp.calls == 2, "two prompts sent")

    banner ("EDA block shown blind")
    val block = LLMProposer.promptBlock (eda, cfg.blindNames)
    println (block)
    assert (! block.contains ("icu") && block.contains ("x1"), "the EDA block must use blind names")
    println ("lLMProposerTest: passed")

end lLMProposerTest
