
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Wed Sep 16 10:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: The Searching Arms of `LLM_NARX_SR` -- one class, driven by an `ArmSpec`
 *
 *      Loop { proposer -> parse -> fit & score [-> trimmer] -> population + traces }
 *           -> term library -> [selector] -> best equation  [-> core / seed back to the proposer, next round]
 *
 *  Every searching arm of the paper is this pipeline with the knobs of its
 *  `ArmSpec`:  llm-loop = an LLM proposer, no selector (V1);  llm-cem = an LLM
 *  proposer, the union of its proposals as the library, the cross-entropy method
 *  as the selector (V4, the paper's model), with a trimmer inside the loop (V3),
 *  without the EDA block ("blind") or with LLM <-> selector rounds (V5) as
 *  ablations;  genetic / random = the population rule / grammar sampling as the
 *  proposer, no selector (V8 / V9);  stat-cem = no loop, the grammar expanded in
 *  full as the library, the same selector (V6, the library no LLM wrote).  The
 *  seed of every loop is the plain-ARX baseline on the exogenous shortlist (the
 *  `Baseline` anchor's equation), so an arm is credited only with what its search
 *  added; the selector may pick any technique of `SubsetSearch` (@see
 *  `TermLibrary.selectBy`), and a CEM selector's inclusion probabilities are the
 *  stable core an LLM is shown in the next round.  The `llm` factory gives every
 *  LLM run a fresh, counted client (calls, tokens, cost per run); a `CachedClient`
 *  inside the factory makes a run reproducible and free to repeat.
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
/** The `PipelineArm` class runs the search pipeline an `ArmSpec` describes.
 *  @param spec        the arm: proposer, trimmer, selector, library, rounds, prompt options
 *  @param llm         the factory for a fresh LLM client per run (required by an LLM proposer)
 *  @param maxLag      the maximum lag in proposals and libraries
 *  @param nFreqs      the number of wave frequencies available to proposals and libraries
 *  @param perRound    the number of proposals requested per round
 *  @param maxRounds   the maximum number of loop rounds (LLM calls) per loop
 *  @param patience    the stagnant rounds that stop a loop
 *  @param topK        the number of best candidates shown to the proposer (and the prior's base)
 *  @param maxTerms    the maximum number of terms per genetic / random proposal
 *  @param maxCols     the maximum number of columns in a sampled CEM set (Some (-1) => none;
 *                         None => hp("maxCols") on the grammar library, none on a loop's)
 *  @param coreThresh  the inclusion probability from which a term counts as stable core
 *  @param prices      the USD per million prompt/reply tokens for the cost column
 *  @param funcs       the functions applied to every lagged variable of the grammar library
 *  @param powers      the exponents applied to every lagged variable of the grammar library
 *  @param nlTerms     whether the proposers and the grammar library use the free-parameter forms
 *                         (@see `NARX_FormulaParser`; defaults to hp("nlTerms") = 1)
 */
class PipelineArm (spec: ArmSpec, llm: Option [() => LLMClient] = None,
                   maxLag: Int = FormulaScorer.hp("maxLag").toInt, nFreqs: Int = FormulaScorer.hp("nFreqs").toInt,
                   perRound: Int = FormulaScorer.hp("perRound").toInt, maxRounds: Int = FormulaScorer.hp("maxRounds").toInt,
                   patience: Int = FormulaScorer.hp("patience").toInt, topK: Int = FormulaScorer.hp("topK").toInt,
                   maxTerms: Int = 12, maxCols: Option [Int] = None,
                   coreThresh: Double = 0.8, prices: (Double, Double) = (0.0, 0.0),
                   funcs: Seq [String] = TermLibrary.LIB_FUNCTIONS, powers: Seq [Double] = TermLibrary.LIB_POWERS,
                   nlTerms: Boolean = FormulaScorer.hp("nlTerms").toInt == 1)
      extends SearchArm:

    require (spec.proposer != ProposerKind.LLM || llm.isDefined, s"PipelineArm: ${spec.name} needs an LLM client factory")

    private val debug = debugf ("PipelineArm", true)                    // debug function
    private val cap   = maxCols.getOrElse (                             // the CEM's column cap
                            if spec.library == LibraryKind.Grammar then FormulaScorer.hp("maxCols").toInt else -1)
    private val cemSample = FormulaScorer.hp("cemSample").toInt         // column sets per CEM round (-1 => the selector's default)
    private val cemIter   = FormulaScorer.hp("cemIter").toInt           // maximum CEM rounds
    private val fitLib    = FormulaScorer.hp("fitLib").toInt == 1       // whether the grammar library's free-parameter terms are fitted first
    private val maxFits   = FormulaScorer.hp("maxFits").toInt           // model fits a run's loops may make (0 => no budget)

    def method: Method = spec.method
    override def variant: String = spec.variant

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Run the arm once and return its record.
     *  @param cfg     the dataset configuration
     *  @param scorer  the fit-and-score engine of the dataset
     *  @param stream  the random number stream / LLM run tag of this run
     */
    def run (cfg: DatasetConfig, scorer: FormulaScorer, stream: Int): ExpRun =
        val eval0 = scorer.evaluations
        val exo   = scorer.shortlist (cfg)
        val freqs = waveFreqs (cfg, scorer, nFreqs)
        val base  = NARX_FormulaParser.parseFormula (cfg.baselineFormulaStr (maxLag, exo), scorer.vars, maxLag, freqs)
        if spec.proposer == ProposerKind.NoLoop then selectOnly (cfg, scorer, stream, exo, freqs, base, eval0)
        else loopThenSelect (cfg, scorer, stream, exo, freqs, base, eval0)
    end run

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** No loop: expand the grammar into the library and select one equation from
     *  it once (a uniform prior for CEM); the selected equation is traced as
     *  `Source.Selected`.
     *  @param cfg     the dataset configuration
     *  @param scorer  the fit-and-score engine
     *  @param stream  the random number stream of the sampler
     *  @param exo     the exogenous shortlist
     *  @param freqs   the wave frequencies of the training rows
     *  @param base    the baseline formula (the answer when nothing can be fitted)
     *  @param eval0   the scorer's fit count before the run
     */
    private def selectOnly (cfg: DatasetConfig, scorer: FormulaScorer, stream: Int, exo: Seq [String],
                            freqs: VectorD, base: ARX_Formula, eval0: Int): ExpRun =
        val tech = spec.selector.get                                    // guaranteed by `ArmSpec`
        val lib0 = TermLibrary.fromGrammar (cfg.resp, exo, maxLag, funcs, powers, freqs, nFreqs, nlTerms = nlTerms)
                              .finite (scorer.fullSet)
        val lib  = if fitLib && nlTerms then lib0.fitted (scorer) else lib0   // free-parameter terms fitted one by one first
        debug ("run", s"$name: grammar library of ${lib.size} terms over ${exo.mkString (", ")}" +
                      (if lib.size < lib0.size then s" (${lib0.size - lib.size} degenerate free-parameter terms dropped)" else "") +
                      s"; wave frequencies ${freqs}")
        val (sel, probs, nEv) = lib.selectBy (tech, scorer, null, stream, cap, cemSample, cemIter)   // the CEM budget
        debug ("run", s"$name: $tech evaluated $nEv sets => ${sel.formulaStr} @ ${sel.score}" +
                      (if probs != null then s"; core: ${lib.showProbs (probs, 8)}" else ""))
        val best  = if sel.failed then scorer.score (base) else sel
        val trace = Trace (1, Source.Selected, sel, Some (tech), sel)
        record (scorer, cfg, stream, best, Seq (best), Seq (trace)).copy (
            nEval = scorer.evaluations - eval0, probs = probs, labels = lib.labels)
    end selectOnly

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Run `rounds` times: the loop with the spec's proposer and trimmer, seeded
     *  with the baseline (and the previous round's selected equation), then, with
     *  a selector, the library of the loop's proposals and the selection; the
     *  answer is the best of the loop's best and the selected equations.
     *  @param cfg     the dataset configuration
     *  @param scorer  the fit-and-score engine
     *  @param stream  the random number stream / LLM run tag
     *  @param exo     the exogenous shortlist
     *  @param freqs   the wave frequencies of the training rows
     *  @param base    the baseline formula (the seed, and the answer when nothing can be fitted)
     *  @param eval0   the scorer's fit count before the run
     */
    private def loopThenSelect (cfg: DatasetConfig, scorer: FormulaScorer, stream: Int, exo: Seq [String],
                                freqs: VectorD, base: ARX_Formula, eval0: Int): ExpRun =
        val isLLM   = spec.proposer == ProposerKind.LLM
        val profile = if isLLM && spec.eda then edaOf (cfg, scorer, cfg.exo, maxLag, nFreqs) else null   // the whole pool
        var core: Seq [String]                 = Seq ()                 // stable-core labels shown to the LLM
        var seeds: Seq [(ARX_Formula, Source)] = Seq ()                 // the previous round's selected equation
        var best: Candidate                    = null                   // best equation so far
        var top: Seq [Candidate]               = Seq ()
        var probs: VectorD                     = null
        var labels: Array [String]             = null
        var (nProp, nVal, nFail, calls)        = (0, 0, 0, 0)
        var (tokens, cost)                     = (0L, 0.0)
        val conv                               = VEC [ConvRow] ()       // convergence rows of every loop
        val traces                             = VEC [Trace] ()         // every entry of every loop, in order
        var error: String                      = null                   // a proposer failure that cut a loop short

        for r <- 1 to spec.rounds if error == null do
            val client = if isLLM then new CountingClient (llm.get (), prices._1, prices._2) else null
            val proposer: Proposer = spec.proposer match
                case ProposerKind.LLM =>
                    val lp = new LLMProposer (client, cfg, scorer.vars, maxLag, freqs, profile, spec.named,
                                              tag = s"[run $stream] [$name]", nonlinear = nlTerms)
                    if core.nonEmpty then lp.setCore (core)
                    lp
                case ProposerKind.Genetic => new GeneticProposer (cfg.resp, exo, maxLag, freqs, stream, maxTerms, nonlinear = nlTerms)
                case ProposerKind.Random  => new GeneticProposer (cfg.resp, exo, maxLag, freqs, stream, maxTerms, random = true, nonlinear = nlTerms)
                case ProposerKind.NoLoop  => throw new IllegalStateException ("PipelineArm: a loop needs a proposer")
            val budget = if maxFits <= 0 then 0 else math.max (1, maxFits - (scorer.evaluations - eval0))   // fits left for this loop
            val loop = new SearchLoop (scorer, topK, spec.trimmer, stream, budget)
            loop.seed (((base, Source.Seed) +: seeds)*)
            loop.run (proposer, perRound, maxRounds, patience)
            if loop.error != null then error = loop.error             // the arm goes on with what it has
            nProp += loop.nProposed; nVal += loop.nValid; nFail += loop.nFailed
            if client != null then { calls += client.calls; tokens += client.tokens; cost += client.cost }
            conv   ++= loop.convergence
            traces ++= loop.traces
            val loopBest = loop.best

            spec.selector match
            case None =>                                                // the loop's best is the answer
                for b <- loopBest if best == null || b.score < best.score do best = b
                top = loop.top (topK)
            case Some (tech) =>
                val cands = if spec.library == LibraryKind.TopK then loop.top (topK)
                            else loop.all.filterNot (_.failed)
                if cands.isEmpty then
                    debug ("run", s"$name round $r: no fitted candidate -- nothing to select from")
                else
                    val lib   = TermLibrary.from (cands).finite (scorer.fullSet)
                    val prior = TermLibrary.frequencyPrior (loop.top (topK), lib)
                    debug ("run", s"$name round $r: library of ${lib.size} terms from ${cands.size} candidates; prior ${lib.showProbs (prior, 8)}")
                    val sel = try                                       // the selection stage, guarded
                        val (c, p, nEv) = lib.selectBy (tech, scorer, prior, stream, cap, cemSample, cemIter)   // the CEM budget
                        debug ("run", s"$name round $r: $tech evaluated $nEv sets => ${c.formulaStr} @ ${c.score}")
                        probs  = p
                        labels = lib.labels
                        core   = if p == null then Seq ()
                                 else (0 until lib.size).filter (j => p(j) >= coreThresh).map (lib.labels (_))
                        c
                    catch case e: Exception =>
                        debug ("run", s"$name round $r: selection failed (${e.getMessage}) -- keeping the loop's best")
                        loopBest.getOrElse (cands.head)
                    traces += Trace (loop.rounds + 1, Source.Selected, sel, Some (tech), sel)
                    val roundBest = if ! sel.failed && loopBest.forall (b => sel.score <= b.score) then sel
                                    else loopBest.getOrElse (sel)
                    if best == null || roundBest.score < best.score then best = roundBest
                    seeds = Seq ((roundBest.formula, Source.Selected))
                    top   = (sel +: loop.top (topK)).filterNot (_.failed).distinctBy (_.key).sortBy (_.score).take (topK)
            end match
        end for

        if best == null then best = scorer.score (base)                 // nothing fitted at all: the baseline
        record (scorer, cfg, stream, best, top, traces.toSeq).copy (
            nProposed = nProp, nValid = nVal, nFailed = nFail, nEval = scorer.evaluations - eval0,
            llmCalls = calls, tokens = tokens, cost = cost, conv = conv.toSeq, probs = probs, labels = labels,
            error = error)                                              // a cut-short loop is reported, not hidden
    end loopThenSelect

end PipelineArm


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `pipelineArmTest` main function runs the ZERO-LLM searching arms on the
 *  Covid-19 dataset (icu, pRate): random and genetic (one stream each) must return
 *  a fitted best scoring no worse than the seeded baseline, with every budget
 *  column filled, a trace behind the winner and the same stream reproducing the
 *  same result; stat-cem's library must have the expected size, its equation be
 *  fitted, drawn from the library and traced as selected, the same stream
 *  reproduce it, and a backward selector on the same library must run too.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.pipelineArmTest
 */
@main def pipelineArmTest (): Unit =

    val cfg = Datasets.COVID2
    given dataset: Map [String, VectorD] = cfg.load ()
    val scorer = new FormulaScorer (hh = 6)
    val base   = scorer.score (NARX_FormulaParser.parseFormula (cfg.baselineFormulaStr (6), scorer.vars, 6))
    println (s"baseline: ${base.formulaStr} @ ${base.score} (in-sample sMAPE ${base.smape})")

    for arm <- Seq (new PipelineArm (ArmSpec.genetic), new PipelineArm (ArmSpec.random)) do
        banner (s"${arm.name} arm, stream 0")
        val r = arm.run (cfg, scorer, 0)
        println (s"best: ${r.best.formulaStr}")
        println (f"score ${r.best.score}%.3f  in-sample sMAPE ${r.best.smape}%.3f  TnT sMAPE ${r.testSmape}%.3f  terms ${r.best.nTerms}  exo ${r.exoUsed.mkString (", ")}")
        r.showQoF ()
        println (s"proposed ${r.nProposed}, valid ${r.nValid}, failed ${r.nFailed}, fits ${r.nEval}, rounds ${r.conv.size}, traces ${r.traces.size}")
        println (r.equations)
        println (s"origin of the winner: ${r.originOf (r.best).getOrElse ("none")}")
        assert (! r.best.failed, "the best candidate must be fitted")
        assert (r.best.score <= base.score + 1e-12, "the best must be at least as good as the seed")
        assert (r.nProposed > 0 && r.nEval > 0 && r.conv.nonEmpty, "budget columns must be filled")
        assert (r.originOf (r.best).isDefined, "the winner must have a trace")
        assert (r.smapeTest.forall (v => ! v.isNaN), "the TnT QoF must be defined")
        val r2 = arm.run (cfg, scorer, 0)
        assert (r2.best.key == r.best.key, "the same stream must reproduce the same best formula")
    end for

    banner ("stat-cem: the grammar library, selected from once")
    val arm = new PipelineArm (ArmSpec.statCEM)
    val r   = arm.run (cfg, scorer, 0)
    println (s"best: ${r.best.formulaStr}")
    println (f"score ${r.best.score}%.3f  in-sample sMAPE ${r.best.smape}%.3f  TnT sMAPE ${r.testSmape}%.3f  terms ${r.best.nTerms}  fits ${r.nEval}")
    r.showQoF ()
    println (s"library (${r.labels.length} terms), top probabilities: " +
             (0 until r.probs.dim).sortBy (j => -r.probs(j)).take (8).map (j => f"${r.labels(j)} ${r.probs(j)}%.2f").mkString (", "))
    assert (arm.name == "stat-cem" && r.method == Method.StatCEM, "the spec derives the method and the label")
    val nLin = 3 * 6 * 4 + 4
    val nLib = nLin + (if FormulaScorer.hp("nlTerms").toInt == 1 then 3 * 6 * TermLibrary.LIB_NL_FORMS.size else 0)
    assert (r.labels.length <= nLib && r.labels.length >= nLin,       // degenerate free-parameter terms are dropped by `fitted`
            s"expected between $nLin and $nLib library terms, got ${r.labels.length}")
    println (s"library: ${r.labels.length} of $nLib terms kept; free parameters in the stat-cem equation: ${r.best.nFree}")
    val cap = FormulaScorer.hp("maxFree").toInt
    assert (cap < 0 || r.best.nFree <= cap, "the selected equation respects the free-parameter cap")
    val fromLib = Par.withMode (Par.LabelMode.Slot) { r.best.formula.preds.forall (t => r.labels.contains (t.label)) }
    assert (! r.best.failed && fromLib, "fitted, from the library")                 // fitted labels carry values, the library's carry ?
    assert (r.traces.size == 1 && r.traces.head.source == Source.Selected, "the selected equation is traced")
    println (f"plain baseline score ${base.score}%.3f vs stat-cem ${r.best.score}%.3f")
    val r2 = arm.run (cfg, scorer, 0)
    assert (r2.best.key == r.best.key, "the same stream must reproduce the same equation")
    val r1 = arm.run (cfg, scorer, 1)
    println (f"stream 1 best: ${r1.best.formulaStr} @ score ${r1.best.score}%.3f, TnT sMAPE ${r1.testSmape}%.3f")

    banner ("the same library with backward elimination as the selector")
    val armB = new PipelineArm (ArmSpec.statCEM.copy (selector = Some (SelectionTech.Backward)))
    val rB   = armB.run (cfg, scorer, 0)
    println (f"${armB.name} best: ${rB.best.formulaStr} @ score ${rB.best.score}%.3f, TnT sMAPE ${rB.testSmape}%.3f  fits ${rB.nEval}")
    assert (armB.name == "stat-cem-sel-backward" && rB.probs == null, "a non-CEM selector leaves no probabilities")
    val fromLibB = Par.withMode (Par.LabelMode.Slot) { rB.best.formula.preds.forall (t => rB.labels.contains (t.label)) }
    assert (! rB.best.failed && fromLibB, "fitted, from the library")

    banner ("labels: a CEM selector is named only where the method does not imply it")
    assert (ArmSpec.genetic.copy (selector = Some (SelectionTech.CEM)).name == "genetic-sel-cem" &&
            ArmSpec.genetic.copy (trimmer = Some (SelectionTech.Backward), selector = Some (SelectionTech.CEM)).name ==
                "genetic-trim-backward-sel-cem" &&
            ArmSpec.llmCEM.name == "llm-cem" && ArmSpec.statCEM.name == "stat-cem",
            "a genetic arm with a selector must not share the plain genetic arm's label")
    println ("pipelineArmTest: passed")

end pipelineArmTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `pipelineArmTest2` main function runs the LLM arms OFFLINE on the Covid-19
 *  dataset (icu, pRate) with a `MockLLMClient` that cycles through scripted
 *  replies (with rejects, a code fence and a TFORM line).  llm-loop: calls
 *  counted (one per round), validity below one, a fitted best no worse than the
 *  seed, the adaptive stop, a trace behind the winner, the same script
 *  reproducing the same best.  llm-cem: the library is the union of the proposed
 *  terms, the selected equation a subset of it scoring no worse than the loop's
 *  best, probabilities, labels and the selection trace recorded.  Then the
 *  ablations: blind, a CEM and a backward trimmer inside the loop, a backward
 *  selector, and two LLM <-> selector rounds.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.pipelineArmTest2
 */
@main def pipelineArmTest2 (): Unit =

    val cfg = Datasets.COVID2
    given dataset: Map [String, VectorD] = cfg.load ()
    val scorer = new FormulaScorer (hh = 6)

    val replies = VEC (
        "y ~= y~1 + x1~2\ny ~= y~(2,1) + log1p(x2~1)\ny ~= y~1 + x9~1",
        "```\ny ~= y~1 + y~2 + x1~1 * y~1\nTFORM: Log1p\ny ~= y~(3,1) + x1~(2,1)\n```",
        "Sure! y ~= y~1\ny ~= y~1 + wave(1)\ny ~= y~1 + sq(x2~2)\ny ~= y~(4,1) + x2~(3,1) + x1~3 + x2~4")
    val mock: Option [() => LLMClient] = Some (() => new MockLLMClient (replies.clone ()))
    def arm (spec: ArmSpec, rounds: Int = 6, pat: Int = 2): PipelineArm =
        new PipelineArm (spec, mock, maxRounds = rounds, patience = pat)
    val base = scorer.score (NARX_FormulaParser.parseFormula (cfg.baselineFormulaStr (6), scorer.vars, 6))

    banner ("llm-loop with a scripted client")
    val loopArm = arm (ArmSpec.llmLoop)
    val r0 = loopArm.run (cfg, scorer, 0)
    println (s"best: ${r0.best.formulaStr}")
    println (f"score ${r0.best.score}%.3f  in-sample sMAPE ${r0.best.smape}%.3f  TnT sMAPE ${r0.testSmape}%.3f  terms ${r0.best.nTerms}")
    r0.showQoF ()
    println (s"proposed ${r0.nProposed}, valid ${r0.nValid} (${r0.validity}), failed ${r0.nFailed}, fits ${r0.nEval}, LLM calls ${r0.llmCalls}, rounds ${r0.conv.size}")
    println (r0.equations)
    println (s"origin of the winner: ${r0.originOf (r0.best).getOrElse ("none")}")
    assert (loopArm.name == "llm-loop" && r0.method == Method.LLMLoop, "the spec derives the method and the label")
    assert (r0.llmCalls == r0.conv.size && r0.llmCalls >= 2, "one LLM call per round")
    assert (r0.validity < 1.0 && r0.nValid > 0, "the x9 line must count as invalid, the others as valid")
    assert (! r0.best.failed && r0.best.score <= base.score + 1e-12, "the best must be fitted and no worse than the seed")
    assert (r0.conv.size < 6, "the loop must stop early once the script repeats")
    assert (r0.originOf (r0.best).isDefined && r0.traces.size == 1 + r0.nValid, "one trace per seed and legal proposal")
    assert (loopArm.run (cfg, scorer, 0).best.key == r0.best.key, "the same script must reproduce the same best")

    banner ("llm-cem: loop, union library, CEM selector")
    val cemArm = arm (ArmSpec.llmCEM)
    val r = cemArm.run (cfg, scorer, 0)
    println (s"best: ${r.best.formulaStr}")
    println (f"score ${r.best.score}%.3f  in-sample sMAPE ${r.best.smape}%.3f  TnT sMAPE ${r.testSmape}%.3f  terms ${r.best.nTerms}")
    r.showQoF ()
    println (s"proposed ${r.nProposed}, valid ${r.nValid}, fits ${r.nEval}, LLM calls ${r.llmCalls}")
    println (s"library (${r.labels.length}): ${r.labels.mkString (", ")}")
    println (s"probabilities: ${r.probs}")
    println (r.equations)
    println (s"origin of the winner: ${r.originOf (r.best).getOrElse ("none")}")
    assert (cemArm.name == "llm-cem" && r.method == Method.LLMCEM, "the spec derives the method and the label")
    assert (r.probs != null && r.labels != null && r.probs.dim == r.labels.length, "probabilities and labels must be recorded")
    val lib = r.labels.toSet
    assert (r.best.formula.preds.forall (t => lib.contains (t.label)), "the selected equation must come from the library")
    assert (! r.best.failed, "the selected equation must be fitted")
    assert (r.traces.exists (_.source == Source.Selected), "the selection stage must leave a trace")
    println (f"loop-only best ${r0.best.score}%.3f vs llm-cem best ${r.best.score}%.3f")
    assert (r.best.score <= r0.best.score + 1e-12, "the selector over the library must score no worse than the loop's best")

    banner ("ablations: blind, trimmers inside the loop, a backward selector, two rounds")
    val specs = Seq (ArmSpec.llmCEM.copy (eda = false),
                     ArmSpec.llmCEM.copy (trimmer = Some (SelectionTech.CEM)),
                     ArmSpec.llmCEM.copy (trimmer = Some (SelectionTech.Backward)),
                     ArmSpec.llmCEM.copy (selector = Some (SelectionTech.Backward)),
                     ArmSpec.llmCEM.copy (rounds = 2))
    val runs = for s <- specs yield
        val a  = arm (s, rounds = 4)
        val ra = a.run (cfg, scorer, 0)
        println (f"${a.name}%-22s best: ${ra.best.formulaStr} @ score ${ra.best.score}%.3f, TnT sMAPE ${ra.testSmape}%.3f, LLM calls ${ra.llmCalls}")
        for t <- ra.traces if t.trimmed do println (s"  trimmed: $t")
        assert (! ra.best.failed, s"${a.name} must return a fitted equation")
        (a.name, ra)
    val names = runs.map (_._1)
    assert (names == Seq ("llm-cem-blind", "llm-cem-trim", "llm-cem-trim-backward", "llm-cem-sel-backward", "llm-cem-2r"),
            s"derived labels: $names")
    val rTrim = runs(1)._2
    assert (rTrim.traces.exists (_.trimmed), "the padded LLM proposal must be trimmed")
    val rSelB = runs(3)._2
    assert (rSelB.probs == null && rSelB.traces.exists (t => t.source == Source.Selected && t.trimmer.contains (SelectionTech.Backward)),
            "a backward selector leaves no probabilities and is traced under its own technique")
    val r2r = runs(4)._2
    r2r.showQoF ()
    assert (r2r.llmCalls > r.llmCalls / 2, "two rounds must consult the LLM in both")
    assert (r2r.traces.count (_.source == Source.Selected) >= 2, "each round's selection is traced")
    println ("pipelineArmTest2: passed")

end pipelineArmTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `pipelineArmTest3` main function runs llm-loop and llm-cem against a
 *  LOCAL model server (`LocalClient`, e.g., ollama serve with llama3.1) through
 *  a `CachedClient`, so a repeated run costs nothing; skips when no server answers.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.pipelineArmTest3
 */
@main def pipelineArmTest3 (): Unit =

    val cfg = Datasets.COVID2
    given dataset: Map [String, VectorD] = cfg.load ()
    val scorer = new FormulaScorer (hh = 6)

    val local = new LocalClient ()
    if ! local.reachable () then
        println ("no local model server reachable -- skipping pipelineArmTest3 (start one: ollama pull llama3.1 && ollama serve)")
    else
        val llm: Option [() => LLMClient] = Some (() => new CachedClient (new LocalClient (), "llm_cache"))
        for spec <- Seq (ArmSpec.llmLoop, ArmSpec.llmCEM) do
            val arm = new PipelineArm (spec, llm, maxRounds = 8, patience = 3)
            val r   = arm.run (cfg, scorer, 0)
            banner (s"${arm.name} with a local model")
            println (s"best: ${r.best.formulaStr}")
            println (f"score ${r.best.score}%.3f  in-sample sMAPE ${r.best.smape}%.3f  TnT sMAPE ${r.testSmape}%.3f")
            println (s"proposed ${r.nProposed}, valid ${r.nValid} (${r.validity}), LLM calls ${r.llmCalls}, tokens ${r.tokens}")
            println (r.equations)
        end for

end pipelineArmTest3
