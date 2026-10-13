
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Mon Sep 14 15:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Search Arms of `LLM_NARX_SR` -- the Contract and the Records
 *           (`Trace`, `ExpRun`)
 *
 *  Every version of the method compared in the paper is a `SearchArm`: one
 *  `run (cfg, scorer, stream)` call, one `ExpRun` record.  The anchors are
 *  `BaselineArm`; every searching arm is a `PipelineArm` driven by an `ArmSpec`
 *  -- who proposes (`ProposerKind`), the trimmer and the selector (the base's
 *  `SelectionTech`), the library the selector works on (`LibraryKind`), the
 *  rounds and the prompt options -- so the arm table of the paper is a table of
 *  specs (@see `LLM_NARX_SR.mainArms`, `ablationArms`).  The harness (@see
 *  `LLM_NARX_SR`) only loops over arms x streams x folds and tabulates the records;
 *  anything computed FROM the records (tests, tables) lives in files that depend
 *  on this one, never the other way round.  What an experiment runs ON -- the
 *  response, the pool of exogenous variables, who picks them -- is `DatasetConfig`
 *  in `forecasting/Datasets.scala` (rev4.0: moved out of this file; rev4.1: to the
 *  base), which this file depends on.
 *  Names that select behaviour are enums (`Method`, `Source`, `ProposerKind`,
 *  `LibraryKind`, the base's `SelectionTech` and `ExoMode`), so a
 *  misspelt one is a compile error; a spec's `method` and row label are derived
 *  from it, never typed by hand.
 *  A `Trace` is the EVENT "in round r, source s proposed f, trimmer t kept fTrim";
 *  a `Candidate` is the scored unit the population holds; several traces may lead
 *  to one candidate (a formula proposed twice, two proposals trimmed to the same
 *  equation), never fewer than one.
 *  @note    AI Assisted Code
 */

package scalation
package modeling
package forecasting
package nonlinear
package llm_narx_sr

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Method` enum names the arms (the method column of the results table).
 *  @param tag  the display tag
 */
enum Method (val tag: String):
    case Baseline   extends Method ("baseline")                         // V0: plain ARX, p = px = maxLag
    case TunedARX   extends Method ("tuned-arx")                        // V0: ARX grid over p x px
    case SelARX     extends Method ("sel-arx")                          // V0: the baseline's lags under the base's feature selection (rev3.2)
    case RandomWalk extends Method ("random-walk")                      // V0: the base's `RandomWalk`
    case Random     extends Method ("random")                           // V9: grammar sampling through the loop
    case Genetic    extends Method ("genetic")                          // V8: crossover / mutation through the loop
    case StatCEM    extends Method ("stat-cem")                         // V6: grammar library + selector, no loop, no LLM
    case LLMLoop    extends Method ("llm-loop")                         // V1: LLM proposes, no selection stage
    case LLMCEM     extends Method ("llm-cem")                          // V4: LLM loop -> union library -> selector
end Method

object Method:
    val anchors: Set [Method] = Set (Baseline, TunedARX, SelARX, RandomWalk)   // the classical anchors
end Method


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Source` enum names who put a formula into a population.
 */
enum Source:
    case Seed, LLM, Genetic, Random, Selected                           // Selected: the selection stage's equation
end Source


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ProposerKind` enum names who proposes structures inside the search loop
 *  (@see `Proposer`): an LLM, the genetic rule, random grammar sampling -- or
 *  `NoLoop`: no proposals at all, the selector works on the grammar library once.
 */
enum ProposerKind:
    case LLM, Genetic, Random, NoLoop
end ProposerKind


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LibraryKind` enum names the term library the selector works on: the
 *  union of every fitted candidate of the loop, the union of its top-K only, or
 *  the grammar expanded in full (the library no proposer wrote, @see
 *  `TermLibrary.fromGrammar`).
 */
enum LibraryKind:
    case Union, TopK, Grammar
end LibraryKind


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ArmSpec` case class is one searching arm of the pipeline
 *
 *      Loop { proposer -> parse -> fit & score [-> trimmer] -> population } -> library -> [selector]
 *
 *  as a choice of its knobs; `PipelineArm` runs it.  The `method` (the row of the
 *  results table) and the `variant` label are DERIVED from the knobs.
 *  @param proposer  who proposes inside the loop (`NoLoop` => no loop, the grammar library is selected from once)
 *  @param trimmer   the selection technique trimming every proposal inside the loop (None => off)
 *  @param selector  the selection technique over the library after the loop (None => the loop's best is the answer)
 *  @param library   the library the selector works on (`Grammar` iff `NoLoop`)
 *  @param rounds    the number of loop -> selector alternations (> 1 => the selected equation
 *                       seeds the next loop and, for an LLM, the stable core is shown to it)
 *  @param eda       whether the EDA profile of the training rows is in the LLM prompt
 *  @param named     whether the LLM prompt shows the real names and the domain sentence
 *  @param tag       an extra label appended to the variant (e.g., "ft" for a fine-tuned proposer)
 */
case class ArmSpec (proposer: ProposerKind, trimmer: Option [SelectionTech] = None,
                    selector: Option [SelectionTech] = None, library: LibraryKind = LibraryKind.Union,
                    rounds: Int = 1, eda: Boolean = true, named: Boolean = false, tag: String = ""):

    require (rounds >= 1, "ArmSpec: rounds must be at least 1")
    require ((proposer == ProposerKind.NoLoop) == (library == LibraryKind.Grammar),
             "ArmSpec: the grammar library is selected from without a loop, and a loop's library is its proposals")
    require (proposer != ProposerKind.NoLoop || (selector.isDefined && trimmer.isEmpty && rounds == 1),
             "ArmSpec: without a loop there is only the selector, once")
    require (rounds == 1 || selector.isDefined, "ArmSpec: rounds > 1 need a selector to seed the next loop")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the method (the row of the results table) the knobs amount to.
     */
    def method: Method = proposer match
        case ProposerKind.LLM     => if selector.isDefined then Method.LLMCEM else Method.LLMLoop
        case ProposerKind.Genetic => Method.Genetic
        case ProposerKind.Random  => Method.Random
        case ProposerKind.NoLoop  => Method.StatCEM

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the variant label: every knob that departs from the method's standard
     *  version, in a fixed order ("" for the standard version), e.g., "blind",
     *  "trim", "trim-backward", "sel-backward", "2r", "blind-trim"; a CEM selector
     *  is part of the LLMCEM and StatCEM methods, so it is named ("sel-cem") only
     *  on a genetic or random arm.
     */
    def variant: String =
        val parts = scala.collection.mutable.ArrayBuffer [String] ()
        if ! eda then parts += "blind"
        if named then parts += "named"
        for t <- trimmer do parts += (if t == SelectionTech.CEM then "trim" else s"trim-${t.toString.toLowerCase}")
        for t <- selector do                                            // a CEM selector is implied by LLMCEM and StatCEM,
            if t != SelectionTech.CEM || method == Method.Genetic || method == Method.Random then
                parts += s"sel-${t.toString.toLowerCase}"                // not by Genetic / Random: name it there
        if library == LibraryKind.TopK then parts += "topk"
        if rounds > 1 then parts += s"${rounds}r"
        if tag.nonEmpty then parts += tag
        parts.mkString ("-")
    end variant

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the row label: the method tag with the variant appended.
     */
    def name: String = if variant.isEmpty then method.tag else s"${method.tag}-$variant"

end ArmSpec


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ArmSpec` companion object holds the standard specs of the paper's arms;
 *  ablations are `copy`s of them (e.g., `llmCEM.copy (eda = false)`).
 */
object ArmSpec:

    val llmLoop = ArmSpec (ProposerKind.LLM)                                                       // V1
    val llmCEM  = ArmSpec (ProposerKind.LLM, selector = Some (SelectionTech.CEM))                  // V4 (the paper's model)
    val statCEM = ArmSpec (ProposerKind.NoLoop, selector = Some (SelectionTech.CEM),
                           library = LibraryKind.Grammar)                                          // V6
    val genetic = ArmSpec (ProposerKind.Genetic)                                                   // V8
    val random  = ArmSpec (ProposerKind.Random)                                                    // V9

end ArmSpec


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ConvRow` case class is one row of a search's convergence log.
 *  @param round      the round number (1-based)
 *  @param nProposed  the cumulative number of proposals made so far
 *  @param bestScore  the best-so-far selection score
 *  @param bestSmape  the best-so-far mean in-sample sMAPE
 */
case class ConvRow (round: Int, nProposed: Int, bestScore: Double, bestSmape: Double)


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Trace` case class records one proposal and what selection made of it.
 *  @param round     the loop round the proposal was made in (0 for a seed)
 *  @param source    who proposed f
 *  @param cand      f as proposed, scored
 *  @param trimmer   the selection technique that trimmed f (None => full version)
 *  @param candTrim  the kept equation fTrim, scored (the same object as cand when untrimmed)
 */
case class Trace (round: Int, source: Source, cand: Candidate, trimmer: Option [SelectionTech],
                  candTrim: Candidate):

    def f: ARX_Formula      = cand.formula                              // as proposed
    def fTrim: ARX_Formula  = candTrim.formula                          // as kept
    def fScore: Double      = cand.score                                // score of f
    def fTrimScore: Double  = candTrim.score                            // score of fTrim
    def trimmed: Boolean    = trimmer.isDefined && (candTrim ne cand)   // whether selection changed f
    def dropped: Seq [Term] =                                           // terms removed (by label)
        if trimmed then f.preds.filterNot (t => fTrim.preds.exists (_.label == t.label)).toSeq else Seq ()
    def gain: Double        = fScore - fTrimScore                       // how much trimming helped (never < 0, @see `SearchLoop.trim`)
    def best: Candidate     = if fTrimScore <= fScore then candTrim else cand
    def key: String         = candTrim.key                              // the population identity

    override def toString: String =
        val tr = trimmer.map (_.toString).getOrElse ("none")
        f"$round%3d  $source%-8s  $tr%-8s  ${fScore}%8.3f -> ${fTrimScore}%8.3f  ${cand.formulaStr}" +
        (if trimmed then s"  ->  ${candTrim.formulaStr}  (dropped: ${dropped.map (_.label).mkString (", ")})" else "")
    end toString

end Trace


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ExpRun` case class is the outcome of one arm run on one dataset.
 *  @param method     the arm
 *  @param variant    the ablation tag of the arm ("" for the standard version)
 *  @param stream     the random number stream of the run (-1 for a deterministic arm)
 *  @param best       the best candidate (formula or base model, in-sample QoF)
 *  @param qofTest    the QoF matrix of the best model on the testing rows (hh x Fit.N_QoF,
 *                        `rollValidate` forecasts, @see `FormulaScorer.tnt`)
 *  @param nProposed  the number of proposals made (seeds excluded)
 *  @param nValid     the number of proposals that parsed and were legal
 *  @param nFailed    the number of legal candidates whose fit failed
 *  @param nEval      the number of model fits made by the run
 *  @param llmCalls   the LLM consultations consumed (0 => no LLM)
 *  @param tokens     the estimated LLM tokens consumed
 *  @param cost       the estimated LLM cost in USD
 *  @param conv       the convergence log (empty for a run without rounds)
 *  @param top        the best equations at the end (best first)
 *  @param traces     every proposal of the run in order (empty for an anchor)
 *  @param exoUsed    the exogenous variables the best equation uses
 *  @param probs      the inclusion probability per library term left by a CEM
 *                        stage (null when the arm has none)
 *  @param labels     the library term labels the probabilities refer to (null when none)
 *  @param error      what cut the run short or made it fail (an unreachable LLM, an exception;
 *                        null when nothing did) -- rev3.2
 */
case class ExpRun (method: Method, variant: String, stream: Int, best: Candidate, qofTest: MatrixD,
                   nProposed: Int = 0, nValid: Int = 0, nFailed: Int = 0, nEval: Int = 0,
                   llmCalls: Int = 0, tokens: Long = 0L, cost: Double = 0.0,
                   conv: Seq [ConvRow] = Seq (), top: Seq [Candidate] = Seq (), traces: Seq [Trace] = Seq (),
                   exoUsed: Seq [String] = Seq (), probs: VectorD = null, labels: Array [String] = null,
                   error: String = null):

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the row label: the method tag with the variant appended.
     */
    def name: String = if variant.isEmpty then method.tag else s"${method.tag}-$variant"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the share of proposals that parsed and were legal (-0.0 if none were made).
     */
    def validity: Double = if nProposed > 0 then nValid.toDouble / nProposed else -0.0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the in-sample sMAPE per horizon of the best model (the fit rows).
     */
    def smapeTrain: VectorD = best.smapeH

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the mean validation sMAPE over horizons of the best model.
     */
    def valSmape: Double = best.smapeVal

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether the run has no fitted result (its best failed or the arm threw).
     */
    def failed: Boolean = best.failed

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether a searching run's best is still a seed (nothing proposed beat it).
     */
    def noImprovement: Boolean =
        nProposed > 0 && originOf (best).exists (t => t.round == 0 && t.source == Source.Seed)
    end noImprovement

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the TnT sMAPE per horizon of the best model (testing rows, rolling).
     */
    def smapeTest: VectorD = qofTest(?, QoF.smape.ordinal)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the mean TnT sMAPE over horizons (the number a comparison uses).
     */
    def testSmape: Double = smapeTest.mean

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print the QoF matrices of the best model the way the base prints them after
     *  `diagnoseAll` ("fitMap QoF = " then `Fit.showFitMap`, one column per horizon):
     *  in-sample on the training rows, then TnT on the testing rows.
     */
    def showQoF (): Unit =
        banner (s"$name (stream $stream): ${best.formulaStr.replace ('\n', ' ')}")
        println ("In-Sample fitMap QoF = ")
        println (Fit.showFitMap (best.qofTrain.ᵀ))
        println ("TnT fitMap QoF = ")
        println (Fit.showFitMap (qofTest.ᵀ))
    end showQoF

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return every trace that led to a candidate (as f or as fTrim), in order.
     *  @param cand  the candidate
     */
    def tracesOf (cand: Candidate): Seq [Trace] =
        traces.filter (t => t.cand.key == cand.key || t.candTrim.key == cand.key)
    end tracesOf

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the first trace that led to a candidate: when and by whom it appeared.
     *  @param cand  the candidate
     */
    def originOf (cand: Candidate): Option [Trace] = tracesOf (cand).headOption

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the run's best equations with their scores, one per line, best first.
     */
    def equations: String =
        top.map (c => f"${c.score}%8.3f  ${c.smape}%8.3f  ${c.nTerms}%2d  ${c.formulaStr}").mkString ("\n")

end ExpRun


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SearchArm` trait is the contract every version of the method fulfils.
 */
trait SearchArm:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the arm's method (the method column of the results table).
     */
    def method: Method

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the arm's ablation tag ("" for the standard version).
     */
    def variant: String = ""

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the arm's row label (method tag plus variant).
     */
    def name: String = if variant.isEmpty then method.tag else s"${method.tag}-$variant"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether the arm is deterministic (run once, stream ignored).
     */
    def deterministic: Boolean = false

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Run the arm once and return its record.
     *  @param cfg     the dataset configuration
     *  @param scorer  the fit-and-score engine of the dataset (its training rows
     *                     are the only rows the arm may learn from)
     *  @param stream  the random number stream of this run
     */
    def run (cfg: DatasetConfig, scorer: FormulaScorer, stream: Int): ExpRun

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the wave frequencies from the periodogram of the TRAINING rows of
     *  the response (the only frequencies a proposal may use).
     *  @param cfg     the dataset configuration
     *  @param scorer  the fit-and-score engine
     *  @param k       the number of frequencies
     */
    protected def waveFreqs (cfg: DatasetConfig, scorer: FormulaScorer, k: Int): VectorD =
        EDA4TS.topFreqs (scorer.trainSet (cfg.resp), k)                 // a direct DFT (@see `EDA4TS.topFreqs`)
    end waveFreqs

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build the EDA profile of the training rows for the given exogenous variables.
     *  @param cfg     the dataset configuration
     *  @param scorer  the fit-and-score engine
     *  @param exo     the exogenous variables profiled (the shortlist or the pool)
     *  @param maxLag  the maximum lag examined
     *  @param nFreqs  the number of periodogram frequencies
     */
    protected def edaOf (cfg: DatasetConfig, scorer: FormulaScorer, exo: Seq [String],
                         maxLag: Int, nFreqs: Int): EDA4TS =
        val data = scorer.fullSet
        new EDA4TS (data (cfg.resp), exo.map (v => v -> data (v)).toMap, scorer.trSize, maxLag, nFreqs)
    end edaOf

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the exogenous variables a candidate's formula uses (none for a base model).
     *  @param cfg   the dataset configuration
     *  @param cand  the candidate
     */
    protected def exoOf (cfg: DatasetConfig, cand: Candidate): Seq [String] =
        if cand.formula == null then Seq ()
        else (cand.formula.preds.flatMap (_.vars).toSet - cfg.resp).toSeq.sorted
    end exoOf

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Wrap a best candidate as a record with its Train-and-Test QoF attached
     *  (the base's `rollValidate` on the testing rows, @see `FormulaScorer.tnt`).
     *  @param scorer  the fit-and-score engine
     *  @param cfg     the dataset configuration
     *  @param stream  the stream of the run
     *  @param best    the best candidate of the run
     *  @param top     the best candidates at the end (best first)
     *  @param traces  every proposal of the run in order
     */
    protected def record (scorer: FormulaScorer, cfg: DatasetConfig, stream: Int, best: Candidate,
                          top: Seq [Candidate] = Seq (), traces: Seq [Trace] = Seq ()): ExpRun =
        val qofTest = if best.failed then MatrixD.fill (scorer.hh, Fit.N_QoF, -0.0)
                      else if best.formula != null then scorer.tnt (best.formula)._2
                      else scorer.tnt (best.model)
        ExpRun (method, variant, stream, best, qofTest, top = top, traces = traces, exoUsed = exoOf (cfg, best))
    end record

end SearchArm

