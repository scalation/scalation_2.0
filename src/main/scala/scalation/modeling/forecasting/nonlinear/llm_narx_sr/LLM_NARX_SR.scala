
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Mon Sep 14 15:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: LLM-NARX-SR -- LLM-Guided Symbolic Regression over NARX Models for Time-Series Forecasting:
 *           the experiment harness
 *
 *  The method searches the STRUCTURE space of `ARX_Formula` models:
 *
 *      data -> exo screening -> EDA profile
 *           -> Loop { proposer -> parse -> fit & score [-> trimmer] -> population + traces }
 *           -> term library -> selector -> best equation  [-> stable core back to the proposer, next round]
 *           -> rolling validation on the testing rows -> plausibility, explanation, Pareto front
 *
 *  The proposer is an LLM, a genetic rule or random grammar sampling (@see
 *  `Proposer`); the trimmer (optional, per proposal) and the selector (over the
 *  library of every proposed term) are selection techniques on term columns
 *  (@see `SelectionTech`, `SubsetSearch`: the cross-entropy method `CEM_Selector`
 *  as the selector, any technique as either -- `SearchLoop.trim` keeps the
 *  proposal when no subset scores better).  Every searching arm is one `ArmSpec`
 *  -- proposer, trimmer, selector, library, rounds, prompt options -- run by
 *  `PipelineArm` (@see `ArmSpec`, `LLM_NARX_SR.mainArms`, `ablationArms`): `llm-loop`
 *  and `genetic`/`random` have no selector, `stat-cem` has no loop (its library is
 *  expanded from the grammar); the anchors are `BaselineArm` (V0, no search).
 *  This object holds the hyper-parameters
 *  (`hp`, one map like `AR.hp`), the client factories, the arm sets (main table
 *  and ablation table), runs arms x streams (`runGrid`) and nested prefixes of
 *  the series (`runFolds`), prints the results the way the rest of the project
 *  prints them -- the cross-arm table as a `MatrixD` with a header, and for the
 *  winners `showBest`: the REPORT block, the coefficient tables per horizon, the
 *  in-sample fitMap on the training rows and the base's own `rollValidate` +
 *  `diagnoseAll` on the testing rows (as `reportBest`/`aRX_DTest4` do) -- and
 *  lists the lineage of a run (`showTraces`).
 *
 *  Row protocol (@see `FormulaScorer`): ScalaTion's split -- the search learns
 *  from the training rows (`Model.trSize`), the reported number is the base's
 *  rolling validation on the testing rows.  LLM runs are reproducible through
 *  `CachedClient` (each stream's transcript is keyed by its prompt, which carries
 *  a "[run <stream>]" tag).
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
/** The `LLMModel` enum names the cloud models the experiments may call, each with
 *  its provider id and its USD per million prompt/reply tokens (update from the
 *  provider's price list), so a model and its prices cannot come apart.
 *  @param id        the provider's model id
 *  @param priceIn   USD per million prompt tokens
 *  @param priceOut  USD per million reply tokens
 */
enum LLMModel (val id: String, val priceIn: Double, val priceOut: Double):
    case GPT4oMini extends LLMModel ("gpt-4o-mini", 0.15, 0.60)
    case GPT4o     extends LLMModel ("gpt-4o", 2.50, 10.00)

    def prices: (Double, Double) = (priceIn, priceOut)                  // the pair the cost column uses
end LLMModel


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LLM_NARX_SR` object is the experiment harness: hyper-parameters, client
 *  factories, the arm sets, the grid and fold runners, the report and the
 *  ARX_D-style display of the winners.
 */
object LLM_NARX_SR:

    /** The search hyper-parameters (the same object as `FormulaScorer.hp`, so
     *  `LLM_NARX_SR.hp("rc") = 5` reaches every arm and scorer).
     */
    val hp: HyperParameter = FormulaScorer.hp

    val STREAMS   = Seq (0, 1, 2, 3, 4, 5, 6, 7)                        // random streams / LLM run tags
    val LLM_MODEL = LLMModel.GPT4oMini                                  // OpenAI model for the cloud experiments
    val CACHE_DIR = "llm_cache"                                         // CachedClient directory (a paper artifact)
    val RESULTS_DIR = "llm_narx_sr_results"                                   // one CSV per grid, a row appended as each run finishes

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a factory for cached cloud clients (OpenAI; the key is read from
     *  the environment by `OpenAIClient`).  Every call builds a fresh client;
     *  the cache directory makes repeated runs free.
     *  @param model  the model (@see `LLMModel`: its id and prices)
     *  @param temp   the sampling temperature
     */
    def cloudLLM (model: LLMModel = LLM_MODEL, temp: Double = 0.8): () => LLMClient =
        () => new CachedClient (new OpenAIClient (model.id, temp), CACHE_DIR)
    end cloudLLM

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a factory for cached local-server clients (e.g., ollama), or None
     *  when no server answers.
     */
    def localLLM (): Option [() => LLMClient] =
        if new LocalClient ().reachable () then Some (() => new CachedClient (new LocalClient (), CACHE_DIR))
        else None
    end localLLM

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a factory for a FINE-TUNED proposer served through an OpenAI-compatible
     *  endpoint (vLLM, ollama, ...), or None: the endpoint and the model name are
     *  read from the environment (LLM_NARX_SR_FT_ENDPOINT, LLM_NARX_SR_FT_MODEL) and the server
     *  must answer.  The fine-tuning itself is outside ScalaTion; this is the slot
     *  that lets a prompt-based and a fine-tuned proposer be compared under the
     *  same scorer, grammar and budget (@see `ftArms`).
     */
    def ftLLM (): Option [() => LLMClient] =
        val ep  = sys.env.get ("LLM_NARX_SR_FT_ENDPOINT")
        val mdl = sys.env.get ("LLM_NARX_SR_FT_MODEL")
        if ep.isDefined && mdl.isDefined && new LocalClient (ep.get, mdl.get).reachable () then
            Some (() => new CachedClient (new LocalClient (ep.get, mdl.get), CACHE_DIR, tag = "ft"))
        else None
    end ftLLM

//  A R M   S E T S

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the three classical anchors (the ARX structures without search and
     *  the random walk, fitted and scored by the same engine as every candidate).
     */
    def anchors: Seq [SearchArm] =
        Seq (new BaselineArm (Method.Baseline), new BaselineArm (Method.TunedARX),
             new BaselineArm (Method.SelARX), new BaselineArm (Method.RandomWalk))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Try one small completion on a client factory and return None when it
     *  answers, or the failure (its exception) when it does not.
     *  @param llm  the client factory
     */
    def probe (llm: () => LLMClient): Option [String] =
        try
            val reply = llm ().complete ("Reply with the single word OK.", "connection check")
            if reply == null then Some ("empty reply") else None
        catch case e: Exception => Some (e.toString)
    end probe

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build the arms of a list of specs on one client factory (@see `PipelineArm`).
     *  @param specs   the arm specs
     *  @param llm     the factory for a fresh LLM client per run (None => zero-LLM specs only)
     *  @param prices  the USD per million prompt/reply tokens
     */
    def armsOf (specs: Seq [ArmSpec], llm: Option [() => LLMClient] = None,
                prices: (Double, Double) = (0.0, 0.0)): Seq [SearchArm] =
        specs.map (s => new PipelineArm (s, llm, prices = prices))
    end armsOf

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the zero-LLM arms: the anchors, random grammar sampling, genetic
     *  search and the grammar-library CEM.
     */
    def zeroLLMArms: Seq [SearchArm] =
        anchors ++ armsOf (Seq (ArmSpec.random, ArmSpec.genetic, ArmSpec.statCEM))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the LLM arms for a client factory: the LLM loop alone (V1), the
     *  model llm-cem (V4) and its ablations without the EDA block ("blind") and
     *  with per-proposal trimming ("trim", V3).
     *  @param llm     the factory for a fresh LLM client per run
     *  @param prices  the USD per million prompt/reply tokens
     */
    def llmArms (llm: () => LLMClient, prices: (Double, Double) = LLM_MODEL.prices): Seq [SearchArm] =
        armsOf (Seq (ArmSpec.llmLoop, ArmSpec.llmCEM, ArmSpec.llmCEM.copy (eda = false),
                     ArmSpec.llmCEM.copy (trimmer = Some (SelectionTech.CEM))), Some (llm), prices)
    end llmArms

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the fine-tuned-proposer arms (tag "ft", no API cost): the LLM
     *  loop and llm-cem driven by a served fine-tuned model (@see `ftLLM`).
     *  @param llm  the factory for the fine-tuned model's client
     */
    def ftArms (llm: () => LLMClient): Seq [SearchArm] =
        armsOf (Seq (ArmSpec.llmLoop.copy (tag = "ft"), ArmSpec.llmCEM.copy (tag = "ft")), Some (llm))
    end ftArms

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the arms of the paper's MAIN table: anchors, stat-cem, genetic and,
     *  with a client, llm-loop and llm-cem.
     *  @param llm     the client factory (None => zero-LLM arms only)
     *  @param prices  the USD per million prompt/reply tokens
     */
    def mainArms (llm: Option [() => LLMClient] = None, prices: (Double, Double) = LLM_MODEL.prices): Seq [SearchArm] =
        anchors ++ armsOf (Seq (ArmSpec.statCEM, ArmSpec.genetic)) ++
        llm.map (f => armsOf (Seq (ArmSpec.llmLoop, ArmSpec.llmCEM), Some (f), prices)).getOrElse (Seq ())
    end mainArms

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the arms of the ABLATION table: random sampling and, with a client,
     *  llm-cem without the EDA block, with trimming inside the loop, and with two
     *  LLM <-> selector rounds.
     *  @param llm     the client factory (None => zero-LLM arms only)
     *  @param prices  the USD per million prompt/reply tokens
     */
    def ablationArms (llm: Option [() => LLMClient] = None, prices: (Double, Double) = LLM_MODEL.prices): Seq [SearchArm] =
        armsOf (Seq (ArmSpec.random)) ++
        llm.map (f => armsOf (Seq (ArmSpec.llmCEM.copy (eda = false),
                                   ArmSpec.llmCEM.copy (trimmer = Some (SelectionTech.CEM)),
                                   ArmSpec.llmCEM.copy (rounds = 2)), Some (f), prices)).getOrElse (Seq ())
    end ablationArms

//  R U N N E R S

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build the scorer of a dataset (ScalaTion's split, the hp settings).
     *  @param dataset  the map of named data columns
     */
    def scorerOf ()(using dataset: Map [String, VectorD]): FormulaScorer = new FormulaScorer (hp("hh").toInt)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Run a grid of arms over streams on one dataset (deterministic arms run
     *  once), print the report and, when `show`, the ARX_D-style output
     *  (`showBest`, which prints and plots) and the traces of the overall best
     *  run -- the best run of every method is listed in one line each; return the runs.
     *  @param title    the experiment title (dataset + configuration)
     *  @param cfg      the dataset configuration
     *  @param arms     the arms to run
     *  @param streams  the random streams / run tags
     *  @param show       whether to call `showBest` (prints and plots) for the winners
     *  @param explainer  the explainer of the winning equation (null => none)
     *  @param dataset    the map of named data columns
     */
    def runGrid (title: String, cfg: DatasetConfig, arms: Seq [SearchArm], streams: Seq [Int] = STREAMS,
                 show: Boolean = true, explainer: Explainer = null)(using dataset: Map [String, VectorD]): Seq [ExpRun] =
        val scorer = scorerOf ()
        val runs   = VEC [ExpRun] ()
        banner (s"$title: ${cfg.name}, exogenous mode ${cfg.exoMode}, using ${scorer.shortlist (cfg).mkString (", ")} of ${cfg.exo.mkString (", ")}")
        println (s"training rows ${scorer.trRng} = fit rows ${scorer.fitRng} + validation rows ${scorer.valRng}; testing rows ${scorer.teRng}")
        val csv = csvFile (title)                                       // the rows are saved as they come
        for arm <- arms; s <- (if arm.deterministic then Seq (-1) else streams) do
            banner (s"${cfg.name}: ${arm.name} run, stream = $s")
            val r = try arm.run (cfg, scorer, s)
                    catch case e: Exception =>
                        println (s"ERROR @ LLM_NARX_SR.runGrid: ${arm.name} (stream $s) failed: $e -- recorded as a failed run")
                        ExpRun (arm.method, arm.variant, s, scorer.failed (null, 0, arm.name, e.toString),
                                MatrixD.fill (scorer.hh, Fit.N_QoF, -0.0), error = e.toString)
            if r.error != null then println (s"NOTE @ LLM_NARX_SR.runGrid: ${arm.name} (stream $s): ${r.error}")
            runs += r
            appendCSV (csv, r, runs.size == 1)
        end for
        report (title, runs.toSeq)
        val fitted = runs.toSeq.filterNot (_.failed)
        if show && fitted.nonEmpty then
            banner (s"$title -- best run per method (by the selection score)")
            for (m, rs) <- byMethod (fitted) do
                val b = rs.minBy (_.best.score)
                println (f"$m%-14s stream ${b.stream}%2d  score ${b.best.score}%8.3f  val sMAPE ${b.valSmape}%8.3f  TnT sMAPE ${b.testSmape}%8.3f  ${b.best.formulaStr}")
            val top = bestOf (fitted)
            banner (s"$title -- overall best: ${top.name} (stream ${top.stream})")
            showBest (top, scorer, explainer)
            showPareto (top, scorer)
            showTraces (top)
        runs.toSeq
    end runGrid

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the CSV file of a grid in `RESULTS_DIR` (created when absent): the title
     *  with every character outside [A-Za-z0-9._-] replaced by '_', and a time stamp.
     *  @param title  the experiment title
     */
    def csvFile (title: String): java.io.File =
        val dir = new java.io.File (RESULTS_DIR)
        if ! dir.exists () then dir.mkdirs ()
        val stamp = java.time.LocalDateTime.now ().format (java.time.format.DateTimeFormatter.ofPattern ("yyyyMMdd_HHmm"))
        new java.io.File (dir, s"${title.replaceAll ("[^A-Za-z0-9._-]", "_")}_$stamp.csv")
    end csvFile

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Append one run as a CSV row (the header first when `header`): the row label,
     *  the stream, the in-sample, validation and TnT sMAPE (per horizon and mean),
     *  the terms, free parameters, budget columns, the exogenous variables, the error
     *  (if any) and the equation.  Never throws: a write failure is printed.
     *  @param file    the CSV file
     *  @param r       the run
     *  @param header  whether to write the header line first
     */
    def appendCSV (file: java.io.File, r: ExpRun, header: Boolean): Unit =
        try
            val hh  = r.smapeTrain.dim
            val w   = new java.io.FileWriter (file, true)
            if header then
                w.write ((Seq ("method", "stream") ++ (1 to hh).map (h => s"train_h$h") ++ Seq ("train_mean", "val_mean") ++
                          (1 to hh).map (h => s"tnt_h$h") ++ Seq ("tnt_mean", "terms", "nfree", "proposed", "valid", "failed", "fits",
                                                                  "llm", "ktok", "cost", "exo", "error", "equation")).mkString (",") + "\n")
            def q (s: String): String = "\"" + (if s == null then "" else s.replace ('\n', ' ').replace ("\"", "'")) + "\""
            val row = Seq (r.name, r.stream.toString) ++ r.smapeTrain.toArray.map (_.toString) ++
                      Seq (r.best.smape.toString, r.valSmape.toString) ++ r.smapeTest.toArray.map (_.toString) ++
                      Seq (r.testSmape, r.best.nTerms, r.best.nFree, r.nProposed, r.nValid, r.nFailed, r.nEval,
                           r.llmCalls, r.tokens / 1000.0, r.cost).map (_.toString) ++
                      Seq (q (r.exoUsed.mkString (" ")), q (r.error), q (r.best.formulaStr + r.best.why))
            w.write (row.mkString (",") + "\n")
            w.close ()
        catch case e: Exception => println (s"ERROR @ LLM_NARX_SR.appendCSV: ${e.getMessage}")
    end appendCSV

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Run the grid on nested prefixes of the series: for each cut the series is
     *  its first `cut` rows, split by `Model.trSize` of its own length, so every
     *  fold tests a different stretch of time.  Returns the runs per fold (each
     *  fold's own results table is printed by `runGrid`, without `showBest`).
     *  @param title    the experiment title
     *  @param cfg      the dataset configuration
     *  @param arms     the arms to run
     *  @param cuts     the series length per fold (ascending; the last is the full series)
     *  @param streams  the random streams / run tags
     *  @param dataset  the map of named data columns
     */
    def runFolds (title: String, cfg: DatasetConfig, arms: Seq [SearchArm], cuts: Seq [Int],
                  streams: Seq [Int] = STREAMS)(using dataset: Map [String, VectorD]): Seq [Seq [ExpRun]] =
        for cut <- cuts yield
            val sub = dataset.map ((k, v) => k -> v(0 until cut))
            runGrid (s"$title [rows 0..${cut-1}]", cfg, arms, streams, show = false)(using sub)
    end runFolds

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Group runs by row label (method tag + variant), in first-seen order.
     *  @param runs  the runs
     */
    def byMethod (runs: Seq [ExpRun]): Seq [(String, Seq [ExpRun])] =
        runs.map (_.name).distinct.map (m => m -> runs.filter (_.name == m))
    end byMethod

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the run with the best selection score (in-sample, the criterion
     *  the search itself used -- never the testing rows), over all runs or over
     *  one method (and variant).
     *  @param runs     the runs
     *  @param method   the method to restrict to (null => all)
     *  @param variant  the variant to restrict to (null => any)
     */
    def bestOf (runs: Seq [ExpRun], method: Method = null, variant: String = null): ExpRun =
        val pool = runs.filter (r => ! r.failed &&                      // a failed run is never the best
                                     (method == null || r.method == method) && (variant == null || r.variant == variant))
        require (pool.nonEmpty, s"bestOf: no fitted runs${if method == null then "" else s" for $method"}")
        pool.minBy (_.best.score)
    end bestOf

//  O U T P U T

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a candidate as a trained ScalaTion model (forecasts filled for
     *  every row, skip set): the base model it holds, or its formula fitted by
     *  the scorer on the training rows.
     *  @param cand    the candidate
     *  @param scorer  the fit-and-score engine of the dataset
     */
    def toModel (cand: Candidate, scorer: FormulaScorer): Forecaster =
        if cand.model != null then scorer.train (cand.model) else scorer.fitFull (cand.formula)   // all training rows
    end toModel

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show a run's best model the way the base shows a model (@see `reportBest`,
     *  `aRX_DTest4`): the REPORT block (name, hyper-parameters, features,
     *  parameters, h = 1 QoF), the coefficient tables `summary ()` and `summary (h)`
     *  for h = 2..hh (each after diagnosing its horizon), the in-sample fitMap on
     *  the training rows, then the base's own `rollValidate` (plotted) and
     *  `diagnoseAll` on the testing rows (which prints the TnT fitMap and plots
     *  every horizon), and the convergence of the search as a `PlotM`; in between,
     *  the term-importance table (@see `Explainer.importance`) and, with an
     *  explainer, the LLM's checked explanation of the equation.
     *  @param run        the run
     *  @param scorer     the fit-and-score engine of the dataset
     *  @param explainer  the explainer (null => no explanation)
     */
    def showBest (run: ExpRun, scorer: FormulaScorer, explainer: Explainer = null): Unit =
        val hh = scorer.hh
        banner (s"${run.name} (stream ${run.stream}): ${run.best.formulaStr}")
        if run.best.failed then { println ("the best candidate failed to fit -- nothing to show"); return }
        val mod = toModel (run.best, scorer)
        val q   = scorer.qofTrain (mod)
        println (mod.report (q(0)))                                     // REPORT block (h = 1 QoF)
        mod match
        case n: NARX_SR_D if n.nFree > 0 =>
            println (s"fitted nonlinear parameters (${n.nFree} free, by $OPTIMIZER_NL) = ${n.parameter_nl}")
        case _ => ()
        end match
        mod match
        case d: Forecaster_D =>
            for h <- 1 to hh do
                scorer.diagnoseH (d, scorer.trRng, h)                   // `summary` reads the last diagnose
                println (if h == 1 then d.summary () else d.summary (h))
        case _ => ()
        end match
        mod match
        case d: ARX_D if run.best.formula != null =>
            banner ("term importance at h = 1 (coefficient x std of the term's column on the training rows)")
            println (Explainer.importanceTable (Explainer.importance (d, run.best.formula, scorer.trSize)))
        case _ => ()
        end match
        if explainer != null then
            try                                                         // an unreachable LLM does not end the report
                val ex = explainer.explain (run.best, scorer)
                banner (s"the LLM's explanation -- ${ex.tableRow}")
                for c <- ex.claims do println (c)
                println (ex.text)
            catch case e: Exception => println (s"ERROR @ LLM_NARX_SR.showBest: no explanation -- $e")
        banner (s"In-Sample fitMap QoF on the training rows ${scorer.trRng} (skip = ${scorer.skip})")
        println (Fit.showFitMap (q.ᵀ))
        banner (s"TnT: rollValidate (rc = ${scorer.rc}) + diagnoseAll on the testing rows ${scorer.teRng}")
        mod.setSkip ()                                                  // diagnose: don't skip any rows
        mod.rollValidate (scorer.rc, doPlot = DO_PLOT)                  // TnT with rolling validation (plot when DO_PLOT)
        mod.diagnoseAll (mod.getY, mod.getYf, scorer.teRng)             // only diagnose on the testing set
        if DO_PLOT && run.conv.nonEmpty then
            val scores = VectorD (run.conv.map (_.bestScore).toArray)
            new PlotM (null, MatrixD (scores), Array ("best score"),
                       s"convergence of ${run.name} (stream ${run.stream}): best score vs. round", lines = true)
    end showBest

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the complexity-accuracy front of a run: over every equation the run
     *  scored (traces, as proposed and as kept, the final population, the winner),
     *  the best in-sample score at each number of terms, keeping only the points no
     *  smaller equation beats -- the Pareto front of terms vs score, fewest first.
     *  @param run  the run
     */
    def paretoFront (run: ExpRun): Seq [Candidate] =
        val cands = (run.traces.flatMap (t => Seq (t.cand, t.candTrim)) ++ run.top :+ run.best)
                        .filterNot (_.failed).distinctBy (_.key)
        val byK   = cands.groupBy (_.nTerms).values.map (_.minBy (_.score)).toSeq.sortBy (_.nTerms)
        val front = VEC [Candidate] ()
        var best  = Double.MaxValue
        for c <- byK if c.score < best do { front += c; best = c.score }
        front.toSeq
    end paretoFront

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print the complexity-accuracy front of a run: per point its terms, in-sample
     *  sMAPE and score (training rows) and, when `tnt`, its TnT sMAPE (the base's
     *  rolling validation on the testing rows, one refit per point), so the reader
     *  sees where the selection criterion (J, @see `FormulaScorer.scoreOf`) placed the winner (*).
     *  @param run     the run
     *  @param scorer  the fit-and-score engine of the dataset
     *  @param tnt     whether to add the TnT sMAPE of every point
     */
    def showPareto (run: ExpRun, scorer: FormulaScorer, tnt: Boolean = true): Unit =
        val front = paretoFront (run)
        banner (s"${run.name} (stream ${run.stream}) -- complexity vs accuracy front (${front.size} points, * = the winner)")
        println (f"${"terms"}%5s ${"train sMAPE"}%11s ${"val sMAPE"}%9s ${"score"}%8s ${if tnt then f"${"TnT sMAPE"}%10s " else ""}  equation")
        for c <- front do
            val te = if ! tnt then ""
                     else
                         val q = if c.model != null then scorer.tnt (toModel (c, scorer)) else scorer.tnt (c.formula)._2
                         f"${q(?, QoF.smape.ordinal).mean}%10.3f "
            println (f"${c.nTerms}%5d ${c.smape}%11.3f ${c.smapeVal}%9.3f ${c.score}%8.3f $te${if c.key == run.best.key then "*" else " "} ${c.formulaStr.replace ('\n', ' ')}")
        end for
    end showPareto

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the cross-arm results table as a matrix with a header (one row per
     *  run): in-sample sMAPE per horizon and its mean (training rows), TnT sMAPE
     *  per horizon and its mean (testing rows), terms, budget columns and the
     *  number of exogenous variables used; printable and `write`-able (`MatrixD.write`).
     *  @param runs  the runs
     */
    def resultsTable (runs: Seq [ExpRun]): (MatrixD, Array [String]) =
        val hh  = runs.map (_.smapeTrain.dim).max
        val hdr = (1 to hh).map (h => s"train_h$h").toArray ++ Array ("train_mean", "val_mean") ++      // val_mean
                  (1 to hh).map (h => s"tnt_h$h").toArray ++ Array ("tnt_mean", "terms", "nfree", "valid", "fail", "fits",
                                                                     "llm", "ktok", "cost", "n_exo")
        val tab = new MatrixD (runs.size, hdr.length)
        for (r, i) <- runs.zipWithIndex do
            tab(i) = r.smapeTrain ++ VectorD (r.best.smape, r.valSmape) ++ r.smapeTest ++
                     VectorD (r.testSmape, r.best.nTerms.toDouble, r.best.nFree.toDouble, r.validity, r.nFailed.toDouble, r.nEval.toDouble,
                              r.llmCalls.toDouble, r.tokens / 1000.0, r.cost, r.exoUsed.size.toDouble)
        end for
        (tab, hdr)
    end resultsTable

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print the results: one line per run (in-sample and TnT sMAPE by horizon,
     *  means, terms, budget, exogenous variables and the best equation), the
     *  per-method means over streams, and the convergence tables.
     *  @param title  the experiment title
     *  @param runs   the collected runs
     *  @param conv   whether to print the convergence tables
     */
    def report (title: String, runs: Seq [ExpRun], conv: Boolean = true): Unit =
        val hh = runs.map (_.smapeTrain.dim).max
        banner (s"$title -- results (in-sample on the fit rows | validation | TnT rolling on the testing rows)")
        val hHdr = (1 to hh).map (h => f"${s"h=$h"}%7s").mkString (" ")
        println (f"${"method"}%-14s ${"strm"}%4s | $hHdr ${"mean"}%7s | ${"val"}%7s | $hHdr ${"mean"}%7s | ${"terms"}%5s ${"free"}%4s ${"valid"}%6s " +
                 f"${"fail"}%4s ${"fits"}%6s ${"LLM"}%4s ${"ktok"}%6s ${"cost"}%7s  exo  best equation")
        for r <- runs do
            val tr = (0 until hh).map (h => f"${r.smapeTrain(h)}%7.3f").mkString (" ")
            val te = (0 until hh).map (h => f"${r.smapeTest(h)}%7.3f").mkString (" ")
            val vd = if r.nProposed == 0 then f"${"--"}%6s" else f"${r.validity * 100}%5.1f%%"
            val eq = if r.failed then s"FAILED: ${if r.error != null then r.error else r.best.reason}"
                     else if r.noImprovement then s"no improvement over the seed (${r.best.formulaStr.replace ('\n', ' ')})"
                     else r.best.formulaStr.replace ('\n', ' ')
            println (f"${r.name}%-14s ${r.stream}%4d | $tr ${r.best.smape}%7.3f | ${r.valSmape}%7.3f | $te ${r.testSmape}%7.3f | ${r.best.nTerms}%5d ${r.best.nFree}%4d $vd " +
                     f"${r.nFailed}%4d ${r.nEval}%6d ${r.llmCalls}%4d ${r.tokens / 1000.0}%6.1f ${r.cost}%7.4f  " +
                     s"${r.exoUsed.mkString (",")}  $eq")
        end for

        banner (s"$title -- per-method means over streams (fitted runs)")
        println (f"${"method"}%-14s ${"n"}%3s | ${"train"}%7s ${"val"}%7s ${"tnt"}%7s | $hHdr (tnt) | ${"terms"}%6s ${"fits"}%8s ${"LLM"}%6s ${"cost"}%8s")
        for (m, rs0) <- byMethod (runs) do
            val rs = rs0.filterNot (_.failed)                           // failed runs are left out of the means
            if rs.isEmpty then println (f"$m%-14s ${0}%3d | every run failed")
            else
                val mTr  = rs.map (_.best.smape).sum / rs.size
                val mVa  = rs.map (_.valSmape).sum / rs.size
                val mTe  = rs.map (_.testSmape).sum / rs.size
                val teH  = (0 until hh).map (h => f"${rs.map (_.smapeTest(h)).sum / rs.size}%7.3f").mkString (" ")
                val mK   = rs.map (_.best.nTerms).sum.toDouble / rs.size
                val mFit = rs.map (_.nEval).sum.toDouble / rs.size
                val mLLM = rs.map (_.llmCalls).sum.toDouble / rs.size
                val mCst = rs.map (_.cost).sum / rs.size
                println (f"$m%-14s ${rs.size}%3d | $mTr%7.3f $mVa%7.3f $mTe%7.3f | $teH | $mK%6.1f $mFit%8.1f $mLLM%6.1f $mCst%8.4f")
        end for

        val (tab, hdr) = resultsTable (runs)
        banner (s"$title -- results table (MatrixD, header: ${hdr.mkString (", ")})")
        println (tab)

        if conv then
            banner (s"$title -- convergence (best-so-far per round)")
            for r <- runs if r.conv.nonEmpty do
                println (s"# ${r.name} stream=${r.stream}")
                println ("round  proposed  bestScore  bestSMAPE")
                for c <- r.conv do println (f"${c.round}%5d  ${c.nProposed}%8d  ${c.bestScore}%9.3f  ${c.bestSmape}%9.3f")
            end for
    end report

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print the lineage of a run: every proposal in order (round, source,
     *  trimmer, score of f -> score of fTrim, f -> fTrim with the dropped terms),
     *  and the origin of the winner.
     *  @param run  the run
     *  @param k    the number of traces to print (-1 => all)
     */
    def showTraces (run: ExpRun, k: Int = -1): Unit =
        banner (s"${run.name} (stream ${run.stream}) -- traces (${run.traces.size} entries)")
        println ("rnd  source    trimmer      f score ->  fTrim score  proposed  [->  kept  (dropped)]")
        for t <- (if k < 0 then run.traces else run.traces.take (k)) do println (t)
        run.originOf (run.best) match
        case Some (t) => println (s"winner ${run.best.formulaStr}: first proposed in round ${t.round} by ${t.source}" +
                                  (if t.trimmed then s", trimmed by ${t.trimmer.get} from ${t.f.formulaStr}" else "") +
                                  s"; reached ${run.tracesOf (run.best).size} time(s)")
        case None     => println (s"winner ${run.best.formulaStr}: no trace (a base model)")
    end showTraces

end LLM_NARX_SR


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lLM_NARX_SRTest` main function runs the ZERO-LLM grid on the Covid-19
 *  dataset (icu, pRate: we choose the exogenous variables) for two streams --
 *  the anchors, random, genetic and stat-cem -- prints the results table and
 *  the ARX_D-style output of the winners.  No network, no key.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.lLM_NARX_SRTest
 */
@main def lLM_NARX_SRTest (): Unit =

    import LLM_NARX_SR._

    val cfg = Datasets.COVID2
    given dataset: Map [String, VectorD] = cfg.load ()

    val runs = runGrid ("Covid-19 icu/pRate (zero-LLM arms)", cfg, zeroLLMArms, streams = Seq (0, 1))
    val top  = bestOf (runs)
    banner (f"best run: ${top.name} stream ${top.stream}, in-sample sMAPE ${top.best.smape}%.3f, TnT sMAPE ${top.testSmape}%.3f")
    println (top.best.formulaStr)

    assert (runs.exists (_.method == Method.StatCEM) && runs.exists (_.method == Method.Genetic), "all zero-LLM arms must run")
    assert (runs.forall (r => ! r.best.failed), "every arm must return a fitted equation")
    assert (runs.forall (r => r.smapeTest.forall (v => ! v.isNaN)), "every run carries a TnT QoF")
    assert (runs.filter (_.method == Method.Baseline).head.exoUsed == Seq ("icu", "pRate"), "Given mode uses the given exogenous variables")
    println ("lLM_NARX_SRTest: passed")

end lLM_NARX_SRTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lLM_NARX_SRTest2` main function runs EVERY arm on the Covid-19 dataset
 *  with a scripted `MockLLMClient` standing in for the LLM, in both exogenous
 *  modes -- icu/pRate given, and the system choosing from the pool -- so the
 *  whole grid, the report, `bestOf` and `showBest` are exercised offline.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.lLM_NARX_SRTest2
 */
@main def lLM_NARX_SRTest2 (): Unit =

    import LLM_NARX_SR._

    val replies = VEC (
        "y ~= y~1 + x1~2\ny ~= y~(2,1) + log1p(x2~1)\ny ~= y~1 + x9~1",
        "y ~= y~1 + y~2 + x1~1 * y~1\nTFORM: Log1p\ny ~= y~(3,1) + x1~(2,1)",
        "y ~= y~1 + wave(1)\ny ~= y~1 + sq(x2~2)\ny ~= y~(4,1) + x2~(3,1)")
    val mock: () => LLMClient = () => new MockLLMClient (replies.clone ())

    for cfg <- Seq (Datasets.COVID2, Datasets.COVID_ALL) do
        given dataset: Map [String, VectorD] = cfg.load ()
        val runs = runGrid (s"${cfg.name} (all arms, scripted LLM)", cfg,
                            mainArms (Some (mock), (0.0, 0.0)) ++ ablationArms (Some (mock), (0.0, 0.0)), streams = Seq (0),
                            show = cfg.exoMode == ExoMode.Auto)          // one showBest (plots) for the Auto run only
        val names = runs.map (_.name).distinct
        println (s"arms run: ${names.mkString (", ")}")
        assert (names.size == 12, s"expected 12 arms, got ${names.size}")
        assert (runs.filter (_.method == Method.LLMCEM).forall (_.llmCalls > 0), "LLM arms must consult the client")
        assert (runs.forall (r => r.smapeTest.forall (v => ! v.isNaN)), "every run carries a TnT QoF")
        if cfg.exoMode == ExoMode.Auto then
            val base = runs.filter (_.method == Method.Baseline).head
            println (s"Auto mode: the baseline used ${base.exoUsed.mkString (", ")}")
            assert (base.exoUsed.nonEmpty && base.exoUsed.size <= hp("exoK").toInt, "Auto mode picks at most exoK variables")
        println (s"best overall: ${bestOf (runs).name} -> ${bestOf (runs).best.formulaStr}")
    end for
    println ("lLM_NARX_SRTest2: passed")

end lLM_NARX_SRTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lLM_NARX_SRTest3` main function runs the main and ablation grids with a
 *  CACHED cloud client (OpenAI, key from OPENAI_API_KEY; every transcript is
 *  written to `llm_cache`, so a repeated run costs nothing), or with a local
 *  model server when one answers and no key is set; the client is PROBED first
 *  (one small completion) and the LLM arms are skipped, with the failure printed,
 *  when it does not answer (rev3.2: on Oct 4 an unreachable api.openai.com ended
 *  the grid after 11 minutes of zero-LLM runs).  Five streams (0..4), so the
 *  stream-paired tests have n = 5; the fit budget of every loop is `hp("maxFits")`
 *  = 100 here, the genetic arm's typical use, so the LLM is compared at an equal
 *  number of model fits.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.lLM_NARX_SRTest3
 */
@main def lLM_NARX_SRTest3 (): Unit =

    import LLM_NARX_SR._

    val cfg = Datasets.COVID2
    given dataset: Map [String, VectorD] = cfg.load ()

    val candidate: Option [(() => LLMClient, (Double, Double), String)] =
        if sys.env.contains ("OPENAI_API_KEY") then Some ((cloudLLM (), LLM_MODEL.prices, s"OpenAI ${LLM_MODEL.id}"))
        else localLLM ().map (f => (f, (0.0, 0.0), "local model server"))
    val llm = candidate.flatMap { (f, p, nm) =>
        probe (f) match
        case None      => println (s"LLM client: $nm answers"); Some ((f, p))
        case Some (why) =>
            println (s"LLM client: $nm does NOT answer -- $why\n(check the network / proxy / key, or start a local server) -- running the zero-LLM arms only")
            None
    }
    if candidate.isEmpty then println ("no OPENAI_API_KEY and no local model server -- running the zero-LLM arms only")
    hp("maxFits") = 100                                                 // the same number of model fits for every loop arm
    val streams   = Seq (0, 1, 2, 3, 4)
    val prices    = llm.map (_._2).getOrElse ((0.0, 0.0))
    val explainer = llm.map (l => new Explainer (l._1 (), cfg)).orNull      // the winner's checked explanation
    val ft        = ftLLM ().map (ftArms).getOrElse (Seq ())                 // a served fine-tuned proposer, if any
    if ft.nonEmpty then println ("fine-tuned proposer found (LLM_NARX_SR_FT_ENDPOINT) -- adding the ft arms to the ablation")
    val main = runGrid ("Covid-19 icu/pRate (main table)", cfg, mainArms (llm.map (_._1), prices), streams = streams,
                        explainer = explainer)
    val abl  = runGrid ("Covid-19 icu/pRate (ablation table)", cfg, ablationArms (llm.map (_._1), prices) ++ ft,
                        streams = streams, explainer = explainer)
    Significance.reportSignificance ("Covid-19 icu/pRate (main table)", main, if llm.isDefined then Method.LLMCEM else Method.Genetic)
    val top  = bestOf (main ++ abl, if llm.isDefined then Method.LLMCEM else Method.StatCEM)
    banner (f"${top.name}: best stream ${top.stream}, in-sample ${top.best.smape}%.3f, validation ${top.valSmape}%.3f, TnT ${top.testSmape}%.3f")
    println (top.best.formulaStr)
    hp("maxFits") = 0

end lLM_NARX_SRTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lLM_NARX_SRTest4` main function runs the zero-LLM arms on the ILI dataset
 *  as a three-fold evaluation on nested prefixes (each prefix split by
 *  `Model.trSize`) and prints, per arm, the mean TnT sMAPE of each fold; skips
 *  if the ILI data file is not available.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.lLM_NARX_SRTest4
 */
@main def lLM_NARX_SRTest4 (): Unit =

    import LLM_NARX_SR._

    val cfg = Datasets.ILI
    val loaded = try Some (cfg.load ())
                 catch case e: Exception =>
                     println (s"ILI dataset not available -- skipping: ${e.getMessage}")
                     None
    loaded match
    case None => ()
    case Some (data) =>
        given dataset: Map [String, VectorD] = data
        val m     = data ("y").dim
        val cuts  = Seq (m - 104, m - 52, m)
        val folds = runFolds ("ILI (zero-LLM arms)", cfg, zeroLLMArms, cuts, streams = Seq (0, 1))
        banner ("ILI (zero-LLM arms) -- mean TnT sMAPE per fold")
        for nm <- folds.flatten.map (_.name).distinct do
            val cells = folds.map (rs =>
                val xs = rs.filter (r => r.name == nm && ! r.best.failed).map (_.testSmape)
                if xs.isEmpty then f"${"--"}%9s" else f"${xs.sum / xs.size}%9.3f").mkString (" ")
            println (f"$nm%-14s $cells")
        end for

end lLM_NARX_SRTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lLM_NARX_SRTest5` main function runs the ZERO-LLM grid with the SYSTEM
 *  choosing the exogenous variables from the Covid-19 pool (`Datasets.COVID_ALL`,
 *  `ExoMode.Auto`): the screening's shortlist is what the anchors and the grammar
 *  arms build on, and the `exo` column of the table shows what each winner used.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.lLM_NARX_SRTest5
 */
@main def lLM_NARX_SRTest5 (): Unit =

    import LLM_NARX_SR._

    val cfg = Datasets.COVID_ALL
    given dataset: Map [String, VectorD] = cfg.load ()
    val scorer = scorerOf ()
    val (ranks, short) = ExoScreen.shortlist (dataset (cfg.resp), cfg.exo.map (v => v -> dataset (v)).toMap,
                                              hp("maxLag").toInt, scorer.trSize, hp("exoK").toInt)
    banner ("exogenous screening on the training rows")
    for r <- ranks do println (r)
    println (s"shortlist: ${short.mkString (", ")}")
    assert (scorer.shortlist (cfg) == short, "the configuration's shortlist is the screening's")

    val runs = runGrid ("Covid-19 auto exo (zero-LLM arms)", cfg, zeroLLMArms, streams = Seq (0, 1))
    val top  = bestOf (runs)
    banner (f"best run: ${top.name} stream ${top.stream}, in-sample sMAPE ${top.best.smape}%.3f, TnT sMAPE ${top.testSmape}%.3f, exo ${top.exoUsed.mkString (", ")}")
    println (top.best.formulaStr)
    assert (runs.forall (r => ! r.best.failed), "every arm must return a fitted equation")
    assert (runs.filter (_.method == Method.Baseline).head.exoUsed == short, "the baseline builds on the shortlist")
    println ("lLM_NARX_SRTest5: passed")

end lLM_NARX_SRTest5


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lLM_NARX_SRTest6` main function compares the SELECTION CRITERIA on the
 *  Covid-19 dataset (icu, pRate): the zero-LLM arms over three streams under the
 *  in-sample J of rev3.0 (hp("valFrac") = 0), the one-block validation J
 *  (valFrac = 0.25, valFolds = 1) and the rolling-origin validation J (valFolds = 3),
 *  and prints, per criterion, the results table and the selection check -- the
 *  Spearman correlation between the score the search minimized and the TnT sMAPE
 *  on the testing rows (the Oct 4 grids: -0.355 under the in-sample J).  The
 *  settings are restored at the end.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.lLM_NARX_SRTest6
 */
@main def lLM_NARX_SRTest6 (): Unit =

    import LLM_NARX_SR._

    val cfg = Datasets.COVID2
    given dataset: Map [String, VectorD] = cfg.load ()
    val (frac0, folds0) = (hp("valFrac").toDouble, hp("valFolds").toInt)

    val settings = Seq (("in-sample J (rev3.0)", 0.0, 1), ("validation J, one block", 0.25, 1), ("validation J, 3 rolling blocks", 0.25, 3))
    val rows = for (label, frac, folds) <- settings yield
        hp("valFrac")  = frac
        hp("valFolds") = folds
        val runs = runGrid (s"Covid-19 icu/pRate -- $label", cfg, zeroLLMArms, streams = Seq (0, 1, 2), show = false)
        val rho  = Significance.selectionCheck (runs)
        val ok   = runs.filterNot (_.failed)
        val byM  = byMethod (ok).map ((m, rs) => f"$m%-12s ${rs.map (_.testSmape).sum / rs.size}%7.2f")
        (label, rho, byM)
    banner ("Covid-19 icu/pRate -- the selection criteria compared (TnT sMAPE means per arm; Spearman of score vs TnT)")
    for (label, rho, byM) <- rows do
        println (f"$label%-32s Spearman = $rho%+6.3f   ${byM.mkString ("  ")}")
    hp("valFrac")  = frac0
    hp("valFolds") = folds0

end lLM_NARX_SRTest6


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lLM_NARX_SRTest7` main function runs the ZERO-LLM grid on the first year of
 *  ETTh1 (hourly oil temperature; the system chooses two of the six loads) for two
 *  streams, prints the results table and the ARX_D-style output of the winners;
 *  skips if the data file is not available.  Horizons h = 1..6 hours, lags 1..6;
 *  the daily cycle is left to the wave terms.  Expect the oil temperature to be
 *  very persistent (small sMAPE for every arm): the question is whether the lagged
 *  loads add anything to the lags of y.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.lLM_NARX_SRTest7
 */
@main def lLM_NARX_SRTest7 (): Unit =

    import LLM_NARX_SR._

    val cfg = Datasets.ETTH1_1Y
    val loaded = try Some (cfg.load ())
                 catch case e: Exception =>
                     println (s"ETTh1 dataset not available -- skipping: ${e.getMessage}")
                     None
    for data <- loaded do
        given dataset: Map [String, VectorD] = data
        val runs = runGrid ("ETTh1 first year (zero-LLM arms)", cfg, zeroLLMArms, streams = Seq (0, 1))
        val top  = bestOf (runs)
        banner (f"best run: ${top.name} stream ${top.stream}, in-sample sMAPE ${top.best.smape}%.3f, validation ${top.valSmape}%.3f, TnT sMAPE ${top.testSmape}%.3f, exo ${top.exoUsed.mkString (", ")}")
        println (top.best.formulaStr)
        Significance.reportSignificance ("ETTh1 first year (zero-LLM arms)", runs, Method.StatCEM)
        assert (runs.exists (_.method == Method.StatCEM) && runs.exists (_.method == Method.Genetic), "all zero-LLM arms must run")
        assert (runs.forall (r => ! r.best.failed), "every arm must return a fitted equation")
        assert (runs.filter (_.method == Method.Baseline).head.exoUsed.size == hp("exoK").toInt, "Auto mode picks exoK loads")
        println ("lLM_NARX_SRTest7: passed")

end lLM_NARX_SRTest7


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lLM_NARX_SRTest8` main function runs the ZERO-LLM grid on the daily GEFCom2014
 *  load (peak load with the daily mean and maximum temperature) for two streams,
 *  prints the results table and the ARX_D-style output of the winners; skips if
 *  the data file is not available.  The lag window and the diagnosis skip are set
 *  to 7 (the weekly cycle of a daily series) and restored at the end.  The
 *  load-temperature relation is V-shaped: before the grid, a free hinge on
 *  yesterday's mean temperature is fitted next to the lags of y and its threshold
 *  must land in the cooling band (55..80 F; the fit gives about 67 F); the test
 *  then lists the winners that carry a hinge term.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.lLM_NARX_SRTest8
 */
@main def lLM_NARX_SRTest8 (): Unit =

    import LLM_NARX_SR._

    val cfg = Datasets.GEF14
    val loaded = try Some (cfg.load ())
                 catch case e: Exception =>
                     println (s"GEFCom2014 dataset not available -- skipping: ${e.getMessage}")
                     None
    for data <- loaded do
        given dataset: Map [String, VectorD] = data
        val (lag0, skip0) = (hp("maxLag").toInt, hp("skip").toInt)
        hp("maxLag") = 7                                                // a daily series: lags 1..7
        hp("skip")   = 7

        banner ("the V shape: a free hinge on yesterday's mean temperature next to the lags of y")
        val scorer = scorerOf ()
        val fh     = NARX_FormulaParser.parseFormula ("y ~= y~(7,1) + hinge(temp~1, ?)", scorer.vars, maxLag = 7)
        val ch     = scorer.score (fh)
        val thresh = "hinge\\(temp~1, ([0-9.]+)\\)".r.findFirstMatchIn (ch.formulaStr).map (_.group (1).toDouble).getOrElse (Double.NaN)
        println (f"${ch.formulaStr}  score ${ch.score}%.3f  validation sMAPE ${ch.smapeVal}%.3f  (${ch.reason})")
        assert (! ch.failed && thresh > 55.0 && thresh < 80.0, s"the fitted hinge threshold should be the cooling point, found $thresh F")

        val runs = runGrid ("GEFCom2014 daily load (zero-LLM arms)", cfg, zeroLLMArms, streams = Seq (0, 1))
        val top  = bestOf (runs)
        banner (f"best run: ${top.name} stream ${top.stream}, in-sample sMAPE ${top.best.smape}%.3f, validation ${top.valSmape}%.3f, TnT sMAPE ${top.testSmape}%.3f")
        println (top.best.formulaStr)
        Significance.reportSignificance ("GEFCom2014 daily load (zero-LLM arms)", runs, Method.StatCEM)
        val hinges = runs.filterNot (_.failed).filter (_.best.formulaStr.contains ("hinge"))
        println (s"runs whose winner has a hinge term: ${hinges.map (r => s"${r.name}/${r.stream}").mkString (", ")}")
        hp("maxLag") = lag0
        hp("skip")   = skip0
        assert (runs.exists (_.method == Method.StatCEM) && runs.exists (_.method == Method.Genetic), "all zero-LLM arms must run")
        assert (runs.forall (r => ! r.best.failed), "every arm must return a fitted equation")
        assert (runs.filter (_.method == Method.Baseline).head.exoUsed == Seq ("temp", "tmax"), "Given mode uses the two temperatures")
        println ("lLM_NARX_SRTest8: passed")

end lLM_NARX_SRTest8


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lLM_NARX_SRTest9` main function runs the MAIN TABLE (every arm, the LLM ones
 *  with a probed cached cloud client as in `lLM_NARX_SRTest3`) on a dataset chosen by
 *  the environment variable LLM_NARX_SR_DATASET: ili (default), covid2, etth1 (first
 *  year) or gef14 (lags 1..7); five streams, hp("maxFits") = 100; the stream-paired
 *  significance report against llm-cem (or genetic without a client); skips when
 *  the data file is absent.  Windows PowerShell: $env:LLM_NARX_SR_DATASET = "gef14".
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.lLM_NARX_SRTest9
 */
@main def lLM_NARX_SRTest9 (): Unit =

    import LLM_NARX_SR._

    val which = sys.env.getOrElse ("LLM_NARX_SR_DATASET", "ili").trim.toLowerCase
    val cfg   = which match
        case "covid2" => Datasets.COVID2
        case "etth1"  => Datasets.ETTH1_1Y
        case "gef14"  => Datasets.GEF14
        case "ili"    => Datasets.ILI
        case _        => println (s"LLM_NARX_SR_DATASET = $which not known (ili, covid2, etth1, gef14) -- using ili"); Datasets.ILI
    val loaded = try Some (cfg.load ())
                 catch case e: Exception =>
                     println (s"${cfg.name}: data file not available -- skipping: ${e.getMessage}")
                     None
    for data <- loaded do
        given dataset: Map [String, VectorD] = data
        val candidate: Option [(() => LLMClient, (Double, Double), String)] =
            if sys.env.contains ("OPENAI_API_KEY") then Some ((cloudLLM (), LLM_MODEL.prices, s"OpenAI ${LLM_MODEL.id}"))
            else localLLM ().map (f => (f, (0.0, 0.0), "local model server"))
        val llm = candidate.flatMap { (f, p, nm) =>
            probe (f) match
            case None      => println (s"LLM client: $nm answers"); Some ((f, p))
            case Some (why) =>
                println (s"LLM client: $nm does NOT answer -- $why -- running the zero-LLM arms only")
                None
        }
        if candidate.isEmpty then println ("no OPENAI_API_KEY and no local model server -- running the zero-LLM arms only")
        val (lag0, skip0) = (hp("maxLag").toInt, hp("skip").toInt)
        if cfg eq Datasets.GEF14 then
            hp("maxLag") = 7                                            // a daily series: lags 1..7
            hp("skip")   = 7
        hp("maxFits") = 100
        val streams   = Seq (0, 1, 2, 3, 4)
        val prices    = llm.map (_._2).getOrElse ((0.0, 0.0))
        val explainer = llm.map (l => new Explainer (l._1 (), cfg)).orNull
        val title     = s"${cfg.name} (main table)"
        val main = runGrid (title, cfg, mainArms (llm.map (_._1), prices), streams = streams, explainer = explainer)
        Significance.reportSignificance (title, main, if llm.isDefined then Method.LLMCEM else Method.Genetic)
        val top = bestOf (main, if llm.isDefined then Method.LLMCEM else Method.StatCEM)
        banner (f"${top.name}: best stream ${top.stream}, in-sample ${top.best.smape}%.3f, validation ${top.valSmape}%.3f, TnT ${top.testSmape}%.3f")
        println (top.best.formulaStr)
        hp("maxFits") = 0
        hp("maxLag")  = lag0
        hp("skip")    = skip0

end lLM_NARX_SRTest9
