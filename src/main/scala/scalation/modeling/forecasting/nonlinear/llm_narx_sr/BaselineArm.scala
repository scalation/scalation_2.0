//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Mon Sep 14 15:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: The Classical Baseline Arms of `LLM_NARX_SR` (V0)
 *
 *  Three deterministic anchors, the classical structures WITHOUT SEARCH, each
 *  fitted and scored by the same engine, rows and ridge penalty as every candidate
 *  of the search arms (@see `FormulaScorer`):  `Baseline` = the plain ARX
 *  structure  y ~= y~(L,1) + x1~(L,1) + ...  with L = hp("maxLag") on the
 *  shortlisted drivers -- the very seed of every search, so an arm can only be
 *  credited with what its search added;  `TunedARX` = the best of the ARX
 *  structures over p in 1..L response lags and px in 0..L exogenous lags by the
 *  in-sample criterion (the classical alternative to searching structure -- the
 *  anchor the paper must beat);  `RandomWalk` = the base's `RandomWalk (y, hh)`
 *  (no model at all).  The exogenous variables are the configuration's shortlist
 *  (@see `FormulaScorer.shortlist`: given, or picked by `ExoScreen`).  Each runs once.
 *
 *  Why the anchors are formulas and not `ARX_D (xe, y, hh)` built by the base's
 *  factory: the factory's regression centers its columns but does not standardize
 *  them, so one ridge penalty lambda acts differently on columns of different
 *  scales (deaths in thousands, a positivity rate in percent), while the search
 *  engine standardizes every term column (@see `FormulaScorer.scaled`); the same
 *  18-term structure then fits differently (Covid-19, icu and pRate: in-sample
 *  sMAPE 20.2 vs 18.4, TnT 30.0 vs 43.5).  An anchor fitted by another regression
 *  than the candidates would confound the comparison.  The factory's own model is
 *  still built in `baselineArmTest` as a reference row and as the compatibility
 *  check that the engine reproduces the base's Train-and-Test protocol exactly.
 *  @note    AI Assisted Code
 */

package scalation
package modeling
package forecasting
package nonlinear
package llm_narx_sr

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BaselineArm` class is one of the classical anchors.
 *  @param kind  which anchor: `Method.Baseline`, `Method.TunedARX` or `Method.RandomWalk`
 */
class BaselineArm (kind: Method = Method.Baseline)
      extends SearchArm:

    require (Method.anchors.contains (kind), s"BaselineArm: $kind is not an anchor")

    private val debug = debugf ("BaselineArm", true)                    // debug function

    def method: Method = kind
    override def deterministic: Boolean = true

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Score the anchor once (the stream is ignored).
     *  @param cfg     the dataset configuration
     *  @param scorer  the fit-and-score engine of the dataset
     *  @param stream  the random number stream (ignored)
     */
    def run (cfg: DatasetConfig, scorer: FormulaScorer, stream: Int): ExpRun =
        val eval0  = scorer.evaluations
        val maxLag = FormulaScorer.hp("maxLag").toInt
        val exo    = scorer.shortlist (cfg)
        debug ("run", s"${cfg.name}: exogenous variables = ${exo.mkString (", ")}")

        kind match
        case Method.Baseline =>
            val px   = if exo.isEmpty then 0 else maxLag
            val cand = arx (scorer, cfg, maxLag, px, exo)
            record (scorer, cfg, -1, cand).copy (nEval = scorer.evaluations - eval0, exoUsed = if px > 0 then exo else Seq ())
        case Method.TunedARX =>
            val cands = for p <- 1 to maxLag; px <- 0 to (if exo.isEmpty then 0 else maxLag) yield
                            (arx (scorer, cfg, p, px, exo), px)
            val (best, pxBest) = cands.minBy (_._1.score)
            record (scorer, cfg, -1, best, cands.map (_._1).sortBy (_.score).take (5))
                .copy (nEval = scorer.evaluations - eval0, exoUsed = if pxBest > 0 then exo else Seq ())
        case Method.SelARX =>
            val px   = if exo.isEmpty then 0 else maxLag
            val f    = NARX_FormulaParser.parseFormula (BaselineArm.arxFormulaStr (cfg.resp, maxLag, px, exo), scorer.vars, maxLag)
            val lib  = new TermLibrary (f.resp, f.preds, f.tFormT)
            val runs = for tech <- Seq (SelectionTech.Forward, SelectionTech.Backward) yield
                           val (c, _, nEv) = lib.selectBy (tech, scorer)
                           debug ("run", s"sel-arx $tech: $nEv sets => ${c.formulaStr} @ ${c.score}")
                           c
            val fitted = runs.filterNot (_.failed)
            val best   = if fitted.isEmpty then scorer.score (f) else fitted.minBy (_.score)
            record (scorer, cfg, -1, best, fitted.sortBy (_.score)).copy (nEval = scorer.evaluations - eval0)
        case _ =>
            val y    = scorer.fullSet (cfg.resp)
            val cand = scorer.score (new RandomWalk (y, scorer.hh), "RandomWalk: y_t+h = y_t-1")
            record (scorer, cfg, -1, cand).copy (nEval = scorer.evaluations - eval0)
        end match
    end run

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build and score the ARX structure with p response lags and px exogenous
     *  lags on the shortlisted drivers,  y ~= y~(p,1) + x1~(px,1) + ...,  through
     *  the engine (one column per lag, standardized, the scorer's ridge penalty),
     *  exactly as a proposal of a search arm is scored (@see `ARX_D.fromTerms`).
     *  @param scorer  the fit-and-score engine
     *  @param cfg     the dataset configuration
     *  @param p       the number of response lags
     *  @param px      the number of exogenous lags (0 => none)
     *  @param exo     the shortlisted exogenous variable names
     */
    private def arx (scorer: FormulaScorer, cfg: DatasetConfig, p: Int, px: Int, exo: Seq [String]): Candidate =
        val fStr = BaselineArm.arxFormulaStr (cfg.resp, p, px, exo)
        val f    = NARX_FormulaParser.parseFormula (fStr, scorer.vars, FormulaScorer.hp("maxLag").toInt)
        scorer.score (f)
    end arx

end BaselineArm


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BaselineArm` companion object builds the formula strings of the anchors.
 */
object BaselineArm:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the ARX formula string with p response lags and px exogenous lags,
     *  y ~= y~(p,1) + x1~(px,1) + ...  (no exogenous terms when px = 0); with
     *  p = px = L it is `DatasetConfig.baselineFormulaStr (L, exo)`, the seed.
     *  @param resp  the response variable name
     *  @param p     the number of response lags (p >= 1)
     *  @param px    the number of exogenous lags (0 => none)
     *  @param exo   the exogenous variable names
     */
    def arxFormulaStr (resp: String, p: Int, px: Int, exo: Seq [String]): String =
        val expo = if px > 0 then exo.map (v => s" + $v~($px,1)").mkString else ""
        s"$resp ~= $resp~($p,1)$expo"
    end arxFormulaStr

end BaselineArm


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `baselineArmTest` main function runs the three anchors on the Covid-19
 *  dataset (icu, pRate) and checks that the baseline IS the seed of the search
 *  arms (same key, same score as `DatasetConfig.baselineFormulaStr`), that the
 *  tuned ARX scores no worse than it, that the random walk is finite and that
 *  every row carries a Train-and-Test QoF.  It then builds the base's own model
 *  of the same structure, `ARX_D (xe, y, hh)` (centered, unstandardized columns,
 *  backfilled exogenous prefix), as a reference row -- NOT an arm -- and runs the
 *  compatibility check: its TnT sMAPE through the engine must be exactly what the
 *  base's own protocol gives (`train`, `setSkip ()`, `rollValidate (rc)`,
 *  `diagnoseAll` on the testing rows, which prints its fitMap and plots, as
 *  `aRX_DTest4` does).
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.baselineArmTest
 */
@main def baselineArmTest (): Unit =

    val cfg = Datasets.COVID2
    given dataset: Map [String, VectorD] = cfg.load ()
    FormulaScorer.hp("lambda") = 1.0
    val scorer = new FormulaScorer (hh = 6)
    val maxLag = FormulaScorer.hp("maxLag").toInt

    val runs = for kind <- Seq (Method.Baseline, Method.TunedARX, Method.SelARX, Method.RandomWalk) yield
        val r = new BaselineArm (kind).run (cfg, scorer, 0)
        banner (s"${r.name}")
        println (s"model: ${r.best.formulaStr}")
        println (f"score ${r.best.score}%.3f  in-sample sMAPE ${r.best.smape}%.3f  validation sMAPE ${r.valSmape}%.3f")
        println (f"TnT sMAPE ${r.testSmape}%.3f  exo ${r.exoUsed.mkString (", ")}  fits ${r.nEval}")
        r.showQoF ()
        if r.top.nonEmpty then println (r.equations)
        r
    val (base, tuned, sel, rw) = (runs(0), runs(1), runs(2), runs(3))
    assert (! base.best.failed && ! tuned.best.failed && ! sel.best.failed, "anchors must fit")
    assert (tuned.best.score <= base.best.score + 1e-12, "the tuned ARX must score no worse than the plain baseline")
    assert (sel.best.score <= base.best.score + 1e-6, "the selected ARX must score no worse than the plain baseline")
    assert (sel.best.nTerms <= base.best.nTerms, "selection keeps a subset of the baseline's lags")
    assert (rw.best.smapeH.forall (v => ! v.isNaN && v > 0.0), "the random walk must be finite")
    assert (runs.forall (r => r.smapeTest.forall (v => ! v.isNaN && v > 0.0)), "every anchor carries a TnT QoF")
    assert (base.exoUsed == Seq ("icu", "pRate"), "the baseline uses the given exogenous variables")

    banner ("the baseline is the seed of every search arm")
    val seed = scorer.score (NARX_FormulaParser.parseFormula (cfg.baselineFormulaStr (maxLag), scorer.vars, maxLag))
    println (s"seed:     ${seed.formulaStr} @ ${seed.score}")
    println (s"baseline: ${base.best.formulaStr} @ ${base.best.score}")
    assert (seed.key == base.best.key && seed.score == base.best.score, "the baseline must be the search seed")

    banner ("reference (not an arm): the base's own ARX_D (xe, y, hh) of the same structure")
    val y   = dataset ("y")
    val xe  = MatrixD (dataset ("icu"), dataset ("pRate")).transpose
    val mod = ARX_D (xe, y, 6, hparam = scorer.hparam (("p", 6), ("px", 6), ("spec", 0)))
    val ref = scorer.score (mod, s"${mod.modelName} [${cfg.baselineFormulaStr (maxLag)}] centered, unstandardized")
    val qof = scorer.tnt (mod)
    println (f"base's regression: score ${ref.score}%.3f  in-sample sMAPE ${ref.smape}%.3f  TnT sMAPE ${qof(?, QoF.smape.ordinal).mean}%.3f")
    println (f"engine (anchor):   score ${base.best.score}%.3f  in-sample sMAPE ${base.best.smape}%.3f  TnT sMAPE ${base.testSmape}%.3f")
    println ("same structure and parameters; the ridge penalty acts on centered vs. standardized columns")

    banner ("compatibility: the base's own TnT protocol on the base's own model")
    val mod2 = ARX_D (xe, y, 6, hparam = scorer.hparam (("p", 6), ("px", 6), ("spec", 0)))
    mod2.train (mod2.getX, mod2.getYy)                                  // initial training (replaced by the first refit)
    mod2.forecastAll ()
    mod2.setSkip ()                                                     // diagnose: don't skip any rows
    mod2.rollValidate (scorer.rc)                                       // TnT with rolling validation
    val ft = mod2.diagnoseAll (mod2.getY, mod2.getYf, Forecaster.teRng (y.dim))
    val ref2 = ft(?, QoF.smape.ordinal)
    println (s"base protocol TnT sMAPE = $ref2")
    println (s"engine        TnT sMAPE = ${qof(?, QoF.smape.ordinal)}")
    assert ((ref2 - qof(?, QoF.smape.ordinal)).normSq < 1e-12, "the engine must reproduce the base's TnT numbers exactly")
    println ("baselineArmTest: passed")

end baselineArmTest
