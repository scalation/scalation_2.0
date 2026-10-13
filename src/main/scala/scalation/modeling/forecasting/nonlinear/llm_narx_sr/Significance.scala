
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Mon Sep 14 15:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Significance Analysis of `LLM_NARX_SR` Runs
 *
 *  Reads `ExpRun` records (@see `SearchArm`) and nothing else from the harness.
 *  The statistics themselves live in `mathstat`: the paired Wilcoxon signed-rank
 *  test (`wilcoxon_test`) and Spearman's rank correlation (`VectorD.scorr`).
 *  Here: the stream-paired comparison of two arms, the one-sample comparison of
 *  an arm against a fixed value (a deterministic arm), the model-vs-every-arm
 *  report of a grid, the per-fold table of `runFolds` (signed-rank over
 *  streams x folds against the fold's baseline) and the selection check (does
 *  the in-sample criterion on the training rows rank structures as the TnT
 *  rolling validation on the testing rows does?).
 *  @note    AI Assisted Code
 */

package scalation
package modeling
package forecasting
package nonlinear
package llm_narx_sr

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SigResult` case class is the outcome of one paired comparison.
 *  @param nPairs  the number of pairs (streams) compared
 *  @param meanA   the mean of method A's paired metric
 *  @param sdA     the standard deviation of method A's paired metric
 *  @param meanB   the mean of method B's paired metric
 *  @param sdB     the standard deviation of method B's paired metric
 *  @param wStat   the Wilcoxon W statistic (min of W+, W-)
 *  @param zStat   the normal-approximation z-score (with continuity correction)
 *  @param pValue  the two-sided p-value
 */
case class SigResult (nPairs: Int, meanA: Double, sdA: Double, meanB: Double, sdB: Double,
                      wStat: Double, zStat: Double, pValue: Double):

    override def toString: String =
        f"n=$nPairs  A: mean=$meanA%8.4f sd=$sdA%8.4f  B: mean=$meanB%8.4f sd=$sdB%8.4f  " +
        f"W=$wStat%6.1f  z=$zStat%6.3f  p=$pValue%7.4f" +
        (if pValue < 0.05 then "  *significant at 0.05*" else "")

end SigResult


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Significance` object compares arms from their run records.
 */
object Significance:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Run the stream-paired Wilcoxon signed-rank test between two methods'
     *  per-stream metric (by default `ExpRun.testSmape`: the mean TnT sMAPE over
     *  horizons on the testing rows); only streams present in both are used.
     *  @param runsA   the first method's runs (one per stream)
     *  @param runsB   the second method's runs (one per stream)
     *  @param metric  the number compared per run
     */
    def compareMethods (runsA: Seq [ExpRun], runsB: Seq [ExpRun],
                        metric: ExpRun => Double = _.testSmape): SigResult =
        val byA     = runsA.map (r => r.stream -> metric (r)).toMap
        val byB     = runsB.map (r => r.stream -> metric (r)).toMap
        val streams = byA.keySet.intersect (byB.keySet).toIndexedSeq.sorted
        require (streams.nonEmpty, "compareMethods: no common streams between the two run sets")
        val a = VectorD (streams.map (byA))
        val b = VectorD (streams.map (byB))
        val (w, z, p) = wilcoxon_test (a, b)
        SigResult (streams.size, a.mean, a.stdev, b.mean, b.stdev, w, z, p)
    end compareMethods

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Run the one-sample signed-rank test of a method's per-stream metric
     *  against a fixed reference value (e.g., a deterministic arm's single run).
     *  @param runsA      the method's runs (one per stream)
     *  @param reference  the fixed reference value
     *  @param metric     the number compared per run
     */
    def compareToConstant (runsA: Seq [ExpRun], reference: Double,
                           metric: ExpRun => Double = _.testSmape): SigResult =
        require (runsA.nonEmpty, "compareToConstant: runsA is empty")
        val a = VectorD (runsA.toIndexedSeq.map (metric))
        val b = VectorD.fill (a.dim)(reference)
        val (w, z, p) = wilcoxon_test (a, b)
        SigResult (a.dim, a.mean, a.stdev, reference, 0.0, w, z, p)
    end compareToConstant

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print the comparison of the model arm against every other arm in `runs`:
     *  stream-paired when the other arm has several runs, one-sample against its
     *  single value when it has one (a deterministic arm).
     *  @param title   the experiment title (dataset + configuration)
     *  @param runs    the collected runs across all methods/streams
     *  @param model   the arm the claims are built on (its standard version, variant "")
     *  @param metric  the number compared per run
     */
    def reportSignificance (title: String, runs: Seq [ExpRun], model: Method = Method.LLMCEM,
                            metric: ExpRun => Double = _.testSmape): Unit =
        val ok    = runs.filterNot (_.failed)                           // failed runs carry no metric
        val mine  = ok.filter (r => r.method == model && r.variant.isEmpty)
        val label = model.tag
        banner (s"$title -- significance of $label vs each other arm (TnT sMAPE)")
        if mine.isEmpty then println (s"$label: no fitted runs in this set")
        else
            for m <- ok.map (_.name).distinct if m != label do
                val other = ok.filter (_.name == m)
                val line  =
                    if other.size == 1 then
                        val a = VectorD (mine.map (metric).toIndexedSeq)
                        f"(deterministic): n=${a.dim}  A: mean=${a.mean}%8.4f sd=${a.stdev}%8.4f  B: ${metric (other.head)}%8.4f  " +
                        f"diff=${a.mean - metric (other.head)}%+8.4f  -- no test (one value; see the fold table)"
                    else try s"(stream-paired): ${compareMethods (mine, other, metric)}"
                         catch case e: IllegalArgumentException => s"skipped (${e.getMessage})"
                println (f"$label vs $m%-14s $line")
            end for
    end reportSignificance

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print, per arm, the TnT sMAPE of each fold (mean over streams), its mean
     *  over folds, the mean difference to the fold's baseline and the two-sided
     *  signed-rank p-value of those differences over streams x folds (n pairs).
     *  @param title  the experiment title
     *  @param cuts   the series length per fold
     *  @param folds  the runs per fold (from `LLM_NARX_SR.runFolds`)
     */
    def reportFolds (title: String, cuts: Seq [Int], folds: Seq [Seq [ExpRun]],
                     refs: Seq [Method] = Seq (Method.Baseline, Method.SelARX)): Unit =
        banner (s"$title -- TnT sMAPE per fold (mean over streams), over folds, and vs the fold's ${refs.map (_.tag).mkString (" / ")}")
        val methods = folds.flatten.map (_.name).distinct
        val hdr     = cuts.map (c => f"${s"..$c"}%9s").mkString (" ")
        val rHdr    = refs.map (r => f"${s"vs ${r.tag}"}%12s ${"n"}%4s ${"p"}%8s").mkString (" ")
        println (f"${"method"}%-14s $hdr ${"mean"}%9s $rHdr")
        for m <- methods do
            val perFold = folds.map (rs =>
                val xs = rs.filter (r => r.name == m && ! r.failed).map (_.testSmape)
                if xs.isEmpty then Double.NaN else xs.sum / xs.size)
            val means   = perFold.filterNot (_.isNaN)
            val mean    = if means.isEmpty then Double.NaN else means.sum / means.size
            val cells   = perFold.map (v => f"$v%9.3f").mkString (" ")
            val vs = for ref <- refs yield
                val diffs = folds.flatMap (rs =>
                                val b = rs.find (r => r.method == ref && ! r.failed).map (_.testSmape)
                                if b.isEmpty then Seq ()
                                else rs.filter (r => r.name == m && ! r.failed).map (_.testSmape - b.get))
                val mDiff = if diffs.isEmpty then Double.NaN else diffs.sum / diffs.size
                val p     = if m == ref.tag || diffs.size < 2 then Double.NaN
                            else wilcoxon_test (VectorD (diffs.toIndexedSeq), new VectorD (diffs.size))._3
                f"$mDiff%+12.3f ${diffs.size}%4d $p%8.4f" + (if ! p.isNaN && p < 0.05 then "*" else " ")
            println (f"$m%-14s $cells $mean%9.3f ${vs.mkString (" ")}")
        end for
    end reportFolds

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return Spearman's rank correlation between the in-sample selection score
     *  (training rows) and the TnT sMAPE (testing rows) over the fitted runs (NaN
     *  if fewer than three): +1 means the in-sample criterion ranks structures as
     *  the rolling validation does, <= 0 that selecting in-sample does not carry over.
     *  @param runs  the collected runs
     */
    def selectionCheck (runs: Seq [ExpRun]): Double =
        val scored = runs.filter (r => ! r.best.failed && ! r.testSmape.isNaN)
        if scored.size < 3 then Double.NaN
        else VectorD (scored.map (_.best.score).toIndexedSeq) scorr VectorD (scored.map (_.testSmape).toIndexedSeq)
    end selectionCheck

end Significance


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `significanceTest` main function tests the comparisons on hand-made run
 *  records: a stream-paired comparison of a uniformly better method is
 *  significant, one against itself is not, the one-sample comparison against a
 *  far-off constant is significant, and the selection check is +1 on runs whose
 *  in-sample and TnT scores agree in rank.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.significanceTest
 */
@main def significanceTest (): Unit =

    val f = NARX_FormulaParser.parseFormula ("y ~= y~1", Set ("y"), 1)
    def run (method: Method, stream: Int, smape: Double, tnt: Double): ExpRun =
        ExpRun (method, "", stream, Candidate (f, smape, MatrixD.fill (1, Fit.N_QoF, smape), 1),
                MatrixD.fill (1, Fit.N_QoF, tnt))

    val a = (0 until 8).map (s => run (Method.LLMCEM, s, 8.0 + 0.1 * s, 9.0 + 0.1 * s))
    val b = (0 until 8).map (s => run (Method.StatCEM, s, 12.0 + 0.1 * s, 13.0 + 0.1 * s))

    banner ("A better than B on every stream")
    val ab = Significance.compareMethods (a, b)
    println (ab)
    assert (ab.nPairs == 8 && ab.pValue < 0.05, "a uniform 4-point gap over 8 streams must be significant")

    banner ("A against itself")
    val aa = Significance.compareMethods (a, a)
    println (aa)
    assert (aa.pValue > 0.5, "no difference must not be significant")

    banner ("one-sample against a far-off constant")
    val ac = Significance.compareToConstant (a, 20.0)
    println (ac)
    assert (ac.pValue < 0.05, "8 values all below 20 must be significant")

    banner ("selection check")
    val rho = Significance.selectionCheck (a ++ b)
    println (f"Spearman (in-sample vs TnT) = $rho%.3f")
    assert (math.abs (rho - 1.0) < 1e-12, "ranks agree exactly: rho = 1")

    Significance.reportSignificance ("hand-made", a ++ b ++ Seq (run (Method.Baseline, -1, 10.0, 11.0)), Method.LLMCEM)
    println ("significanceTest: passed")

end significanceTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `significanceTest2` main function analyses a real grid: the zero-LLM
 *  arms on the Covid-19 dataset (icu, pRate) over three streams, then the
 *  model-vs-every-arm report and the selection check on the returned runs; then
 *  (rev3.2) the same arms as a rolling final evaluation over three nested folds
 *  (the series cut 32 and 16 weeks before its end, then whole: testing windows of
 *  17, 20 and 24 weeks), with the fold-paired signed-rank test of every arm against
 *  the fold's baseline -- the test that can reach significance where three streams
 *  against one deterministic value cannot.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.significanceTest2
 */
@main def significanceTest2 (): Unit =

    import LLM_NARX_SR._

    val cfg = Datasets.COVID2
    given dataset: Map [String, VectorD] = cfg.load ()

    val runs = runGrid ("Covid-19 icu/pRate (zero-LLM arms)", cfg, zeroLLMArms, streams = Seq (0, 1, 2), show = false)
    Significance.reportSignificance ("Covid-19 icu/pRate", runs, Method.Genetic)
    Significance.reportSignificance ("Covid-19 icu/pRate", runs, Method.StatCEM)
    val rho = Significance.selectionCheck (runs)
    println (f"selection check: Spearman (selection score vs TnT sMAPE) over ${runs.size} runs = $rho%.3f")

    val m     = dataset ("y").dim
    val cuts  = Seq (m - 32, m - 16, m)
    val folds = runFolds ("Covid-19 icu/pRate (zero-LLM arms)", cfg, zeroLLMArms, cuts, streams = Seq (0, 1))
    Significance.reportFolds ("Covid-19 icu/pRate (zero-LLM arms)", cuts, folds)
    val rhoF = Significance.selectionCheck (folds.flatten)
    println (f"selection check over the folds: Spearman (selection score vs TnT sMAPE) over ${folds.flatten.size} runs = $rhoF%.3f")

end significanceTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `significanceTest3` main function analyses a rolling final evaluation:
 *  the zero-LLM arms on the ILI dataset over three nested folds (the last 52
 *  weeks of each fold held back), then the per-fold table with the signed-rank
 *  test of every arm against the fold's baseline; skips if the ILI data file
 *  is not available.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.significanceTest3
 */
@main def significanceTest3 (): Unit =

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
        Significance.reportFolds ("ILI (zero-LLM arms)", cuts, folds)

end significanceTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `significanceTest4` main function runs the zero-LLM arms on the daily
 *  GEFCom2014 load as a three-fold evaluation on nested prefixes (the series up
 *  to two years before its end, one year before, and whole; each prefix split by
 *  `Model.trSize`), with lags 1..7, and prints the fold table with the fold-paired
 *  signed-rank tests against the baseline and sel-arx; skips if the data file is
 *  not available.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.significanceTest4
 */
@main def significanceTest4 (): Unit =

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
        val m     = data ("y").dim
        val cuts  = Seq (m - 730, m - 365, m)
        val folds = runFolds ("GEFCom2014 daily load (zero-LLM arms)", cfg, zeroLLMArms, cuts, streams = Seq (0, 1))
        Significance.reportFolds ("GEFCom2014 daily load (zero-LLM arms)", cuts, folds)
        val rho = Significance.selectionCheck (folds.flatten)
        println (f"selection check over the folds: Spearman (selection score vs TnT sMAPE) over ${folds.flatten.size} runs = $rho%.3f")
        hp("maxLag") = lag0
        hp("skip")   = skip0

end significanceTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `significanceTest5` main function runs the zero-LLM arms on the first year
 *  of ETTh1 as a three-fold evaluation on nested prefixes (the first eight, ten
 *  and twelve months; each prefix split by `Model.trSize`) and prints the fold
 *  table with the fold-paired signed-rank tests against the baseline and sel-arx;
 *  skips if the data file is not available.  Long: three grids of 5840..8760 hourly rows.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.significanceTest5
 */
@main def significanceTest5 (): Unit =

    import LLM_NARX_SR._

    val cfg = Datasets.ETTH1_1Y
    val loaded = try Some (cfg.load ())
                 catch case e: Exception =>
                     println (s"ETTh1 dataset not available -- skipping: ${e.getMessage}")
                     None
    for data <- loaded do
        given dataset: Map [String, VectorD] = data
        val m     = data ("y").dim
        val cuts  = Seq (m - 2920, m - 1460, m)
        val folds = runFolds ("ETTh1 first year (zero-LLM arms)", cfg, zeroLLMArms, cuts, streams = Seq (0, 1))
        Significance.reportFolds ("ETTh1 first year (zero-LLM arms)", cuts, folds)
        val rho = Significance.selectionCheck (folds.flatten)
        println (f"selection check over the folds: Spearman (selection score vs TnT sMAPE) over ${folds.flatten.size} runs = $rho%.3f")

end significanceTest5
