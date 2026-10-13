
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Mon Sep 14 15:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Plausibility Checks of `LLM_NARX_SR` Equations
 *
 *  Reads `ExpRun` records and their equations (refitted through the scorer's
 *  `ARX_D` path) and reports what a domain reader asks of an equation:
 *  (1) sign consistency -- the NET effect of each exogenous variable (the sum of
 *      the standardized h = 1 coefficients of its terms) has the sign domain
 *      knowledge expects (@see `DatasetConfig.expectedSigns`);
 *  (2) residual cleanliness -- the residual tests of `EDA4TS.residualReport`
 *      (Ljung-Box whiteness, ACF spikes, collinearity, heteroscedasticity);
 *  (3) structural stability -- how similar the best equations of one arm are
 *      across streams (mean pairwise Jaccard similarity of their term sets).
 *  Nothing here influences a search; it is the post-hoc table of the paper.
 *  @note    AI Assisted Code
 */

package scalation
package modeling
package forecasting
package nonlinear
package llm_narx_sr

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PlausibilityReport` case class holds the checks of one equation.
 *  @param formulaStr       the equation (formula string)
 *  @param signConsistency  the share of expected-sign variables whose net effect agrees (NaN if none)
 *  @param nSignChecked     the number of variables checked
 *  @param netEffects       the standardized net effect per variable (h = 1)
 *  @param cleanliness      the share of residual tests passed
 *  @param notes            the residual findings in words
 */
case class PlausibilityReport (formulaStr: String, signConsistency: Double, nSignChecked: Int,
                               netEffects: Map [String, Double], cleanliness: Double,
                               notes: Seq [String]):

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return one formatted line for a per-equation table.
     */
    def tableRow: String =
        val eff = netEffects.toSeq.sortBy (_._1).map ((v, e) => f"$v ${e}%+.2f").mkString (" ")
        f"${formulaStr.replace ('\n', ' ').take (56)}%-56s  sign=${signConsistency}%5.2f (n=$nSignChecked)  " +
        f"clean=${cleanliness}%5.2f  net: $eff"
    end tableRow

end PlausibilityReport


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PlausibilityChecker` class checks equations against domain expectations
 *  and residual tests.
 *  @param expectedSigns  the expected sign (+1/-1) of each variable's net effect
 */
class PlausibilityChecker (expectedSigns: Map [String, Int] = Map ()):

    private val debug = debugf ("PlausibilityChecker", false)           // debug function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the standardized net effect of each variable at h = 1: the sum over
     *  the single-variable terms of the variable of slope x standard deviation
     *  of the term's column (products and waves are not attributed).
     *  @param mod  the fitted model of the formula
     *  @param f    the formula (its terms match the model's columns in order)
     */
    def netEffects (mod: ARX_D, f: ARX_Formula): Map [String, Double] =
        val b   = mod.slopes(?, 0)                                      // h = 1 slopes in original units
        val x   = mod.getX
        val eff = scala.collection.mutable.LinkedHashMap [String, Double] ()
        for (t, j) <- f.preds.zipWithIndex if t.vars.size == 1 do
            val v = t.vars.head
            eff (v) = eff.getOrElse (v, 0.0) + b(j) * x(?, j).stdev
        end for
        eff.toMap
    end netEffects

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the sign-consistency score and the number of variables checked.
     *  @param effects  the net effect per variable
     */
    def signConsistency (effects: Map [String, Double]): (Double, Int) =
        val checked = expectedSigns.toSeq.filter ((v, _) => effects.contains (v))
        if checked.isEmpty then (Double.NaN, 0)
        else
            val agree = checked.count ((v, s) => math.signum (effects (v)) == s.toDouble)
            (agree.toDouble / checked.size, checked.size)
    end signConsistency

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check one equation: refit it through the scorer (training rows), read its
     *  net effects and run the residual tests on its training residuals.
     *  @param cand    the scored equation
     *  @param scorer  the fit-and-score engine that fitted it
     *  @param eda     the EDA profile (for the residual tests)
     */
    def evaluate (cand: Candidate, scorer: FormulaScorer, eda: EDA4TS): PlausibilityReport =
        val f = cand.formula
        if f == null then                                               // a base model (anchor): no formula to read
            return PlausibilityReport (cand.formulaStr, Double.NaN, 0, Map (), Double.NaN, Seq ("base model: not checked"))
        try
            val mod  = scorer.fit (f)
            val eff  = netEffects (mod, f)
            val (sc, n) = signConsistency (eff)
            val skip = math.max (f.maxLag, scorer.skip)
            val tr   = scorer.trSize
            val e    = (mod.getY - mod.getYf(?, 1))(skip until tr)
            val (notes, clean) = eda.residualReport (e, mod.getX (skip until tr), mod.getFname, mod.getY (skip until tr))
            PlausibilityReport (f.formulaStr, sc, n, eff, clean, notes)
        catch case ex: Exception =>
            debug ("evaluate", s"could not check '${f.formulaStr}': ${ex.getMessage}")
            PlausibilityReport (f.formulaStr, Double.NaN, 0, Map (), Double.NaN, Seq (s"not checked: ${ex.getMessage}"))
    end evaluate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the structural stability of a set of equations: the mean pairwise
     *  Jaccard similarity of their term-label sets (NaN for fewer than two).
     *  @param cands  the equations (e.g., one arm's best per stream)
     */
    def stability (cands: Seq [Candidate]): Double =
        val sets = cands.filter (_.formula != null)
                        .map (c => c.formula.preds.map (_.label).toSet ++ (if c.formula.tFormT != null then Set ("TFORM") else Set ()))
        if sets.size < 2 then Double.NaN
        else
            val pairs = for i <- sets.indices; j <- sets.indices if j > i yield
                val union = sets(i) union sets(j)
                if union.isEmpty then 1.0 else (sets(i) intersect sets(j)).size.toDouble / union.size
            pairs.sum / pairs.size
    end stability

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print the plausibility table of a grid: per arm, the stability of its best
     *  equations over streams and the mean sign consistency and cleanliness; then
     *  one row per run.
     *  @param title   the experiment title
     *  @param runs    the collected runs
     *  @param scorer  the fit-and-score engine that produced them
     *  @param eda     the EDA profile of the dataset's training rows
     */
    def report (title: String, runs: Seq [ExpRun], scorer: FormulaScorer, eda: EDA4TS): Unit =
        banner (s"$title -- plausibility (stability over streams, sign consistency, residual cleanliness)")
        val reports = runs.map (r => r -> evaluate (r.best, scorer, eda))
        println (f"${"method"}%-14s ${"n"}%3s ${"stability"}%9s ${"sign"}%6s ${"clean"}%6s")
        for m <- runs.map (_.name).distinct do
            val rs   = reports.filter (_._1.name == m)
            val sign = rs.map (_._2.signConsistency).filterNot (_.isNaN)
            val cln  = rs.map (_._2.cleanliness).filterNot (_.isNaN)
            val mS   = if sign.isEmpty then Double.NaN else sign.sum / sign.size
            val mC   = if cln.isEmpty then Double.NaN else cln.sum / cln.size
            println (f"$m%-14s ${rs.size}%3d ${stability (rs.map (_._1.best))}%9.2f $mS%6.2f $mC%6.2f")
        end for
        banner (s"$title -- per run")
        for (r, p) <- reports do
            println (f"${r.name}%-14s ${r.stream}%3d  ${p.tableRow}")
            for n <- p.notes do println (s"                    - $n")
        end for
    end report

end PlausibilityChecker


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `plausibilityCheckerTest` main function tests the checks on a synthetic
 *  series where z drives y positively at lag 2 and w drives it negatively at
 *  lag 1: the net effects must carry those signs (also through a log1p term and
 *  a two-lag window), the sign consistency must be 1 against the matching
 *  expectation and 0.5 against a half-wrong one, and stability must be 1 for
 *  identical equations and below 1 for different ones.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.plausibilityCheckerTest
 */
@main def plausibilityCheckerTest (): Unit =

    import scalation.random.Normal

    val n     = 300
    val noise = Normal (0.0, 1.0, 31)
    val zz    = Normal (0.0, 4.0, 32)
    val ww    = Normal (0.0, 4.0, 33)
    val z     = VectorD (for _ <- 0 until n yield 20.0 + zz.gen)
    val w     = VectorD (for _ <- 0 until n yield 20.0 + ww.gen)
    val y     = new VectorD (n)
    y(0) = 40.0; y(1) = 41.0
    for t <- 2 until n do y(t) = 10.0 + 0.6 * y(t-1) + 1.5 * z(t-2) - 1.0 * w(t-1) + noise.gen

    given dataset: Map [String, VectorD] = Map ("y" -> y, "z" -> z, "w" -> w)
    val scorer = new FormulaScorer (hh = 2)
    val eda    = new EDA4TS (y, Map ("z" -> z, "w" -> w), scorer.trSize)
    val vars   = dataset.keySet

    val checker = new PlausibilityChecker (Map ("z" -> 1, "w" -> -1))
    val f1 = NARX_FormulaParser.parseFormula ("y ~= y~1 + z~(2,1) + log1p(w~1)", vars, 3)
    val c1 = scorer.score (f1)
    val p1 = checker.evaluate (c1, scorer, eda)
    banner ("right model")
    println (p1.tableRow)
    for nt <- p1.notes do println (s"  - $nt")
    assert (p1.netEffects ("z") > 0.0 && p1.netEffects ("w") < 0.0, s"net effects must carry the planted signs: ${p1.netEffects}")
    assert (p1.signConsistency == 1.0 && p1.nSignChecked == 2, "both expected signs agree")

    val wrong = new PlausibilityChecker (Map ("z" -> 1, "w" -> 1))
    val p2    = wrong.evaluate (c1, scorer, eda)
    assert (math.abs (p2.signConsistency - 0.5) < 1e-12, "one of two expectations agrees")

    banner ("stability")
    val f2 = NARX_FormulaParser.parseFormula ("y ~= y~1 + z~2", vars, 3)
    val c2 = scorer.score (f2)
    assert (checker.stability (Seq (c1, c1)) == 1.0, "identical equations: stability 1")
    val st = checker.stability (Seq (c1, c2))
    println (f"stability of two different equations = $st%.3f")
    assert (st > 0.0 && st < 1.0, "different equations with shared terms: 0 < stability < 1")
    println ("plausibilityCheckerTest: passed")

end plausibilityCheckerTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `plausibilityCheckerTest2` main function prints the plausibility table of
 *  a real grid: the zero-LLM arms on the Covid-19 dataset (icu, pRate), whose
 *  expected signs are positive for both covariates.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.plausibilityCheckerTest2
 */
@main def plausibilityCheckerTest2 (): Unit =

    import LLM_NARX_SR._

    val cfg = Datasets.COVID2
    given dataset: Map [String, VectorD] = cfg.load ()

    val runs    = runGrid ("Covid-19 icu/pRate (zero-LLM arms)", cfg, zeroLLMArms, streams = Seq (0, 1), show = false)
    val scorer  = scorerOf ()
    val eda     = new EDA4TS (dataset ("y"), cfg.exo.map (v => v -> dataset (v)).toMap, scorer.trSize,
                              hp("maxLag").toInt, hp("nFreqs").toInt)
    val checker = new PlausibilityChecker (cfg.expectedSigns)
    checker.report ("Covid-19 icu/pRate", runs, scorer, eda)

end plausibilityCheckerTest2
