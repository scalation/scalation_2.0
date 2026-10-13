
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Fri Sep 11 10:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Framework: Cross-Entropy Method (CEM) for Feature Selection
 *
 *  Feature selection as a search over inclusion PROBABILITIES rather than over
 *  feature sets: column j carries a probability p_j of belonging to a good model.
 *  Each round samples `nSample` distinct column sets (column j included with
 *  probability p_j), evaluates each with the caller's `eval` function (build,
 *  train, test), keeps the elite fraction and moves p towards the elite's
 *  inclusion frequencies,
 *
 *      p_j <- (1 - alpha) p_j + alpha freq_j,      p_j kept in [eps, 1 - eps],
 *
 *  so no column is ever ruled in or out for good.  Settings follow the CE tutorial
 *  (de Boer et al. 2005, Remarks 2.3 and 2.6): sample size N = c n with c = 5,
 *  elite fraction rho = ln(n)/n (kept in [0.05, 0.3]), smoothing alpha = 0.7, and
 *  the stopping rule on the elite THRESHOLD gamma (the worst elite score): stop
 *  when gamma has not improved for `patience` = 5 rounds, when every p_j has
 *  reached its floor or ceiling (degenerate distribution), after `maxIter`
 *  rounds, or when the sampler cannot find an unseen set.  Two additions from
 *  iCEM (Pinneri et al. 2021): a fraction `keep` of the previous round's elites
 *  joins the next round's ranking (elite memory, no refit needed), and the
 *  incumbent best set always counts as an elite (elitism).  No set is fitted
 *  twice.  The result is the best set evaluated, never the mean of the
 *  distribution.
 *
 *  Compared with the greedy Forward/Backward/Stepwise selectors it is not greedy
 *  (a column dropped in one round can return), it keeps diversity by sampling
 *  instead of enumerating neighbours, and it takes a PRIOR: p starts from `prior`
 *  when one is given (e.g., inclusion probabilities suggested by an earlier
 *  search stage), so the final p measures how far the evidence moved that prior.
 *  The final p is a result in itself: columns near one are the stable core.
 *  @see de Boer, Kroese, Mannor and Rubinstein, "A Tutorial on the Cross-Entropy
 *       Method", Annals of Operations Research 134 (2005) 19-67.
 *  @see Pinneri et al., "Sample-efficient Cross-Entropy Method for Real-time
 *       Planning", CoRL 2020 (PMLR 155, 2021), for elite memory.
 *
 *  @note    AI Assisted Code
 */

package scalation
package modeling

import scala.collection.mutable.{ArrayBuffer => VEC, HashSet, LinkedHashSet => LSET}
import scala.math.{ceil, log, max, min}
import scala.util.control.Breaks.{break, breakable}

import scalation.mathstat._
import scalation.random.Random

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CEM_Selector` class selects a subset of the columns 0 until n by the
 *  Cross-Entropy Method, using the caller's evaluator to build and score the
 *  model of a column set (@see `Forecaster_Reg.cemSelAll` for the wiring into
 *  feature selection).
 *  @param n         the number of columns (candidate features) 0 until n
 *  @param eval      the evaluator: build, train and test the model on a column set,
 *                   returning its `BestStep` (col is not used; qof must hold the metric qk)
 *  @param fixed     the columns always included (e.g., the intercept column 0)
 *  @param prior     the prior inclusion probability per column (null => 0.5 each;
 *                   entries for fixed columns are ignored)
 *  @param nSample   the number of distinct column sets sampled per round
 *                   (-1 => C_N times the number of free columns, at least MIN_SAMPLE)
 *  @param elite     the fraction of a round's sets kept as the elite (-1 => ln(n)/n in [0.05, 0.3])
 *  @param alpha     the smoothing rate of the probability update
 *  @param eps       the floor/ceiling keeping probabilities inside (0, 1)
 *  @param maxIter   the maximum number of rounds
 *  @param patience  the rounds without improving the elite threshold after which the run stops
 *  @param maxCols   the maximum number of columns in a sampled set (-1 => n)
 *  @param keep      the fraction of the previous round's elites carried into the next ranking
 *  @param stream    the random number stream
 *  @param qk        index of Quality of Fit (QoF) to use for comparing quality
 */
class CEM_Selector (n: Int, eval: LSET [Int] => BestStep, fixed: LSET [Int] = LSET (),
                    prior: VectorD = null,
                    nSample: Int = CEM_Selector.N_SAMPLE, elite: Double = CEM_Selector.ELITE,
                    alpha: Double = CEM_Selector.ALPHA, eps: Double = CEM_Selector.EPS,
                    maxIter: Int = CEM_Selector.MAX_ITER, patience: Int = CEM_Selector.PATIENCE,
                    maxCols: Int = -1, keep: Double = CEM_Selector.KEEP, stream: Int = 0)
                   (using qk: Int):

    private val debug = debugf ("CEM_Selector", false)                    // debug function
    private val flaw  = flawf ("CEM_Selector")                            // flaw function

    private val rng   = Random (stream)                                   // uniform (0, 1) random number generator
    private val free  = (0 until n).filterNot (fixed.contains)            // the columns the sampler decides on
    private val cap   = if maxCols < 0 then n else maxCols                // the largest set allowed
    private val maxi  = Fit.maxi.contains (qk)                            // whether the metric is maximized
    private val seen  = HashSet [Set [Int]] ()                            // the sets evaluated so far
    private val trace = VEC [BestStep] ()                                 // the best-so-far after each round
    private var bestStep = BestStep ()()                                  // the best set found so far

    /** The sample size per round and the elite fraction actually used (defaults
     *  resolved from the number of free columns, @see the class documentation).
     */
    val nS: Int     = if nSample > 0 then nSample
                      else max (CEM_Selector.MIN_SAMPLE, (CEM_Selector.C_N * free.size).toInt)
    val rho: Double = if elite > 0.0 then elite
                      else min (0.3, max (0.05, log (max (2, free.size)) / max (2, free.size)))

    require (n > 0 && fixed.forall (j => 0 <= j && j < n), s"CEM_Selector: fixed columns must be in 0 until $n")
    require (fixed.size <= cap, s"CEM_Selector: ${fixed.size} fixed columns exceed maxCols = $cap")
    require (prior == null || prior.dim == n, s"CEM_Selector: prior.dim ${prior.dim} != n = $n")
    require (rho <= 1.0 && 0.0 < alpha && alpha <= 1.0 && 0.0 < eps && eps < 0.5 && 0.0 <= keep && keep <= 1.0,
             "CEM_Selector: need 0 < elite <= 1, 0 < alpha <= 1, 0 < eps < 0.5, 0 <= keep <= 1")

    /** The inclusion probabilities, one per column (fixed columns stay at 1).
     */
    private val p = VectorD (for j <- 0 until n yield
                                 if fixed.contains (j) then 1.0
                                 else if prior == null then 0.5
                                 else clamp (prior(j)))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a copy of the current inclusion probabilities.
     */
    def probs: VectorD = p.copy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the best step found so far (qof is null before `run`).
     */
    def best: BestStep = bestStep

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the best-so-far step after each completed round (the trace).
     */
    def bestByRound: Seq [BestStep] = trace.toSeq

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of distinct column sets evaluated (models fitted).
     */
    def nEvaluated: Int = seen.size

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Keep a probability inside [eps, 1 - eps].
     *  @param v  the probability
     */
    private def clamp (v: Double): Double = min (1.0 - eps, max (eps, v))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the metric of a step, with a failed fit (null or NaN) scored as worst.
     *  @param s  the step
     */
    private def score (s: BestStep): Double =
        if s == null || s.qof == null || s.qof(qk).isNaN then Fit.extreme (qk) else s.qof(qk)
    end score

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether metric value a is better than b (direction from `Fit.maxi`).
     *  @param a  the first metric value
     *  @param b  the second metric value
     */
    private def better (a: Double, b: Double): Boolean = if maxi then a > b else a < b

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Draw one column set from the current probabilities: each free column with
     *  probability p_j, at least one free column, at most `cap` columns (the least
     *  likely extras dropped), plus the fixed columns; in ascending column order.
     */
    private def draw (): LSET [Int] =
        val cols = VEC [Int] ()
        for j <- free if rng.gen < p(j) do cols += j
        if cols.isEmpty && free.nonEmpty then cols += free.maxBy (p(_))     // never an empty draw
        while cols.size + fixed.size > cap do cols -= cols.minBy (p(_))     // enforce the size cap
        LSET.from ((fixed ++ cols).toSeq.sorted)
    end draw

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Draw a column set not evaluated before, or None when `CEM_Selector.MAX_TRIES`
     *  draws in a row hit sets already seen (the distribution has collapsed).
     */
    private def sample (): Option [LSET [Int]] =
        var tries = 0
        while tries < CEM_Selector.MAX_TRIES do
            val cols = draw ()
            if ! seen.contains (cols.toSet) then return Some (cols)
            tries += 1
        end while
        None
    end sample

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Evaluate a column set, scoring a failed build as worst rather than stopping
     *  the run (a sampled set may be singular).
     *  @param cols  the column set
     */
    private def evaluate (cols: LSET [Int]): BestStep =
        val step = try eval (cols)
                   catch case e: Exception =>
                       debug ("evaluate", s"cols = $cols failed: ${e.getMessage}")
                       BestStep (-1, null, null, cols)()
        seen += cols.toSet
        step
    end evaluate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether the distribution is degenerate: every free column's
     *  probability sits at the floor or the ceiling.
     */
    private def degenerate: Boolean = free.forall (j => p(j) <= eps + 1E-9 || p(j) >= 1.0 - eps - 1E-9)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Run the Cross-Entropy Method and return the best step found: its model,
     *  its QoF vector and its column set (`mod_cols`).
     */
    def run (): BestStep =
        var stagnant = 0
        var gamma    = Fit.extreme (qk)                                   // the elite threshold (worst elite score)
        var memory   = VEC [BestStep] ()                                  // elites carried over from the last round
        breakable {
            for it <- 1 to maxIter do
                val round = VEC [BestStep] ()                             // this round's freshly evaluated sets
                var dry   = false
                while round.size < nS && ! dry do
                    sample () match
                    case Some (cols) => round += evaluate (cols)
                    case None        => dry = true
                end while
                if round.isEmpty then
                    debug ("run", s"round $it: no unseen sets left, stopping")
                    break ()

                val pool   = round ++ memory                                      // fresh samples plus remembered elites
                val ranked = if maxi then pool.sortBy (-score (_)) else pool.sortBy (score (_))
                val nElite = max (1, ceil (rho * round.size).toInt)
                val top    = ranked.take (nElite)
                val elites = if bestStep.qof != null && ! top.exists (_.mod_cols.toSet == bestStep.mod_cols.toSet)
                             then top :+ bestStep else top                        // elitism: the incumbent always counts

                for j <- free do                                                  // move p towards the elite frequencies
                    val freq = elites.count (_.mod_cols.contains (j)).toDouble / elites.size
                    p(j) = clamp ((1.0 - alpha) * p(j) + alpha * freq)

                val newGamma = score (top.last)                                   // the elite threshold of this round
                val tol      = CEM_Selector.GAMMA_TOL * (1.0 + math.abs (newGamma))   // relative improvement that counts
                if better (newGamma, gamma) && math.abs (newGamma - gamma) > tol then
                    gamma = newGamma; stagnant = 0
                else stagnant += 1
                if better (score (ranked.head), score (bestStep)) then bestStep = ranked.head
                memory = top.take (max (1, ceil (keep * nElite).toInt))          // elite memory for the next round
                trace += bestStep
                debug ("run", f"round $it%3d: evaluated ${round.size}%3d, gamma = $gamma%9.4f, best = ${score (bestStep)}%9.4f " +
                              s"cols = ${bestStep.mod_cols}, stagnant = $stagnant")
                if dry || stagnant >= patience || degenerate then break ()
            end for
        } // breakable
        if bestStep.qof == null then flaw ("run", "no column set could be evaluated")
        bestStep
    end run

end CEM_Selector


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CEM_Selector` companion object holds the default settings.
 */
object CEM_Selector:

    val N_SAMPLE   = -1                                                   // sets per round: -1 => C_N x free columns
    val C_N        = 5.0                                                  // sample size multiplier (tutorial: 5 to 10)
    val MIN_SAMPLE = 40                                                   // smallest sample size per round
    val ELITE      = -1.0                                                 // elite fraction: -1 => ln(n)/n in [0.05, 0.3]
    val ALPHA      = 0.7                                                  // smoothing rate (tutorial: 0.4 to 0.9)
    val EPS        = 0.05                                                 // probability floor / ceiling
    val MAX_ITER   = 30                                                   // maximum number of rounds
    val PATIENCE   = 5                                                    // rounds without a better elite threshold (tutorial: d = 5)
    val KEEP       = 0.3                                                  // fraction of elites remembered (iCEM: 0.3)
    val GAMMA_TOL  = 1E-4                                                 // relative improvement of gamma that resets patience
    val MAX_TRIES  = 200                                                  // draws before the sampler gives up

end CEM_Selector


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `cEM_SelectorTest` main function tests the `CEM_Selector` class on a
 *  synthetic regression: 12 candidate columns of which 3 (plus the intercept)
 *  generate y, compared by BIC (whose penalty rejects idle columns, unlike R^2-bar
 *  on nearly noise-free data).  The selector must recover exactly the true set,
 *  its inclusion probabilities must separate true from noise columns, the same
 *  stream must reproduce the same run, and a prior favouring the true columns
 *  must also recover them (its evaluation count is printed for comparison).
 *  > runMain scalation.modeling.cEM_SelectorTest
 */
@main def cEM_SelectorTest (): Unit =

    val rk     = QoF.bic.ordinal                                          // compare models by BIC (passed explicitly)
    val (m, n) = (120, 13)                                                // rows, columns (0 = intercept)
    val rn     = Random (5)
    val x      = new MatrixD (m, n)
    for i <- 0 until m do
        x(i, 0) = 1.0
        for j <- 1 until n do x(i, j) = rn.gen * 10.0
    val truth  = Set (0, 2, 5, 9)
    val y      = VectorD (for i <- 0 until m yield
                              3.0 + 2.0 * x(i, 2) - 1.5 * x(i, 5) + 0.8 * x(i, 9) + 6.0 * (rn.gen - 0.5))

    def evalCols (cols: LSET [Int]): BestStep =                           // build, train and test a regression on cols
        val mod = new Regression (x(?, cols), y)
        mod.train ()
        val qof = mod.test ()._2
        BestStep (-1, qof, mod, cols)(using rk)(qof(rk))
    end evalCols

    banner ("CEM over 12 candidate columns (intercept fixed), criterion BIC")
    val cem  = new CEM_Selector (n, evalCols, fixed = LSET (0), stream = 0)(using rk)
    val best = cem.run ()
    println (s"sample size per round = ${cem.nS}, elite fraction = ${cem.rho}")
    println (s"best cols = ${best.mod_cols}, BIC = ${best.qof(rk)}, evaluated = ${cem.nEvaluated}, rounds = ${cem.bestByRound.size}")
    println (s"inclusion probabilities = ${cem.probs}")
    assert (best.mod_cols.toSet == truth, s"expected exactly the true columns $truth, got ${best.mod_cols}")

    val pr = cem.probs
    val pT = truth.filter (_ != 0).map (pr(_)).min
    val pN = (1 until n).filterNot (truth.contains).map (pr(_)).max
    println (f"min p (true columns) = $pT%.3f, max p (noise columns) = $pN%.3f")
    assert (pT > pN, "true columns must end with higher inclusion probability than noise columns")

    banner ("reproducibility: the same stream gives the same result")
    val again = new CEM_Selector (n, evalCols, fixed = LSET (0), stream = 0)(using rk)
    val best2 = again.run ()
    assert (best2.mod_cols.toSet == best.mod_cols.toSet && again.nEvaluated == cem.nEvaluated,
            "the same stream must reproduce the run")

    banner ("a prior that favours the true columns")
    val prior = VectorD (for j <- 0 until n yield if truth.contains (j) then 0.8 else 0.2)
    val cemP  = new CEM_Selector (n, evalCols, fixed = LSET (0), prior = prior, stream = 0)(using rk)
    val bestP = cemP.run ()
    println (s"with prior: best cols = ${bestP.mod_cols}, evaluated = ${cemP.nEvaluated} (uniform prior: ${cem.nEvaluated})")
    assert (bestP.mod_cols.toSet == truth, "the prior run must also recover the true columns")

end cEM_SelectorTest

