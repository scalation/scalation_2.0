
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// YousefChange --------- Note: comment only -- the scaladoc below no longer names the sub-package built on it
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Tue Sep 15 09:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Screening of Candidate Exogenous Variables for
 *           Forecasting (which columns are eligible, which drivers, at which lag,
 *           with which sign)
 *
 *  Screening, as in `Model.screen`, is the cheap data-only filter that runs BEFORE
 *  any model is fitted; selection (which of the shortlisted variables enter the
 *  equation, at which lags, with which transforms) is the job of the search that
 *  follows (feature selection, CEM, the LLM).  Three steps, each usable alone:
 *   1. `pool`      -- from a data matrix with a header (e.g., `MatrixD.loadH`), every
 *                     column that can be a driver of the response: not the response,
 *                     not excluded, finite, and neither constant nor cumulative
 *                     (monotone) on the training rows
 *   2. `rank`      -- each candidate's cross-correlation with y at lags 1..maxLag on
 *                     the TRAINING rows only (a driver must LEAD the response;
 *                     @see `VectorD.ccorr`: x.ccorr (y, k) correlates x_i with y_i+k),
 *                     strongest first, next to its lag-0 correlation
 *   3. `shortlist` -- `Model.screen` itself, fed the time-series dependency matrix of
 *                     `dependency` (x-vs-y: the strongest lead; x-vs-x: lag-0
 *                     correlation): in decreasing dependency, keep a candidate when its
 *                     |ccf| exceeds thr1 and its correlation with every candidate
 *                     already kept stays below thr2 (collinearity), the first k kept
 *  and, for the common case, `screen` does the three in one call, returning the
 *  (xe, y) pair the forecasters take (`ARX_D (xe, y, hh)`, `ARX (xe, y, hh)`, ...)
 *  with the k chosen columns in the file's order.
 *  What the data cannot decide is whether a column is a driver or another
 *  measurement of the response (e.g., excess mortality next to Covid-19 deaths:
 *  ccf 0.86 at lag 1, the reporting delay), or whether a column is comparable
 *  over the whole period (case counts under changing testing regimes); that is
 *  domain knowledge, given through `exclude` (or, in an LLM-guided search, asked of the LLM).
 *  With those two excluded, the screen finds `Example_Covid.BEST2_EXO` (hospital
 *  load and the positivity rate) on its own -- the choice made explicit and
 *  repeatable for any dataset.
 */

package scalation
package modeling
package forecasting

import scala.collection.immutable.ListMap
import scala.collection.mutable.{ArrayBuffer => VEC}

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ExoRank` case class is one candidate's screening result.
 *  @param name     the candidate exogenous variable
 *  @param bestLag  the lag (1..maxLag) at which |ccf| with the response is largest
 *  @param ccf      the cross-correlation at that lag (signed)
 *  @param corr0    the lag-0 (contemporaneous) correlation with the response
 */
case class ExoRank (name: String, bestLag: Int, ccf: Double, corr0: Double):

    def dep: Double    = math.abs (ccf)                                 // the dependency with the response
    def leads: Boolean = dep > math.abs (corr0)                         // stronger ahead of y than alongside it

    override def toString: String =
        f"$name%-40s lag $bestLag%2d  ccf $ccf%7.3f  corr0 $corr0%7.3f" + (if leads then "  leads" else "")

end ExoRank


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ExoPool` case class is the eligible pool cut from a data matrix.
 *  @param y        the response series (full length)
 *  @param xe       the candidate series by name, in column order (full length)
 *  @param dropped  the columns left out, each with the reason
 */
case class ExoPool (y: VectorD, xe: Map [String, VectorD], dropped: Seq [(String, String)])


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ExoScreen` object pools, ranks and screens candidate exogenous variables.
 */
object ExoScreen:

    private val debug = debugf ("ExoScreen", false)                     // debug function
    private val flaw  = flawf ("ExoScreen")                             // flaw function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Cut the eligible pool from a data matrix with a header: every column but the
     *  response that is not excluded, has finite values, and is neither constant nor
     *  cumulative on the training rows.  A cumulative (running total) column moves
     *  in one direction for at least thrMono of its steps: it carries a trend, not a
     *  driver signal, so its correlation with the response would be spurious.
     *  @param data     the data matrix (rows = time, columns = named series)
     *  @param header   the column names (one per column of data)
     *  @param resp     the name of the response column
     *  @param tr       the number of leading (training) rows the rules look at
     *  @param exclude  the columns domain knowledge rules out (e.g., proxies of the response)
     *  @param thrMono  the fraction of one-direction steps from which a column is cumulative
     */
    def pool (data: MatrixD, header: Array [String], resp: String, tr: Int,
              exclude: Set [String] = Set (), thrMono: Double = 0.98): ExoPool =
        require (header.length == data.dim2, s"pool: ${header.length} names for ${data.dim2} columns")
        require (0 < tr && tr <= data.dim, s"pool: tr = $tr must be in 1..${data.dim}")
        val jy = header.indexOf (resp)
        require (jy >= 0, s"pool: response '$resp' not in the header ${header.mkString (", ")}")

        val y  = data(?, jy)
        if ! finite (y) then flaw ("pool", s"the response '$resp' has non-finite values")

        var xe      = ListMap [String, VectorD] ()                      // insertion (column) order kept
        val dropped = VEC [(String, String)] ()
        for j <- header.indices if j != jy do
            val nm  = header(j)                                         // variable/feature name
            val x   = data(?, j)                                        // column for variable x_j
            val why = if exclude contains nm then "excluded (domain knowledge)"
                      else if ! finite (x) then "non-finite values (not a numeric column?)"
                      else if x(0 until tr).stdev == 0.0 then "constant on the training rows"
                      else
                          val mono = monotone (x(0 until tr))
                          if mono >= thrMono then f"cumulative: $mono%.2f of its steps go one way on the training rows"
                          else null
            if why == null then xe = xe + (nm -> x)
            else
                debug ("pool", s"$nm dropped: $why")
                dropped += nm -> why
        end for
        ExoPool (y, xe, dropped.toSeq)
    end pool

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the cross-correlation of the response y_t with x_t-k for lags
     *  k = 1..maxLag on the first tr rows (NaN where undefined, e.g., a constant x).
     *  @param y       the response series (full length)
     *  @param x       the candidate series (full length)
     *  @param maxLag  the maximum lag examined
     *  @param tr      the number of leading (training) rows used
     */
    def ccf (y: VectorD, x: VectorD, maxLag: Int, tr: Int): VectorD =
        val (yTr, xTr) = (y(0 until tr), x(0 until tr))
        VectorD (for k <- 1 to maxLag yield xTr.ccorr (yTr, k))
    end ccf

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Rank the candidates by the largest |ccf| over lags 1..maxLag, strongest
     *  first, each with its lag-0 correlation; candidates whose ccf is undefined
     *  (non-finite data, a constant series) are left out.
     *  @param y       the response series (full length)
     *  @param xe      the candidate series by name (full length)
     *  @param maxLag  the maximum lag examined
     *  @param tr      the number of leading (training) rows used
     */
    def rank (y: VectorD, xe: Map [String, VectorD], maxLag: Int, tr: Int): Seq [ExoRank] =
        val ranks = VEC [ExoRank] ()
        for (nm, x) <- xe.toSeq.sortBy (_._1) do
            val c = ccf (y, x, maxLag, tr)
            if ! finite (c) then debug ("rank", s"$nm: ccf undefined, left out")
            else
                val k = c.abs.argmax ()
                ranks += ExoRank (nm, k + 1, c(k), x(0 until tr) corr y(0 until tr))
        end for
        ranks.toSeq.sortBy (r => -r.dep)
    end rank

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the [ x, y ] matrix of the training rows with the candidates in
     *  ranking order (strongest first) and y as the last column -- the layout
     *  `Model.screen` expects.
     *  @param ranks  the ranking (@see `rank`)
     *  @param y      the response series (full length)
     *  @param xe     the candidate series by name (full length)
     *  @param tr     the number of leading (training) rows used
     */
    def xyOf (ranks: Seq [ExoRank], y: VectorD, xe: Map [String, VectorD], tr: Int): MatrixD =
        MatrixD ((ranks.map (r => xe(r.name)(0 until tr)) :+ y(0 until tr)).toIndexedSeq).ᵀ
    end xyOf

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the dependency matrix `Model.screen` needs for time-series candidates:
     *  its x-vs-x block is the lag-0 correlation of the candidates (collinearity),
     *  while its x-vs-y column and row hold each candidate's strongest cross-
     *  correlation LEADING y over lags 1..maxLag (signed, from the ranking) instead
     *  of the lag-0 correlation `xy.corr` would give -- a driver is judged by how it
     *  moves ahead of the response, not alongside it.
     *  @param ranks  the ranking (@see `rank`), in the column order of xy
     *  @param xy     the [ x, y ] matrix of the training rows (@see `xyOf`)
     */
    def dependency (ranks: Seq [ExoRank], xy: MatrixD): MatrixD =
        val n = ranks.size
        require (xy.dim2 == n + 1, s"dependency: ${xy.dim2} columns for $n candidates + y")
        val dep = xy.corr                                               // lag-0 correlations on the training rows
        for i <- 0 until n do
            dep(i, n) = ranks(i).ccf                                    // x_i vs y: its strongest lead
            dep(n, i) = ranks(i).ccf
        end for
        dep
    end dependency

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Rank the candidates and keep a shortlist of at most k of them with
     *  `Model.screen` on the time-series dependency matrix: in decreasing lead
     *  dependency, keep a candidate when its |ccf| exceeds thr1 and its lag-0
     *  correlation with every candidate already kept is below thr2; the screen's
     *  first k (its greedy pass accepts in dependency order, so the first k of its
     *  uncapped result are exactly the capped result).  Returns the ranking and the
     *  shortlist (in ranking order).
     *  @param y       the response series (full length)
     *  @param xe      the candidate series by name (full length)
     *  @param maxLag  the maximum lag examined
     *  @param tr      the number of leading (training) rows used
     *  @param k       the largest shortlist wanted
     *  @param thr1    the smallest dependency with the response kept
     *  @param thr2    the largest correlation with an already kept candidate tolerated
     */
    def shortlist (y: VectorD, xe: Map [String, VectorD], maxLag: Int, tr: Int, k: Int,
                   thr1: Double = 0.05, thr2: Double = 0.95): (Seq [ExoRank], Seq [String]) =
        val ranks = rank (y, xe, maxLag, tr)
        val xy    = xyOf (ranks, y, xe, tr)
        val (_, sel) = Model.screen (xy, thr1, thr2)(dependency (ranks, xy))
        val kept  = sel.toArray.sorted.take (k).map (i => ranks(i).name).toSeq  // ascending index = ranking order
        debug ("shortlist", s"passed thr1 but not kept (collinear or beyond k = $k): " +
                            ranks.filter (r => r.dep > thr1 && ! kept.contains (r.name)).map (_.name).mkString (", "))
        (ranks, kept)
    end shortlist

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The one-call screen for a forecaster: from a data matrix with a header (e.g.,
     *  `MatrixD.loadH`), the response's name and k, return the exogenous matrix of
     *  the k variables chosen on the training rows -- all rows, columns in the file's
     *  order -- the response vector and the chosen names (in that order): the (xe, y)
     *  pair `ARX_D (xe, y, hh)` and the other forecasters take.
     *  @param data     the data matrix (rows = time, columns = named series)
     *  @param header   the column names (one per column of data)
     *  @param resp     the name of the response column
     *  @param k        the number of exogenous variables wanted (at most)
     *  @param maxLag   the maximum lead examined (the lag window of the forecaster)
     *  @param tr       the number of training rows (0 => `Model.trSize` of the rows)
     *  @param exclude  the columns domain knowledge rules out (proxies of the response)
     *  @param thr1     the smallest dependency with the response kept
     *  @param thr2     the largest correlation with an already kept candidate tolerated
     *  @param show     whether to print the dropped columns, the ranking and the choice
     */
    def screen (data: MatrixD, header: Array [String], resp: String, k: Int, maxLag: Int = 6,
                tr: Int = 0, exclude: Set [String] = Set (), thr1: Double = 0.05, thr2: Double = 0.95,
                show: Boolean = true): (MatrixD, VectorD, Array [String]) =
        val tr_ = if tr > 0 then tr else Model.trSize (data.dim)
        val p   = pool (data, header, resp, tr_, exclude)
        val (ranks, kept) = shortlist (p.y, p.xe, maxLag, tr_, k, thr1, thr2)
        val names = header.filter (kept.contains)                        // the chosen columns in the file's order
        val xe    = if names.isEmpty then new MatrixD (data.dim, 0)
                    else MatrixD (names.toIndexedSeq.map (p.xe)).ᵀ            // full columns, all rows
        if show then
            banner (s"ExoScreen.screen: response $resp, ${p.xe.size} candidates, first $tr_ rows, leads 1..$maxLag")
            for (nm, why) <- p.dropped do println (f"dropped $nm%-40s $why")
            for r <- ranks do println (r)
            println (s"chosen (k = $k, in the file's order): ${names.mkString (", ")}")
        end if
        (xe, p.y, names)
    end screen

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether every value of the vector is finite.
     *  @param x  the vector
     */
    private def finite (x: VectorD): Boolean = ! x.exists (v => v.isNaN || v.isInfinite)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the fraction of the steps x_i - x_i-1 that go one way (all up or all
     *  down => 1.0), the signature of a cumulative series.
     *  @param x  the vector
     */
    private def monotone (x: VectorD): Double =
        var up = 0
        var dn = 0
        for i <- 1 until x.dim do
            if x(i) >= x(i-1) then up += 1
            if x(i) <= x(i-1) then dn += 1
        math.max (up, dn).toDouble / math.max (1, x.dim - 1)
    end monotone

end ExoScreen


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `exoScreenTest` main function screens a pool of nine Covid-19 columns for
 *  the weekly deaths on the training rows (`Model.trSize` of 116 = 92).  What the
 *  data alone says: excess mortality (ccf 0.863 at lag 1) and new cases (0.814 at
 *  lag 2) come first, hospital and icu patients (lag 1) and the positivity rate
 *  (lag 3) follow; tests per case and new vaccinations correlate negatively.
 *  Excess mortality is another count of the same deaths (one week ahead by the
 *  reporting delay), which no statistic can tell from a driver, and new cases are
 *  not comparable over the period (testing regimes changed), so both are excluded
 *  by domain knowledge -- as in `Example_Covid`, which never uses them.  Then the
 *  shortlist of two is hospital patients and the positivity rate, i.e., `BEST2_EXO`
 *  with hospital load in place of icu load (collinear, 0.989 > thr2); with three
 *  kept, tests per case join (negative).  Last, the same `Model.screen` with its
 *  default lag-0 dependency shows why the lead matters: it would take new tests
 *  before the positivity rate, which leads deaths by three weeks (2nd by lead,
 *  only 4th alongside).
 *  > runMain scalation.modeling.forecasting.exoScreenTest
 */
@main def exoScreenTest (): Unit =

    val pool  = Array ("new_cases", "reproduction_rate", "icu_patients", "hosp_patients", "new_tests",
                       "positive_rate", "tests_per_case", "new_vaccinations", "excess_mortality")
    val data  = Example_Covid.context (pool)
    val y     = data ("y")
    val xe    = data - "y"
    val tr    = Model.trSize (y.dim)
    val (ranks, short) = ExoScreen.shortlist (y, xe, maxLag = 6, tr, k = 2)

    banner (s"ranking on the first $tr rows (lags 1..6)")
    for r <- ranks do println (r)
    banner ("shortlist (k = 2), data only")
    println (short.mkString (", "))

    assert (ranks.map (_.name).take (5) == Seq ("excess_mortality", "new_cases", "hosp_patients", "positive_rate", "icu_patients"),
            "ranking: excess mortality, new cases, hosp, positivity rate, icu")
    assert (ranks.sliding (2).forall (p => p(0).dep >= p(1).dep), "ranking is by decreasing dependency")
    assert (ranks.forall (r => 1 <= r.bestLag && r.bestLag <= 6), "best lags lie in 1..maxLag")
    assert (ranks.find (_.name == "new_cases").get.bestLag == 2, "new cases lead deaths by two weeks")
    assert (ranks.filter (r => r.name == "tests_per_case" || r.name == "new_vaccinations").forall (_.ccf < 0),
            "tests per case and new vaccinations correlate negatively with deaths")
    assert (short == Seq ("excess_mortality", "new_cases"), "the data alone shortlists excess mortality and new cases")

    banner ("shortlist with excess_mortality (a proxy of the response) and new_cases (testing regimes) excluded")
    val xe2 = xe - "excess_mortality" - "new_cases"
    val (_, short2) = ExoScreen.shortlist (y, xe2, maxLag = 6, tr, k = 2)
    val (_, short3) = ExoScreen.shortlist (y, xe2, maxLag = 6, tr, k = 3)
    println (s"k = 2: ${short2.mkString (", ")}")
    println (s"k = 3: ${short3.mkString (", ")}")
    assert (short2 == Seq ("hosp_patients", "positive_rate"), "k = 2: hospital patients and the positivity rate (BEST2_EXO, hosp for icu)")
    assert (short3 == Seq ("hosp_patients", "positive_rate", "tests_per_case"), "k = 3: tests per case join (icu collinear with hosp)")

    banner ("the same Model.screen with its default dependency (lag 0): drivers judged alongside y, not ahead of it")
    val ranks2   = ExoScreen.rank (y, xe2, maxLag = 6, tr)
    val xy       = ExoScreen.xyOf (ranks2, y, xe2, tr)
    val (_, sel) = Model.screen (xy, 0.05, 0.95)()                     // dep = xy.corr, lag 0 only
    val lag0     = sel.toArray.sortBy (i => -math.abs (ranks2(i).corr0)).take (3).map (i => ranks2(i).name).toSeq
    println (s"lag-0 screen, k = 3: ${lag0.mkString (", ")}")
    println (s"lead screen,  k = 3: ${short3.mkString (", ")}")
    val byLead = ranks2.map (_.name)
    val byLag0 = ranks2.sortBy (r => -math.abs (r.corr0)).map (_.name)
    println (s"positive_rate: rank ${byLead.indexOf ("positive_rate") + 1} by lead, ${byLag0.indexOf ("positive_rate") + 1} by lag 0")
    assert (lag0 == Seq ("hosp_patients", "new_tests", "positive_rate"), "lag 0 would take new tests before the positivity rate")
    assert (byLead.indexOf ("positive_rate") == 1 && byLag0.indexOf ("positive_rate") == 3,
            "the positivity rate leads deaths by three weeks: 2nd by lead, 4th alongside")
    println ("exoScreenTest: passed")

end exoScreenTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `exoScreenTest2` main function does the whole screening automatically from
 *  the file: `MatrixD.loadH` reads covid_19_weekly.csv with its header (the date
 *  column skipped), the rows every Covid-19 test uses are kept (0 until 116, @see
 *  `Example_Covid.clip`), `pool` decides which of the 15 other columns are eligible
 *  (the five running totals -- people vaccinated, fully vaccinated, boosters, and
 *  the two cumulative excess-mortality columns -- are dropped as cumulative), and
 *  `shortlist` picks from what is left.  Nothing about the columns is written in
 *  the test except the response's name and, in the second part, the two domain
 *  exclusions (excess mortality, new cases); the outcome must equal `exoScreenTest`'s.
 *  > runMain scalation.modeling.forecasting.exoScreenTest2
 */
@main def exoScreenTest2 (): Unit =

    val (data, header) = MatrixD.loadH ("covid_19_weekly.csv", skipCol = 1) // the date column is skipped
    val rows = data (0 until 116)                                       // the rows every Covid-19 test uses
    val tr   = Model.trSize (rows.dim)
    val resp = "new_deaths"
    println (s"columns = ${header.mkString (", ")}")
    println (s"rows = ${rows.dim}, training rows = $tr, response = $resp")

    banner ("pool: eligible columns and the reasons for dropping the others")
    val p = ExoScreen.pool (rows, header, resp, tr)
    println (s"candidates (${p.xe.size}): ${p.xe.keys.mkString (", ")}")
    for (nm, why) <- p.dropped do println (f"dropped $nm%-40s $why")

    val cumulative = Set ("people_vaccinated", "people_fully_vaccinated", "total_boosters",
                          "excess_mortality_cumulative_absolute", "excess_mortality_cumulative_per_million")
    assert (! p.xe.contains (resp), "the response is not a candidate")
    assert (p.dropped.map (_._1).toSet == cumulative, "exactly the five running totals are dropped")
    assert (p.dropped.forall (_._2.startsWith ("cumulative")), "each is dropped as cumulative")
    assert (p.xe.size == 10 && p.xe.keys.toSeq == header.filterNot (n => n == resp || cumulative.contains (n)).toSeq,
            "the candidates are the other columns, in column order")
    assert (p.y == Example_Covid.context (Array ("icu_patients"))("y"), "the response equals the example loader's y")

    banner (s"ranking and shortlist (k = 2) on the first $tr rows, data only")
    val (ranks, short) = ExoScreen.shortlist (p.y, p.xe, maxLag = 6, tr, k = 2)
    for r <- ranks do println (r)
    println (s"shortlist: ${short.mkString (", ")}")
    assert (ranks.size == 10 && ranks.head.name == "excess_mortality", "excess mortality ranks first (a proxy of deaths)")
    assert (short == Seq ("excess_mortality", "new_cases"), "the data alone shortlists excess mortality and new cases")

    banner ("with the domain exclusions of excess_mortality and new_cases")
    val p2 = ExoScreen.pool (rows, header, resp, tr, exclude = Set ("excess_mortality", "new_cases"))
    val (_, short2) = ExoScreen.shortlist (p2.y, p2.xe, maxLag = 6, tr, k = 2)
    val (_, short3) = ExoScreen.shortlist (p2.y, p2.xe, maxLag = 6, tr, k = 3)
    println (s"dropped: ${p2.dropped.map (d => s"${d._1} (${d._2})").mkString ("; ")}")
    println (s"k = 2: ${short2.mkString (", ")}")
    println (s"k = 3: ${short3.mkString (", ")}")
    assert (p2.dropped.count ((nm, why) => Set ("excess_mortality", "new_cases").contains (nm) && why.startsWith ("excluded")) == 2,
            "both exclusions are recorded")
    assert (short2 == Seq ("hosp_patients", "positive_rate"), "k = 2 as in exoScreenTest")
    assert (short3 == Seq ("hosp_patients", "positive_rate", "tests_per_case"), "k = 3 as in exoScreenTest")
    println ("exoScreenTest2: passed")

end exoScreenTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `exoScreenTest3` main function is the user's path from a file to a forecaster
 *  in three lines: load the file with its header, screen it for the response and k,
 *  build the model on what was chosen (excess mortality and new cases excluded, as
 *  everywhere).  With k = 3 the chosen columns come back in the file's order
 *  (hospital patients, positivity rate, tests per case); the `ARX_D` model on the
 *  k = 2 choice -- hospital patients and the positivity rate, `BEST2_EXO` with
 *  hospital for icu load -- is then trained and tested exactly as in `aRX_DTest4`
 *  (TnT with rolling validation), so its sMAPE can be set against the BEST2_EXO run.
 *  > runMain scalation.modeling.forecasting.exoScreenTest3
 */
@main def exoScreenTest3 (): Unit =

    val (data, header) = MatrixD.loadH ("covid_19_weekly.csv", skipCol = 1) // 1. the file, date column skipped
    val rows = data (0 until 116)                                       //    the rows every Covid-19 test uses
    val exclude = Set ("excess_mortality", "new_cases")                 //    domain knowledge: a proxy of y, testing regimes
    val (xe3, _, names3) = ExoScreen.screen (rows, header, "new_deaths", k = 3, exclude = exclude)
    assert (names3.toSeq == Seq ("hosp_patients", "positive_rate", "tests_per_case"), "k = 3 in the file's order")
    assert (xe3.dims == (116, 3) && xe3(?, 1) == rows(?, header.indexOf ("positive_rate")), "xe holds the full chosen columns")

    val (xe, y, names) = ExoScreen.screen (rows, header, "new_deaths", k = 2, exclude = exclude)   // 2. the k exogenous variables
    assert (names.toSeq == Seq ("hosp_patients", "positive_rate") && xe.dims == (116, 2) && y.dim == 116)

    val hh = 6                                                          // 3. the forecaster on the choice
    RidgeRegression.hp("lambda") = 1.0
    AR.hp.set (("p", 6), ("px", 6), ("spec", 0))                        // the paper's baseline: p = px = maxLag
    val mod = ARX_D (xe, y, hh)
    banner (s"TnT Forecasts: ${mod.modelName} on ${names.mkString (", ")}")
    mod.tnT_Test ()                                                     // Train and Test with Rolling Validation
    println ("exoScreenTest3: passed")

end exoScreenTest3

