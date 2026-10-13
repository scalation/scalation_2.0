
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Tue Sep 15 15:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Subset Search Algorithms for Feature Selection, written once
 *
 *  The greedy feature-selection techniques (`Forward`, `Backward`, `Stepwise`, `Beam`)
 *  as pure algorithms over an EVALUATOR, in the way `CEM_Selector` is already written:
 *  the algorithm decides which column sets to try, the host (a regression, a
 *  forecaster at one horizon or jointly over horizons, a formula term library)
 *  supplies `eval: LSET [Int] => BestStep`, which builds, trains and scores one
 *  column set and returns a `BestStep` whose qof holds the criterion qk (qof null
 *  => the fit failed and the set is skipped).  Nothing here prints, builds a model
 *  or knows about rows, spaces or horizons -- all of that is the evaluator's, so
 *  the contract "every QoF compared is on the original scale, on the selection rows"
 *  is met in ONE place per host.  `run` dispatches on `SelectionTech`; `cem` wraps
 *  `CEM_Selector`.  Every search returns a `Path`: the best-so-far after each step
 *  (the trace hosts turn into their key-metrics table), the best step found and
 *  the number of sets evaluated.
 *
 *  Conventions (shared by all techniques):
 *      fixed   columns in every set (e.g., the intercept column 0; empty for a
 *              `fromTerms` matrix or a term library)
 *      units   columns that enter or leave together (a sin/cos wave pair);
 *              defaults to singletons
 *      ties    the first candidate found wins (as in `Predictor.forwardSel`)
 *  @note    AI Assisted Code
 */

package scalation
package modeling

import scala.collection.mutable.{ArrayBuffer => VEC, LinkedHashSet => LSET}

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SubsetSearch` object holds the subset search algorithms behind the
 *  feature-selection techniques.
 */
object SubsetSearch:

    private val debug = debugf ("SubsetSearch", false)                   // debug function
    private val flaw  = flawf ("SubsetSearch")                           // flaw function

    /** The move set of the beam search: (columns added, columns removed) per move.
     */
    val BEAM_MOVES: Seq [(Int, Int)] = Seq ((1, 0), (2, 0), (0, 1), (0, 2), (1, 1), (2, 1))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The result of a search: the best-so-far after each step (the trace), the
     *  best step found, the number of column sets evaluated and, for the
     *  cross-entropy method, the final inclusion probabilities (null otherwise).
     *  @param steps  the best-so-far after each step, in order
     *  @param best   the best step found (qof null when nothing could be evaluated)
     *  @param nEval  the number of column sets evaluated
     *  @param probs  the inclusion probability per column (CEM only)
     */
    case class Path (steps: VEC [BestStep], best: BestStep, nEval: Int, probs: VectorD = null):

        def cols: LSET [Int] = if best.mod_cols == null then LSET [Int] () else best.mod_cols
        def found: Boolean   = best.qof != null

    end Path

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the better of two steps, tolerating failed (qof null) steps: a
     *  failed candidate never replaces the current one, and any fitted candidate
     *  replaces a failed current one.  Ties keep the current step.
     *  @param cur   the current best step
     *  @param cand  the candidate step
     *  @param qk    index of Quality of Fit (QoF) to use for comparing quality
     */
    def pick (cur: BestStep, cand: BestStep)(using qk: Int): BestStep =
        if cand == null || cand.qof == null then cur
        else if cur == null || cur.qof == null then cand
        else if cand gt cur.qof(qk) then cand
        else cur
    end pick

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Evaluate one column set through the evaluator and return the step with
     *  its column index and set recorded (a failed evaluation keeps qof null).
     *  @param eval  the evaluator
     *  @param j     the column the move is about (the added, or the removed one)
     *  @param cols  the column set
     *  @param qk    index of Quality of Fit (QoF) to use for comparing quality
     */
    private def step (eval: LSET [Int] => BestStep, j: Int, cols: LSET [Int])(using qk: Int): BestStep =
        val s = eval (cols)
        if s == null || s.qof == null then BestStep (j, null, null, cols)()
        else BestStep (j, s.qof, s.mod, cols)(s.qof(qk))
    end step

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the free columns of a set (those not fixed), one per unit.
     *  @param n      the number of columns
     *  @param fixed  the columns always kept
     *  @param units  the grouping of columns into units
     */
    private def freeUnits (n: Int, fixed: LSET [Int], units: Int => Set [Int]): Seq [Set [Int]] =
        val seen = LSET [Int] ()
        val out  = VEC [Set [Int]] ()
        for j <- 0 until n if ! fixed.contains (j) && ! seen.contains (j) do
            val u = units (j).filterNot (fixed.contains)
            seen ++= u
            out  += u
        end for
        out.toSeq
    end freeUnits

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a copy of a column set with a unit added (in column order).
     *  @param cols  the column set
     *  @param u     the unit to add
     */
    private def plus (cols: LSET [Int], u: Set [Int]): LSET [Int] =
        val out = cols.clone ()
        for j <- u.toSeq.sorted do out += j
        out
    end plus

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a copy of a column set with a unit removed.
     *  @param cols  the column set
     *  @param u     the unit to remove
     */
    private def minus (cols: LSET [Int], u: Set [Int]): LSET [Int] = cols.filterNot (u.contains)

//  T H E   T E C H N I Q U E S

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** FORWARD SELECTION: start from the fixed columns and at each step add the
     *  unit whose addition gives the best criterion; go on until every column is
     *  in (or `maxCols` is reached) and return the best set seen along the path,
     *  as `Predictor.forwardSelAll` does (it does not stop at the first step that
     *  fails to improve).  The first step of the trace is the fixed set itself
     *  when it is non-empty (e.g., the intercept-only model).
     *  @param n        the number of columns 0 until n
     *  @param eval     the evaluator of a column set
     *  @param fixed    the columns in every set (defaults to the intercept column 0)
     *  @param units    the grouping of columns into units (defaults to singletons)
     *  @param maxCols  the largest set allowed (-1 => no cap)
     *  @param qk       index of Quality of Fit (QoF) to use for comparing quality
     */
    def forward (n: Int, eval: LSET [Int] => BestStep, fixed: LSET [Int] = LSET (0),
                 units: Int => Set [Int] = Set (_), maxCols: Int = -1)(using qk: Int): Path =
        val cap   = if maxCols < 0 then n else maxCols
        val steps = VEC [BestStep] ()
        var nEval = 0
        var cols  = LSET.from (fixed)
        var best  = if cols.isEmpty then BestStep ()()
                    else { nEval += 1; step (eval, -1, cols) }
        if best.qof != null then steps += best
        var go    = cols.size < cap
        while go do
            var cand = BestStep ()()
            for u <- freeUnits (n, fixed, units) if u.forall (j => ! cols.contains (j)) do
                val cj = plus (cols, u)
                if cj.size <= cap then
                    nEval += 1
                    cand = pick (cand, step (eval, u.min, cj))
            end for
            if cand.qof == null then go = false                          // nothing to add (or every addition failed)
            else
                cols   = cand.mod_cols
                steps += cand
                best   = pick (best, cand)
                go     = cols.size < cap && cols.size < n
        end while
        debug ("forward", s"$nEval sets evaluated => cols = ${best.mod_cols}")
        Path (steps, best, nEval)
    end forward

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** BACKWARD ELIMINATION: start from every column and at each step remove the
     *  unit whose removal gives the best criterion; go on down to `minCols`
     *  columns (never below the fixed ones) and return the best set seen along
     *  the path, as `Predictor.backwardElimAll` does.  The first step of the
     *  trace is the full set.
     *  @param n        the number of columns 0 until n
     *  @param eval     the evaluator of a column set
     *  @param fixed    the columns in every set (defaults to the intercept column 0)
     *  @param units    the grouping of columns into units (defaults to singletons)
     *  @param minCols  the smallest set kept (defaults to 1)
     *  @param qk       index of Quality of Fit (QoF) to use for comparing quality
     */
    def backward (n: Int, eval: LSET [Int] => BestStep, fixed: LSET [Int] = LSET (0),
                  units: Int => Set [Int] = Set (_), minCols: Int = 1)(using qk: Int): Path =
        val floor = math.max (minCols, fixed.size)
        val steps = VEC [BestStep] ()
        var nEval = 1
        var cols  = LSET.from (0 until n)
        var best  = step (eval, -1, cols)
        if best.qof != null then steps += best
        var go    = cols.size > floor
        while go do
            var cand = BestStep ()()
            for u <- freeUnits (n, fixed, units) if u.forall (cols.contains) do
                val cj = minus (cols, u)
                if cj.size >= floor then
                    nEval += 1
                    cand = pick (cand, step (eval, u.min, cj))
            end for
            if cand.qof == null then go = false                          // nothing removable (or every removal failed)
            else
                cols   = cand.mod_cols
                steps += cand
                best   = pick (best, cand)
                go     = cols.size > floor
        end while
        debug ("backward", s"$nEval sets evaluated => cols = ${best.mod_cols}")
        Path (steps, best, nEval)
    end backward

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** STEPWISE SELECTION: at each step compare the best forward move with the
     *  best backward move and take the forward one when it is at least as good
     *  (within `slack`, which favours additions early on: slack / l^2 at step l)
     *  and improves on the last accepted criterion (within `slack`), else the
     *  backward one when it improves, else -- when `swap` -- the swap of the two
     *  columns when that improves; stop when no move improves, or when the
     *  addition would re-add the unit just removed (the slack let a slightly
     *  worse addition in, its removal then improved: the search has converged
     *  and would otherwise cycle until the step limit).  The rule and the slack
     *  are those of `Predictor.stepwiseSelAll`; the swap is evaluated by the same
     *  evaluator as every other set.
     *  @param n      the number of columns 0 until n
     *  @param eval   the evaluator of a column set
     *  @param fixed  the columns in every set (defaults to the intercept column 0)
     *  @param units  the grouping of columns into units (defaults to singletons)
     *  @param swap   whether to allow a swap step (one column out, one in)
     *  @param slack  the slack base (positive when qk is maximized, negative when
     *                    minimized; NaN => the magnitude of the package's `slack_base`
     *                    in the direction of qk, i.e., 25 or -25)
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     */
    def stepwise (n: Int, eval: LSET [Int] => BestStep, fixed: LSET [Int] = LSET (0),
                  units: Int => Set [Int] = Set (_), swap: Boolean = true,
                  slack: Double = Double.NaN)(using qk: Int): Path =
        val slack_ = if ! slack.isNaN then slack
                     else if Fit.maxi.contains (qk) then math.abs (slack_base) else - math.abs (slack_base)
        val steps  = VEC [BestStep] ()
        var nEval  = 0
        var cols   = LSET.from (fixed)
        var best   = BestStep ()()
        var last_q = Fit.extreme (qk)                                    // the last accepted criterion
        var prev   = (0, -1)                                             // the previous move: (+1 added / -1 removed, its unit)
        var go     = true
        var l      = 1
        while go && l < n do
            val units_ = freeUnits (n, fixed, units)
            var bestf  = BestStep ()()                                   // best forward move
            var bestb  = BestStep ()()                                   // best backward move
            for u <- units_ if u.forall (j => ! cols.contains (j)) do
                nEval += 1
                bestf  = pick (bestf, step (eval, u.min, plus (cols, u)))
            end for
            for u <- units_ if u.forall (cols.contains) && cols.size - u.size >= math.max (1, fixed.size) do
                nEval += 1
                bestb  = pick (bestb, step (eval, u.min, minus (cols, u)))
            end for
            val sl   = slack_ / (l * l)                                  // slack shrinks with the step
            val fOk  = bestf.qof != null
            val bOk  = bestb.qof != null

            if fOk && (! bOk || (bestf ge bestb.qof(qk) - sl)) && (bestf gt last_q - sl) then
                if prev == (-1, bestf.col) then go = false               // would re-add the unit just removed: converged
                else
                    cols   = bestf.mod_cols                              // ADD the forward unit
                    last_q = bestf.qof(qk)
                    steps += bestf
                    best   = pick (best, bestf)
                    prev   = (1, bestf.col)
            else if bOk && (bestb gt last_q) then
                cols   = bestb.mod_cols                                  // REMOVE the backward unit
                last_q = bestb.qof(qk)
                steps += bestb
                best   = pick (best, bestb)
                prev   = (-1, bestb.col)
            else if swap && fOk && bOk then
                val cj = plus (minus (cols, units (bestb.col)), units (bestf.col))
                nEval += 1
                val bestfb = step (eval, bestf.col, cj)
                if bestfb.qof != null && (bestfb gt last_q) then
                    cols   = cj
                    last_q = bestfb.qof(qk)
                    steps += bestfb
                    best   = pick (best, bestfb)
                    prev   = (0, -1)                                     // a swap is not undone by one move
                else go = false
            else go = false
            l += 1
        end while
        debug ("stepwise", s"$nEval sets evaluated => cols = ${best.mod_cols}")
        Path (steps, best, nEval)
    end stepwise

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** BEAM SEARCH: phase I evaluates every single unit added to the fixed
     *  columns and keeps the `bk` best sets (the beam); phase II expands every
     *  set of the beam by the `moves` (columns added, columns removed), keeps
     *  the `bk` best new sets, and repeats until `patience` rounds pass without
     *  a better best or no new set can be formed.  The moves and the patience
     *  are those of `Forecaster_Reg.beamSelAll`; here the beam is ordered by the
     *  criterion in its own direction (maximized or minimized), so the search is
     *  correct for R^2 as well as for sMAPE.
     *  @param n         the number of columns 0 until n
     *  @param eval      the evaluator of a column set
     *  @param fixed     the columns in every set (defaults to the intercept column 0)
     *  @param units     the grouping of columns into units (defaults to singletons)
     *  @param bk        the beam width (the sets kept per round)
     *  @param moves     the move set, (units added, units removed) per move
     *  @param patience  the rounds without a better best after which the search stops
     *  @param qk        index of Quality of Fit (QoF) to use for comparing quality
     */
    def beam (n: Int, eval: LSET [Int] => BestStep, fixed: LSET [Int] = LSET (0),
              units: Int => Set [Int] = Set (_), bk: Int = 3, moves: Seq [(Int, Int)] = BEAM_MOVES,
              patience: Int = 4)(using qk: Int): Path =
        val maxi   = Fit.maxi.contains (qk)
        def rank (s: BestStep): Double = if maxi then -s.qof(qk) else s.qof(qk)   // ascending = better first
        def top (cands: Seq [BestStep]): Seq [BestStep] = cands.filter (_.qof != null).sortBy (rank).take (bk)

        def keep (pool: VEC [BestStep], s: BestStep): Unit =
            if s.qof != null then
                pool += s
                if pool.size > bk then                                      // drop the worst (the latest among ties,
                    val worst = pool.map (rank).max                         // as a stable sort would)
                    pool.remove (pool.lastIndexWhere (rank (_) == worst))
        end keep

        val units_ = freeUnits (n, fixed, units)
        val steps  = VEC [BestStep] ()
        val seen   = LSET [Set [Int]] ()
        var nEval  = 0

        // phase I: one unit each
        val first  = VEC [BestStep] ()
        for u <- units_ do
            val cj = plus (LSET.from (fixed), u)
            seen  += cj.toSet
            nEval += 1
//          first += step (eval, u.min, cj)
            keep (first, step (eval, u.min, cj))                            // bounded pool
        end for

        var beam_ = top (first.toSeq)
        if beam_.isEmpty then
            flaw ("beam", "no single-unit set could be evaluated")
            return Path (steps, BestStep ()(), nEval)
        var best   = beam_.head
        steps += best

        // phase II: local search by the move set
        var left = patience
        var go   = true
        while go && left > 0 do
            val next = VEC [BestStep] ()
            for s <- beam_ do
                val cols  = s.mod_cols
                val in_   = units_.filter (u => u.forall (j => ! cols.contains (j)))
                val out_  = units_.filter (u => u.forall (cols.contains))
                for (nIn, nOut) <- moves if in_.size >= nIn && out_.size >= nOut do
                    for added <- in_.combinations (nIn); removed <- out_.combinations (nOut) do
                        var cj = cols
                        for u <- removed do cj = minus (cj, u)
                        for u <- added   do cj = plus (cj, u)
                        val key = cj.toSet
                        if cj.size > fixed.size && ! seen.contains (key) then
                            seen  += key
                            nEval += 1
                            val j  = if added.nonEmpty then added.head.min else removed.head.min
//                          next  += step (eval, j, cj)
                            keep (next, step (eval, j, cj))                 // bounded pool
                    end for
                end for
            end for
            val nextBeam = top (next.toSeq)
            if nextBeam.isEmpty then go = false
            else
                val cand = nextBeam.head
                if cand gt best.qof(qk) then
                    best = cand
                    left = patience
                else left -= 1
                beam_  = nextBeam
                steps += best
        end while
        debug ("beam", s"$nEval sets evaluated => cols = ${best.mod_cols}")
        Path (steps, best, nEval)
    end beam

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** CROSS-ENTROPY SELECTION: run `CEM_Selector` on the evaluator and return
     *  its trace (best-so-far per round), best set, evaluations and final
     *  inclusion probabilities.
     *  @param n        the number of columns 0 until n
     *  @param eval     the evaluator of a column set
     *  @param fixed    the columns in every set (defaults to the intercept column 0)
     *  @param prior    the prior inclusion probability per column (null => 0.5 each)
     *  @param nSample  the number of column sets sampled per round
     *  @param maxIter  the maximum number of rounds
     *  @param maxCols  the largest set allowed (-1 => no cap)
     *  @param stream   the random number stream
     *  @param qk       index of Quality of Fit (QoF) to use for comparing quality
     */
    def cem (n: Int, eval: LSET [Int] => BestStep, fixed: LSET [Int] = LSET (0), prior: VectorD = null,
             nSample: Int = CEM_Selector.N_SAMPLE, maxIter: Int = CEM_Selector.MAX_ITER,
             maxCols: Int = -1, stream: Int = 0)(using qk: Int): Path =
        val sel  = new CEM_Selector (n, eval, fixed, prior, nSample, maxIter = maxIter, maxCols = maxCols, stream = stream)
        val best = sel.run ()
        Path (VEC.from (sel.bestByRound), best, sel.nEvaluated, sel.probs)
    end cem

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Run the search of a selection technique (the dispatcher hosts call).
     *  @param tech   the selection technique
     *  @param n      the number of columns 0 until n
     *  @param eval   the evaluator of a column set
     *  @param fixed  the columns in every set (defaults to the intercept column 0)
     *  @param units  the grouping of columns into units (defaults to singletons)
     *  @param swap   whether stepwise may swap
     *  @param bk     the beam width
     *  @param prior  the prior inclusion probabilities (CEM)
     *  @param stream the random number stream (CEM)
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     */
    def run (tech: SelectionTech, n: Int, eval: LSET [Int] => BestStep, fixed: LSET [Int] = LSET (0),
             units: Int => Set [Int] = Set (_), swap: Boolean = true, bk: Int = 3,
             prior: VectorD = null, stream: Int = 0)(using qk: Int): Path =
        tech match
        case SelectionTech.Forward  => forward  (n, eval, fixed, units)
        case SelectionTech.Backward => backward (n, eval, fixed, units)
        case SelectionTech.Stepwise => stepwise (n, eval, fixed, units, swap)
        case SelectionTech.Beam     => beam     (n, eval, fixed, units, bk)
        case SelectionTech.CEM      => cem      (n, eval, fixed, prior, stream = stream)
    end run

end SubsetSearch


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `subsetSearchTest` main function tests every technique of `SubsetSearch`
 *  on a synthetic set function that needs no model: with a planted set P over
 *  columns 0 until 8 (column 0 fixed), the criterion of a set S is the size of
 *  the symmetric difference |S Δ P| plus a small tie-breaker, to be MINIMIZED
 *  (qk = smapeC), so that every technique must return exactly P, and the
 *  evaluation counts and trace lengths are what the algorithms promise.  A
 *  second function plants a sin/cos "wave pair" (columns 4 and 5 as one unit)
 *  and checks that the pair enters and leaves as a whole.
 *  > runMain scalation.modeling.subsetSearchTest
 */
@main def subsetSearchTest (): Unit =

    given qk: Int = QoF.smapeC.ordinal                                   // minimized criterion

    def evalFor (planted: Set [Int]): LSET [Int] => BestStep = (cols: LSET [Int]) =>
        val s   = cols.toSet
        val qof = new VectorD (Fit.N_QoF)
        qof(qk) = ((s diff planted).size + (planted diff s).size).toDouble + 0.001 * s.size
        BestStep (-1, qof, null, cols)(qof(qk))

    val n       = 8
    val planted = Set (0, 1, 3, 6)
    val eval    = evalFor (planted)

    banner ("forward")
    val pf = SubsetSearch.forward (n, eval)
    println (s"forward:  cols = ${pf.cols}, best = ${pf.best.qof(qk)}, steps = ${pf.steps.size}, nEval = ${pf.nEval}")
    assert (pf.cols.toSet == planted, s"forward must find the planted set: ${pf.cols}")
    assert (pf.steps.size == n, "forward's trace: the fixed set then one row per added column")
    assert (pf.nEval == 1 + 7 + 6 + 5 + 4 + 3 + 2 + 1, s"forward evaluations: ${pf.nEval}")

    banner ("backward")
    val pb = SubsetSearch.backward (n, eval)
    println (s"backward: cols = ${pb.cols}, best = ${pb.best.qof(qk)}, steps = ${pb.steps.size}, nEval = ${pb.nEval}")
    assert (pb.cols.toSet == planted, s"backward must find the planted set: ${pb.cols}")
    assert (pb.steps.size == n, "backward's trace: the full set then one row per removed column")

    banner ("stepwise")
    val ps = SubsetSearch.stepwise (n, eval, slack = 0.0)
    println (s"stepwise: cols = ${ps.cols}, best = ${ps.best.qof(qk)}, steps = ${ps.steps.size}, nEval = ${ps.nEval}")
    assert (ps.cols.toSet == planted, s"stepwise must find the planted set: ${ps.cols}")
    assert (ps.steps.size == 3, "stepwise stops once no move improves: three additions")

    banner ("beam")
    val pm = SubsetSearch.beam (n, eval, bk = 3)
    println (s"beam:     cols = ${pm.cols}, best = ${pm.best.qof(qk)}, steps = ${pm.steps.size}, nEval = ${pm.nEval}")
    assert (pm.cols.toSet == planted, s"beam must find the planted set: ${pm.cols}")

    banner ("cem")
    val pc = SubsetSearch.cem (n, eval, stream = 0)
    println (s"cem:      cols = ${pc.cols}, best = ${pc.best.qof(qk)}, rounds = ${pc.steps.size}, nEval = ${pc.nEval}")
    println (s"cem:      probs = ${pc.probs}")
    assert (pc.cols.toSet == planted, s"cem must find the planted set: ${pc.cols}")
    assert (pc.probs != null && pc.probs.dim == n, "cem returns one inclusion probability per column")

    banner ("run: every technique through the dispatcher agrees")
    for tech <- SelectionTech.values do
        val p = SubsetSearch.run (tech, n, eval)
        println (s"$tech: cols = ${p.cols}")
        assert (p.cols.toSet == planted, s"$tech through run must find the planted set")

    banner ("units: a wave pair (columns 4, 5) moves as a whole")
    val pair    = Set (4, 5)
    val unitOf  = (j: Int) => if pair.contains (j) then pair else Set (j)
    val planted2 = Set (0, 2, 4, 5)
    val eval2    = evalFor (planted2)
    val pf2 = SubsetSearch.forward (n, eval2, units = unitOf)
    val pb2 = SubsetSearch.backward (n, eval2, units = unitOf)
    println (s"forward with units:  cols = ${pf2.cols}, steps = ${pf2.steps.map (_.mod_cols.size)}")
    println (s"backward with units: cols = ${pb2.cols}, steps = ${pb2.steps.map (_.mod_cols.size)}")
    assert (pf2.cols.toSet == planted2 && pb2.cols.toSet == planted2, "the planted set with the pair")
    assert (pf2.steps.forall (s => pair.subsetOf (s.mod_cols.toSet) || pair.forall (j => ! s.mod_cols.contains (j))),
            "forward never holds half a pair")
    assert (pb2.steps.forall (s => pair.subsetOf (s.mod_cols.toSet) || pair.forall (j => ! s.mod_cols.contains (j))),
            "backward never holds half a pair")

    banner ("a failed evaluation is skipped, not chosen")
    val eval3 = (cols: LSET [Int]) => if cols.contains (7) then BestStep (-1, null, null, cols)() else eval (cols)
    val pf3 = SubsetSearch.forward (n, eval3)
    println (s"forward with column 7 failing: cols = ${pf3.cols}, steps = ${pf3.steps.size}")
    assert (pf3.cols.toSet == planted && pf3.steps.forall (s => ! s.mod_cols.contains (7)), "column 7 never enters")
    println ("subsetSearchTest: passed")

end subsetSearchTest

