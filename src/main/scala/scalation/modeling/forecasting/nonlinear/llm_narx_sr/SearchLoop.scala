
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Mon Sep 14 15:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: The Propose-Score-Feed-Back Loop of `LLM_NARX_SR`
 *
 *  A `SearchLoop` keeps a de-duplicated population of scored formulas (keyed by
 *  `ARX_Formula.key`) and runs rounds of
 *
 *      propose (top-K traces shown) -> [trim] -> score -> add to population
 *
 *  for any `Proposer` (an LLM, a genetic rule, random grammar sampling) -- the
 *  proposals may carry free nonlinear parameters (@see `NARX_Formula`), which the
 *  scorer fits; two proposals that differ only in the start or fitted values of
 *  their parameters are one candidate (`FormulaScorer.keyOf`) --, with an
 *  ADAPTIVE length: the loop stops after `patience` consecutive rounds without
 *  a better best score, when a round yields nothing new twice in a row, or at
 *  `maxRounds`.  Every entry into the population -- a seed, a proposal, the
 *  selection stage's equation -- is recorded as a `Trace` (who proposed what,
 *  in which round, and what the trimmer kept), so the origin of every equation
 *  can be read back.  With a trimmer (@see `SelectionTech`) each legal proposal
 *  f with at least hp("minTerms") terms is scored, trimmed to fTrim by that
 *  selection technique over its own terms -- one CEM round, backward elimination,
 *  forward, stepwise or beam, all on the scorer's criterion (@see
 *  `TermLibrary.selectBy`, `SubsetSearch`; f itself when no proper subset scores
 *  better, so trimming never worsens a candidate) -- and fTrim is what the
 *  population holds and the proposer sees next (with the dropped terms);
 *  validity counts f.
 *  It counts proposals, legal proposals, fit failures and logs the best score per
 *  round, so every arm reports the same budget and convergence columns.  No arm
 *  logic lives here.
 *  @note    AI Assisted Code
 */

package scalation
package modeling
package forecasting
package nonlinear
package llm_narx_sr

import scala.collection.mutable.{ArrayBuffer => VEC, LinkedHashMap => LMAP}

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Proposer` trait generates new formulas, given the best traces so far.
 *  A proposal that could not be turned into a legal formula (a rejected string)
 *  is returned as `Left (reason)`, so validity can be counted.
 */
trait Proposer:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return who this proposer is (recorded in every trace it causes).
     */
    def source: Source

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Propose n new formulas (or rejects with reasons).
     *  @param top  the origin traces of the current best candidates, best first
     *              (each carries the proposed f, the kept fTrim and their scores)
     *  @param n    the number of proposals requested
     */
    def propose (top: Seq [Trace], n: Int): Seq [Either [String, ARX_Formula]]

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Take the round's feedback: the proposals that were legal but failed or were
     *  rejected by the scorer, each with its reason, and diagnostic notes on the
     *  current best (@see `FormulaScorer.notes`); a proposer may ignore it.
     *  @param failed  (formula string, reason) of this round's failed and rejected proposals
     *  @param notes   the diagnostic notes ("" when none)
     */
    def feedback (failed: Seq [(String, String)], notes: String): Unit = ()

end Proposer


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SearchLoop` class runs propose -> [trim] -> score -> feed back over a
 *  de-duplicated population of scored formulas.
 *  @param scorer   the fit-and-score engine
 *  @param topK     the number of best candidates shown to the proposer each round
 *  @param trimmer  the selection technique that trims every proposal (None => full version;
 *                      any `SelectionTech`, @see `trim`)
 *  @param stream   the random number stream of a stochastic trimmer (CEM)
 */
class SearchLoop (scorer: FormulaScorer, topK: Int = FormulaScorer.hp("topK").toInt,
                  trimmer: Option [SelectionTech] = None, stream: Int = 0,
                  maxFits: Int = FormulaScorer.hp("maxFits").toInt):

    private val debug      = debugf ("SearchLoop", true)                // debug function
    private val flaw       = flawf ("SearchLoop")                       // flaw function
    private val scored     = LMAP [String, Candidate] ()                // key -> every formula scored (f and fTrim)
    private val population = LMAP [String, Candidate] ()                // key -> kept candidates
    private val traceLog   = VEC [Trace] ()                             // every entry, in order
    private val convLog    = VEC [ConvRow] ()                           // best-so-far per round
    private val minTerms   = FormulaScorer.hp("minTerms").toInt         // smallest proposal a trimmer touches
    private val maxFree    = FormulaScorer.hp("maxFree").toInt          // free parameters allowed per proposal (-1 => no cap)
    private val fits0      = scorer.evaluations                         // fits made before this loop (the budget counts from here)
    private var bestSoFar  = Double.MaxValue                            // best score seen (seeds included)
    private var nRounds    = 0                                          // rounds completed
    private var _nProposed, _nValid, _nFailed = 0                       // counters (seeds excluded)
    private var roundFailed = VEC [(String, String)] ()                 // this round's failed/rejected proposals with reasons
    private var _error: String = null                                   // the proposer's failure that ended the loop, if any

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the population's best candidates, best (lowest score) first;
     *  failed fits are excluded (they stay in the population so they are not
     *  scored again).
     *  @param k  the number of candidates to return
     */
    def top (k: Int = topK): Seq [Candidate] =
        population.values.toSeq.filterNot (_.failed).sortBy (_.score).take (k)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the origin traces of the best candidates, best first (what a
     *  proposer is shown).
     *  @param k  the number of candidates
     */
    def topTraces (k: Int = topK): Seq [Trace] = top (k).flatMap (originOf)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the best candidate, if any fit succeeded.
     */
    def best: Option [Candidate] = top (1).headOption

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return every kept candidate (fitted ones first, best first).
     */
    def all: Seq [Candidate] = population.values.toSeq.sortBy (_.score)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the first trace that led to a candidate (as f or as fTrim).
     *  @param cand  the candidate
     */
    def originOf (cand: Candidate): Option [Trace] =
        traceLog.find (t => t.candTrim.key == cand.key || t.cand.key == cand.key)
    end originOf

    def traces: Seq [Trace]     = traceLog.toSeq                        // every entry, in order
    def convergence: Seq [ConvRow] = convLog.toSeq                      // the convergence log
    def rounds: Int    = nRounds                                        // rounds completed
    def nProposed: Int = _nProposed                                     // proposals made (seeds excluded)
    def nValid: Int    = _nValid                                        // legal proposals
    def nFailed: Int   = _nFailed                                       // legal candidates whose fit failed
    def size: Int      = population.size                                // distinct candidates kept
    def error: String  = _error                                         // why the proposer stopped the loop (null => it did not)
    def fits: Int      = scorer.evaluations - fits0                     // model fits made since this loop was created

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Seed the population with formulas (e.g., the plain ARX baseline), so the
     *  proposer starts from a meaningful anchor; seeds are not counted as proposals.
     *  @param fs  the seed formulas, each with its source (`Seed`, or `Selected`
     *             for the equation a selection stage handed back)
     */
    def seed (fs: (ARX_Formula, Source)*): Unit =
        for (f, src) <- fs do enter (f, src, 0)
        bestSoFar = best.map (_.score).getOrElse (Double.MaxValue)
    end seed

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Score a formula once (per key) and return its candidate.
     *  @param f  the formula
     */
    private def candidateOf (f: ARX_Formula): Candidate =
        scored.getOrElseUpdate (FormulaScorer.keyOf (f), scorer.score (f))
    end candidateOf

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Trim a formula by a selection technique over its own terms and return
     *  the kept equation: the best-scoring subset the technique explored, or the
     *  formula itself when no proper subset scores better (so trimming never
     *  worsens a candidate -- a CEM round samples subsets and need not see the
     *  full set, backward elimination starts from it) or no subset could be
     *  fitted.  Every technique of `SubsetSearch` is available through
     *  `TermLibrary.selectBy`, all on the scorer's criterion: the CEM trimmer
     *  runs one round from a flat prior, the others their full search.
     *  @param f     the formula
     *  @param tech  the selection technique
     */
    def trim (f: ARX_Formula, tech: SelectionTech): ARX_Formula =
        val lib  = new TermLibrary (f.resp, f.preds, f.tFormT)          // the kept equation gets its own parameters
        val cand = tech match
            case SelectionTech.CEM => lib.selectBy (tech, scorer, stream = stream, maxIter = 1)._1
            case _                 => lib.selectBy (tech, scorer, stream = stream)._1
        if cand.failed then
            flaw ("trim", s"$tech left no fitted equation -- f kept")
            f
        else if cand.score < candidateOf (f).score then cand.formula      // a proper subset scores better
        else f                                                            // nothing to gain: keep the proposal
    end trim

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Enter a formula: score it, trim it when a trimmer is set, record the
     *  trace and keep the (trimmed) candidate in the population unless already
     *  there; return the trace.
     *  @param f       the formula as proposed
     *  @param source  who proposed it
     *  @param round   the round it was proposed in (0 for a seed)
     */
    def enter (f: ARX_Formula, source: Source, round: Int): Trace =
        val cand = candidateOf (f)
        val (tech, kept) = trimmer match
            case Some (t) if ! cand.failed && f.nTerms >= minTerms =>
                val fT = try trim (f, t) catch case e: Exception => { debug ("enter", s"trim failed: ${e.getMessage}"); f }
                if FormulaScorer.keyOf (fT) == FormulaScorer.keyOf (f) then (None, cand) else (Some (t), candidateOf (fT))
            case _ => (None, cand)
        val tr = Trace (round, source, cand, tech, kept)
        if ! population.contains (kept.key) then population (kept.key) = kept
        traceLog += tr
        tr
    end enter

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Score a batch of proposals, updating the population and the counters;
     *  return the number of new (not yet kept) legal candidates.
     *  @param props   the proposals (formulas or rejects)
     *  @param source  who proposed them
     *  @param round   the round they were proposed in
     */
    def evaluate (props: Seq [Either [String, ARX_Formula]], source: Source, round: Int): Int =
        var nNew = 0
        roundFailed = VEC ()
        for p <- props do
            _nProposed += 1
            p match
            case Right (f) if maxFree >= 0 && NARX_Formula (f).nFree > maxFree =>
                val why = s"${NARX_Formula (f).nFree} free parameters (?), at most $maxFree allowed"
                roundFailed += ((f.formulaStr, why))
                debug ("evaluate", s"rejected: ${f.formulaStr} -- $why")
            case Right (f) =>
                _nValid += 1
                val before = population.size
                val tr     = enter (f, source, round)
                if population.size > before then
                    nNew += 1
                    if tr.candTrim.failed then _nFailed += 1
                if tr.candTrim.failed && tr.candTrim.reason != null then roundFailed += ((f.formulaStr, tr.candTrim.reason))
            case Left (reason) => debug ("evaluate", s"rejected: $reason")
        end for
        nNew
    end evaluate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Run rounds of propose -> score until the stop rule fires; return the
     *  number of rounds run in this call.  The loop also stops when the fit budget
     *  (`maxFits` > 0) is spent, and when the proposer fails (an LLM that cannot
     *  be reached): the failure is kept in `error`, the population stays as it is.
     *  After every round the proposer gets the round's feedback (@see `Proposer.feedback`).
     *  @param proposer   the proposal source
     *  @param perRound   the number of proposals requested per round
     *  @param maxRounds  the maximum number of rounds
     *  @param patience   the consecutive rounds without a better best score that stop the loop
     */
    def run (proposer: Proposer, perRound: Int, maxRounds: Int, patience: Int): Int =
        var stagnant, dry, done = 0
        var stop = false
        while ! stop && done < maxRounds do
            val props = try proposer.propose (topTraces (), perRound)
                        catch case e: Exception =>
                            _error = s"${proposer.source} proposer failed in round ${nRounds + 1}: $e"
                            flaw ("run", _error)
                            null
            if props == null then return done
            val nNew = evaluate (props, proposer.source, nRounds + 1)
            done += 1; nRounds += 1
            best match
            case Some (b) =>
                if b.score < bestSoFar then { bestSoFar = b.score; stagnant = 0 } else stagnant += 1
                convLog += ConvRow (nRounds, _nProposed, b.score, b.smape)
                debug ("run", f"round $nRounds%3d: best = ${b.smape}%8.3f (validation ${b.smapeVal}%8.3f, score ${b.score}%8.3f)  " +
                              f"new $nNew  valid ${_nValid}/${_nProposed}  pop ${population.size}  fits $fits")
                proposer.feedback (roundFailed.toSeq, scorer.notes (b))
            case None =>
                stagnant += 1
                convLog += ConvRow (nRounds, _nProposed, -0.0, -0.0)
                proposer.feedback (roundFailed.toSeq, "")
            dry  = if nNew == 0 then dry + 1 else 0
            stop = stagnant >= patience || dry >= 2 || (maxFits > 0 && fits >= maxFits)
        end while
        done
    end run

end SearchLoop


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `searchLoopTest` main function tests the `SearchLoop` class with a
 *  scripted proposer on a series with a planted structure: the population
 *  de-duplicates by key, rejects are counted as invalid, the best improves when
 *  the planted formula is proposed, the loop stops on stagnation (patience) and
 *  when the proposer runs dry, every entry has a trace, and with a trimmer (CEM,
 *  backward) a proposal padded with idle terms is trimmed to a better-scoring
 *  subset of its own terms that keeps the planted structure, its dropped terms
 *  recorded, while a proposal no subset improves is kept as proposed.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.searchLoopTest
 */
@main def searchLoopTest (): Unit =

    import scalation.random.Normal

    val n     = 200
    val burn  = 50                                                      // start-up rows dropped: the series starts at its level
    val noise = Normal (0.0, 1.0, 21)
    val zz    = Normal (0.0, 4.0, 22)
    val uu    = Normal (0.0, 1.0, 23)
    val zAll  = VectorD (for _ <- 0 until n + burn yield 10.0 + zz.gen)
    val yAll  = new VectorD (n + burn)
    yAll(0) = 30.0; yAll(1) = 31.0
    for t <- 2 until n + burn do yAll(t) = 5.0 + 0.7 * yAll(t-1) + 1.5 * zAll(t-2) + noise.gen
    val (y, z) = (yAll(burn until n + burn), zAll(burn until n + burn))
    val u      = VectorD (for _ <- 0 until n yield uu.gen)              // white noise: a variable that explains nothing

    given dataset: Map [String, VectorD] = Map ("y" -> y, "z" -> z, "u" -> u)
    val vars   = dataset.keySet
    val scorer = new FormulaScorer (hh = 2)
    def parse (s: String): Either [String, ARX_Formula] = NARX_FormulaParser.tryParseFormula (s, vars, maxLag = 4)

    /** A proposer that plays a script: round 1 two weak formulas and one reject,
     *  round 2 the planted formula and a duplicate, then nothing new.
     */
    class Scripted extends Proposer:
        private var round = 0
        def source: Source = Source.LLM
        def propose (top: Seq [Trace], k: Int): Seq [Either [String, ARX_Formula]] =
            round += 1
            round match
            case 1 => Seq (parse ("y ~= y~2"), parse ("y ~= z~1"), parse ("y ~= w~1"))
            case 2 => Seq (parse ("y ~= y~1 + z~2"), parse ("y ~= z~2 + y~1"))
            case _ => Seq (parse ("y ~= y~2"))
    end Scripted

    val loop = new SearchLoop (scorer, topK = 3)
    loop.seed ((NARX_FormulaParser.parseFormula ("y ~= y~1", vars, maxLag = 4), Source.Seed))
    val seedScore = loop.best.get.score
    println (s"seed: ${loop.best.get.formulaStr} @ $seedScore")

    val done = loop.run (new Scripted, perRound = 3, maxRounds = 10, patience = 2)
    banner ("after the scripted run")
    println (s"rounds = $done, proposed = ${loop.nProposed}, valid = ${loop.nValid}, failed = ${loop.nFailed}, pop = ${loop.size}")
    println (s"best = ${loop.best.get.formulaStr} @ ${loop.best.get.score}")
    for r <- loop.convergence do println (s"  $r")
    banner ("traces")
    for t <- loop.traces do println (s"  $t")

    assert (done == 4 && loop.nProposed == 7, "4 rounds (3 + 2 + 1 + 1 proposals), then stagnant/dry stop")
    assert (loop.nValid == loop.nProposed - 1, "exactly one reject (unknown variable w)")
    assert (loop.size == 4, "population: seed + y~2 + z~1 + (y~1 + z~2), duplicates collapsed")
    assert (loop.best.get.formula.preds.map (_.label).toSet == Set ("y~1", "z~2"), "the planted formula must win")
    assert (loop.best.get.score < seedScore, "the best must improve over the seed")
    assert (done < 10, "the loop must stop early (patience or dry rounds)")
    assert (loop.traces.size == 1 + loop.nValid, "one trace per seed and per legal proposal")
    assert (loop.all.forall (c => loop.originOf (c).isDefined), "every kept candidate has an origin trace")
    val org = loop.originOf (loop.best.get).get
    assert (org.round == 2 && org.source == Source.LLM, s"the winner was proposed in round 2 by the LLM: $org")
    assert (loop.traces.count (_.key == loop.best.get.key) == 2, "the winner was proposed twice (duplicate collapsed)")

    banner ("trimmers on a padded proposal: y~1 + z~2 + z~1 (the structure at both horizons) + four idle terms")
    // y_t = 5 + 0.7 y_t-1 + 1.5 z_t-2 needs y~1, z~2 at h = 1 and, directly, z~1 as well at h = 2; the lags
    // y~4, z~4 and the white-noise variable u carry no information at either horizon (the series starts at
    // its level, so no start-up transient makes a padded lag useful).  Under the criterion J an idle term
    // costs more than the in-sample gain it can buy, so a trimmer must drop at least one, keep terms of
    // the proposal only, keep the structure, and never return a worse equation.
    val padded = NARX_FormulaParser.parseFormula ("y ~= y~1 + z~2 + z~1 + y~4 + z~4 + u~1 + u~2", vars, maxLag = 4)
    for tech <- Seq (SelectionTech.CEM, SelectionTech.Backward) do
        val loopT = new SearchLoop (scorer, topK = 3, trimmer = Some (tech), stream = 0)
        val tr    = loopT.enter (padded, Source.LLM, 1)
        println (s"  $tr")
        assert (tr.trimmer.contains (tech), s"the trace must name the trimmer $tech")
        assert (tr.fTrim.nTerms < tr.f.nTerms, s"$tech must drop an idle term")
        assert (tr.fTrim.preds.forall (t => tr.f.preds.exists (_.label == t.label)), "kept terms come from the proposal")
        assert (Seq ("y~1", "z~2", "z~1").forall (l => tr.fTrim.preds.exists (_.label == l)), "the structure stays")
        assert (tr.candTrim.score < tr.cand.score, "the kept equation scores better than the proposal")
        assert (loopT.size == 1 && loopT.best.get.key == tr.key, "the population holds the kept equation")
    end for

    banner ("a proposal no subset improves is kept as proposed")
    val loopK = new SearchLoop (scorer, topK = 3, trimmer = Some (SelectionTech.Backward), stream = 0)
    val trK   = loopK.enter (NARX_FormulaParser.parseFormula ("y ~= y~1 + z~2 + z~1", vars, maxLag = 4), Source.LLM, 1)
    println (s"  $trK")
    assert (trK.candTrim.score <= trK.cand.score, "trimming never worsens a candidate")
    assert (trK.trimmer.isEmpty == (trK.fTrim.key == trK.f.key), "the trace names the trimmer iff something was dropped")
    println ("searchLoopTest: passed")

end searchLoopTest
