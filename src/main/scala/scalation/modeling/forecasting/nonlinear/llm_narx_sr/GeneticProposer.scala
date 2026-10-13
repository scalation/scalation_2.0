
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Mon Sep 14 15:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Random Terms and the Genetic Proposer (no LLM) for `LLM_NARX_SR`
 *
 *  `RandomTerms` draws terms and formulas from the grammar with ScalaTion's random
 *  streams (@see `scalation.random.Random`), so a run is reproduced by its
 *  stream number -- with `nonlinear` the free-parameter forms as well (pow(x~k, ?),
 *  log(? + x~k), exp(? * x~k), sat, hinge, trans, ema, dq, wave(?), @see
 *  `NARX_FormulaParser`), each draw with its own parameter.  `GeneticProposer` is the
 *  zero-LLM update rule: offspring of the population's best formulas by crossover
 *  (union split) and mutation (add, drop, replace a term, shift a lag, toggle the
 *  log1p transform); an offspring gets its own copies of the inherited parameters
 *  (`NARX_Formula.fresh`), so fitting it never touches a parent.  With
 *  `random = true` it ignores the population and samples the grammar uniformly
 *  (the no-feedback control).  Both emit `NARX_Formula`s directly -- there is
 *  nothing to parse -- and never a reject.
 *  @note    AI Assisted Code
 */

package scalation
package modeling
package forecasting
package nonlinear
package llm_narx_sr

import scala.collection.mutable.{ArrayBuffer => VEC}

import scalation.mathstat._
import scalation.random.Random

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RandomTerms` class draws terms and formulas from the grammar.
 *  @param resp    the response variable name
 *  @param exo     the exogenous variable names
 *  @param maxLag  the maximum lag
 *  @param freqs   the wave frequencies (null or empty => no wave terms)
 *  @param stream  the random number stream
 *  @param funcs      the function names a term may be wrapped in
 *  @param powers     the exponents a term may be raised to
 *  @param nonlinear  whether to draw the free-parameter forms as well (defaults to hp("nlTerms") = 1)
 */
class RandomTerms (resp: String, exo: Seq [String], maxLag: Int, freqs: VectorD, stream: Int,
                   funcs: Seq [String] = Seq ("log1p", "sq"), powers: Seq [Double] = Seq (0.5, 1.5, 2.0),
                   nonlinear: Boolean = FormulaScorer.hp("nlTerms").toInt == 1):

    private val rng   = Random (stream)                                 // uniform (0, 1) generator
    private val nFreq = if freqs == null then 0 else freqs.dim          // wave pairs available

    /** The free-parameter forms a draw may produce (each with a new parameter).
     */
    val NL_FORMS: Seq [Term => Seq [Term]] = Seq (
        (t: Term) => Seq (pow (t, power ())),
        (t: Term) => Seq (log (shift () + t)),
        (t: Term) => Seq (exp (rate () * t)),
        (t: Term) => Seq (sat (t, half ())),
        (t: Term) => Seq (hinge (t, thresh ())),
        (t: Term) => Seq (trans (t, thresh (), width ())),
        (t: Term) => Seq (ema (t, decay ())),
        (t: Term) => Seq (dq (t, cap ())),
        (_: Term) => wave (freq ()))                                    // a free-frequency pair (no source term)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a uniform random integer in 0 until n.
     *  @param n  the number of values
     */
    def randInt (n: Int): Int = math.min (n - 1, (rng.gen * n).toInt)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a uniform random number in (0, 1).
     */
    def uniform: Double = rng.gen

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Draw a lagged variable: the response with probability 1/2, else an
     *  exogenous variable, at a lag in 1..maxLag.
     */
    def atom (): FormulaVar =
        val v = if exo.isEmpty || uniform < 0.5 then resp else exo(randInt (exo.size))
        FormulaVar (v, 1 + randInt (maxLag))
    end atom

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Draw one term: a lagged variable (1/2), a function of one (1/5), a power of
     *  one (1/10), a product of two (1/10) or a wave pair's member (1/10, when
     *  frequencies exist; both members are returned so the pair stays whole); with
     *  `nonlinear`, 15% of the lagged-variable draws become a free-parameter form
     *  (`NL_FORMS`) of the variable instead.
     */
    def term (): Seq [Term] =
        val u = uniform
        if u < 0.5 then
            if nonlinear && u < 0.15 then NL_FORMS (randInt (NL_FORMS.size))(atom ())
            else Seq (atom ())
        else if u < 0.7 then Seq (ARX_Formula.functions (funcs(randInt (funcs.size)))(atom ()))
        else if u < 0.8 then Seq (pow (atom (), powers(randInt (powers.size))))
        else if u < 0.9 || nFreq == 0 then Seq (ProductTerm (atom (), atom ()))
        else wave (freqs, 1 + randInt (nFreq))                          // pairs 1..k: the grammar's wave(k)
    end term

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Normalize the wave terms of a term list to the grammar's wave(k): when any
     *  wave term is present, the pairs 1..k (k the largest pair index present)
     *  replace them, so the formula string stays parseable (@see `ARX_Formula.formulaStr`).
     *  @param preds  the terms
     */
    def normalizeWaves (preds: VEC [Term]): VEC [Term] =
        val (waves, rest) = preds.partition (_.isInstanceOf [WaveTerm])
        if waves.isEmpty || nFreq == 0 then rest
        else
            val k = math.min (nFreq, waves.map (_.asInstanceOf [WaveTerm].idx).max)
            rest ++ wave (freqs, k)
    end normalizeWaves

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Draw a formula with about n terms (duplicates collapsed) and, with
     *  probability pTForm, the log1p response transform.
     *  @param n       the number of term draws
     *  @param pTForm  the probability of a TFORM: Log1p directive
     */
    def formula (n: Int, pTForm: Double = 0.2): ARX_Formula =
        val preds = VEC [Term] ()
        for _ <- 1 to n do preds ++= term ()
        NARX_Formula (FormulaVar (resp), normalizeWaves (preds), if uniform < pTForm then TransformT.Log1p else null)
    end formula

end RandomTerms


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `GeneticProposer` class proposes offspring of the best formulas by
 *  crossover and mutation (or random grammar samples when `random`).
 *  @param resp      the response variable name
 *  @param exo       the exogenous variable names
 *  @param maxLag    the maximum lag
 *  @param freqs     the wave frequencies (null => none)
 *  @param stream    the random number stream
 *  @param maxTerms  the maximum number of terms per offspring
 *  @param random     whether to ignore the population and sample the grammar
 *  @param pCross     the probability that an offspring is a crossover of two parents
 *  @param nonlinear  whether the draws include the free-parameter forms (defaults to hp("nlTerms") = 1)
 */
class GeneticProposer (resp: String, exo: Seq [String], maxLag: Int, freqs: VectorD, stream: Int,
                       maxTerms: Int = 12, random: Boolean = false, pCross: Double = 0.5,
                       nonlinear: Boolean = FormulaScorer.hp("nlTerms").toInt == 1)
      extends Proposer:

    protected val rt = new RandomTerms (resp, exo, maxLag, freqs, stream, nonlinear = nonlinear)

    def source: Source = if random then Source.Random else Source.Genetic

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Propose n formulas: random grammar samples when `random` or the population
     *  is empty, else offspring of parents drawn rank-biased from the top (the
     *  KEPT equations of the traces, i.e., after any trimming).
     *  @param top  the origin traces of the current best candidates, best first
     *  @param n    the number of proposals requested
     */
    def propose (top: Seq [Trace], n: Int): Seq [Either [String, ARX_Formula]] =
        for _ <- 1 to n yield
            val f = if random || top.isEmpty then rt.formula (2 + rt.randInt (5)) else offspring (top)
            Right (cap (f))
    end propose

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Breed one offspring: a parent (crossed with a second parent with
     *  probability pCross), then mutated.
     *  @param top  the origin traces of the current best candidates, best first
     */
    protected def offspring (top: Seq [Trace]): ARX_Formula =
        val a     = parent (top)
        val child = if top.size > 1 && rt.uniform < pCross then crossover (a, parent (top)) else a
        mutate (child)
    end offspring

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Draw a parent from the top, rank-biased (the best twice as likely as the
     *  second, and so on down: index = floor of a squared uniform).
     *  @param top  the origin traces of the current best candidates, best first
     */
    protected def parent (top: Seq [Trace]): ARX_Formula =
        val u = rt.uniform
        top(math.min (top.size - 1, (u * u * top.size).toInt)).fTrim
    end parent

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Cross two formulas: a random split of the union of their terms (each term
     *  kept with probability 1/2, at least one kept); the transform of parent a.
     *  @param a  the first parent
     *  @param b  the second parent
     */
    protected def crossover (a: ARX_Formula, b: ARX_Formula): ARX_Formula =
        val union = NARX_Formula.distinct (a.preds ++ b.preds)
        val kept  = union.filter (_ => rt.uniform < 0.5)
        val preds = if kept.isEmpty then VEC (union(rt.randInt (union.size))) else kept
        NARX_Formula (a.resp, NARX_Formula.fresh (preds), a.tFormT)   // own copies of the inherited parameters
    end crossover

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Mutate a formula by one random structural move: add a term (1/4), drop a
     *  term (1/4), replace a term (1/5), shift the lag of a lagged variable (1/5)
     *  or toggle the log1p transform (1/10).
     *  @param f  the formula to mutate
     */
    protected def mutate (f: ARX_Formula): ARX_Formula =
        val preds = NARX_Formula.fresh (f.preds)                        // own copies of the parameters
        val u     = rt.uniform
        if u < 0.25 || (preds.size <= 1 && u < 0.5) then preds ++= rt.term ()
        else if u < 0.5 then drop (preds)
        else if u < 0.7 then
            drop (preds); preds ++= rt.term ()
        else if u < 0.9 then
            val j = rt.randInt (preds.size)
            preds(j) = shift (preds(j))
        else
            return NARX_Formula (f.resp, preds, if f.tFormT == null then TransformT.Log1p else null)
        NARX_Formula (f.resp, rt.normalizeWaves (preds), f.tFormT)
    end mutate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Drop one random term; dropping a wave member drops its pair and every
     *  higher pair (wave(k) becomes wave(k-1)).
     *  @param preds  the terms (modified in place)
     */
    private def drop (preds: VEC [Term]): Unit =
        preds(rt.randInt (preds.size)) match
        case w: WaveTerm    => preds.filterInPlace (t => ! (t.isInstanceOf [WaveTerm] && t.asInstanceOf [WaveTerm].idx >= w.idx))
        case w: ParWaveTerm => preds.filterInPlace (t => ! (t.isInstanceOf [ParWaveTerm] && (t.asInstanceOf [ParWaveTerm].freq eq w.freq)))
        case t              => preds -= t
    end drop

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Shift the lag of a lagged variable by one step (within 1..maxLag), through
     *  a function, power or product wrapper; other terms are returned unchanged.
     *  @param t  the term
     */
    protected def shift (t: Term): Term =
        t match
        case FormulaVar (nm, l) =>
            val l2 = if l >= maxLag then l - 1
                     else if l <= 1 then l + 1
                     else if rt.uniform < 0.5 then l - 1 else l + 1
            FormulaVar (nm, l2)
        case TransformedTerm (s, fn, op) => TransformedTerm (shift (s), fn, op)
        case PowerTerm (s, p, safe)      => PowerTerm (shift (s), p, safe)
        case ProductTerm (a, b)          => if rt.uniform < 0.5 then ProductTerm (shift (a), b) else ProductTerm (a, shift (b))
        case ShiftTerm (s, p)            => ShiftTerm (shift (s), p)    // the free-parameter forms: shift the source
        case ScaleTerm (s, a)            => ScaleTerm (shift (s), a)
        case ParPowerTerm (s, p)         => ParPowerTerm (shift (s), p)
        case SatTerm (s, k)              => SatTerm (shift (s), k)
        case HingeTerm (s, c)            => HingeTerm (shift (s), c)
        case TransTerm (s, c, w)         => TransTerm (shift (s), c, w)
        case EmaTerm (s, d)              => EmaTerm (shift (s), d)
        case DqTerm (s, mx)              => DqTerm (shift (s), mx)
        case other                       => other
    end shift

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Cap the number of terms of a formula (random drops beyond maxTerms) and
     *  make sure at least one term remains.
     *  @param f  the formula
     */
    private def cap (f: ARX_Formula): ARX_Formula =
        val preds = f.preds.clone ()
        while preds.size > maxTerms do drop (preds)
        val maxFree = FormulaScorer.hp("maxFree").toInt
        def nFree (ts: VEC [Term]): Int = NARX_Formula.params (ts).count (_.free)
        while maxFree >= 0 && nFree (preds) > maxFree do
            val js = preds.indices.filter (j => NARX_Formula.params (Seq (preds(j))).exists (_.free))
            val j  = js(rt.randInt (js.size))
            preds(j) match
                case p: ParTerm => preds(j) = p.source                    // the plain term under the parameter
                case w: ParWaveTerm => preds.filterInPlace (t => ! (t.isInstanceOf [ParWaveTerm] && (t.asInstanceOf [ParWaveTerm].freq eq w.freq)))
                case _ => preds.remove (j)
        end while
        if preds.isEmpty then preds ++= rt.term ()
        NARX_Formula (f.resp, rt.normalizeWaves (NARX_Formula.distinct (preds)), f.tFormT)
    end cap

end GeneticProposer


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `geneticProposerTest` main function tests `RandomTerms` and
 *  `GeneticProposer`: every proposal is a legal formula within the grammar
 *  (variables, lags, term cap), the same stream reproduces the same proposals,
 *  different streams differ, and offspring of a population keep its terms in play.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.geneticProposerTest
 */
@main def geneticProposerTest (): Unit =

    val vars   = Set ("y", "hosp", "icu")
    val freqs  = VectorD (0.13043, 0.06521)
    val maxLag = 6

    banner ("random grammar sampling: legal, within the lag bound, reproducible")
    val gp  = new GeneticProposer ("y", Seq ("hosp", "icu"), maxLag, freqs, stream = 0, random = true)
    val ps  = gp.propose (Seq (), 30).map (_.toOption.get)
    for f <- ps.take (8) do println (s"  ${f.formulaStr}")
    for f <- ps do
        assert (f.preds.nonEmpty && f.preds.size <= 12, s"term count out of range: ${f.formulaStr}")
        assert (f.preds.forall (t => t.lag <= maxLag && t.vars.subsetOf (vars)), s"illegal term in ${f.formulaStr}")
        val back = NARX_FormulaParser.tryParseFormula (f.formulaStr, vars, maxLag, freqs)
        assert (back.isRight && back.toOption.get.key == f.key, s"formula string does not round-trip: ${f.formulaStr}")
    val nNL = ps.count (f => NARX_Formula (f).nFree > 0)
    println (s"proposals with free parameters: $nNL of ${ps.size}")
    assert (nNL > 0, "random sampling must draw free-parameter forms (hp nlTerms = 1)")
    val again = new GeneticProposer ("y", Seq ("hosp", "icu"), maxLag, freqs, stream = 0, random = true)
                    .propose (Seq (), 30).map (_.toOption.get)
    assert (ps.map (_.key) == again.map (_.key), "the same stream must reproduce the same proposals")
    val other = new GeneticProposer ("y", Seq ("hosp", "icu"), maxLag, freqs, stream = 1, random = true)
                    .propose (Seq (), 30).map (_.toOption.get)
    assert (ps.map (_.key) != other.map (_.key), "a different stream must give different proposals")

    banner ("offspring of a population")
    val f1  = NARX_FormulaParser.parseFormula ("y ~= y~1 + hosp~2", vars, maxLag, freqs)
    val f2  = NARX_FormulaParser.parseFormula ("y ~= y~(2,1) + log1p(icu~1) + wave(1)", vars, maxLag, freqs)
    val c1  = Candidate (f1, 10.0, MatrixD.fill (1, Fit.N_QoF, 10.0), 2)
    val c2  = Candidate (f2, 12.0, MatrixD.fill (1, Fit.N_QoF, 12.0), 5)
    val top = Seq (Trace (0, Source.Seed, c1, None, c1), Trace (0, Source.Seed, c2, None, c2))
    val kids = new GeneticProposer ("y", Seq ("hosp", "icu"), maxLag, freqs, stream = 2).propose (top, 20).map (_.toOption.get)
    for k <- kids.take (8) do println (s"  ${k.formulaStr}")
    val parentTerms = (f1.preds ++ f2.preds).map (_.label).toSet
    val inherited   = kids.count (k => k.preds.exists (t => parentTerms.contains (t.label)))
    println (s"offspring carrying a parent term: $inherited of ${kids.size}")
    assert (inherited >= kids.size / 2, "most offspring must inherit terms from the parents")
    assert (kids.forall (k => k.preds.nonEmpty && k.preds.size <= 12), "offspring within the term cap")

    banner ("an offspring never shares a parameter object with its parent")
    val f3   = NARX_FormulaParser.parseFormula ("y ~= y~1 + pow(hosp~1, ?) + log(? + icu~1)", vars, maxLag, freqs)
    val c3   = Candidate (f3, 9.0, MatrixD.fill (1, Fit.N_QoF, 9.0), 3)
    val kid3 = new GeneticProposer ("y", Seq ("hosp", "icu"), maxLag, freqs, stream = 3)
                   .propose (Seq (Trace (0, Source.Seed, c3, None, c3)), 10).map (_.toOption.get)
    val parentPars = NARX_Formula (f3).params
    for k <- kid3; p <- NARX_Formula (k).params do
        assert (! parentPars.exists (_ eq p), s"shared parameter object in ${k.formulaStr}")
    println (s"${kid3.count (k => NARX_Formula (k).nFree > 0)} of ${kid3.size} offspring carry free parameters, all their own")
    println ("geneticProposerTest: passed")

end geneticProposerTest
