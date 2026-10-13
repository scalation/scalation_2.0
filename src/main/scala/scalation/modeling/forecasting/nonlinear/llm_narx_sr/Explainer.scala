
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Tue Sep 15 14:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Grounded Explanations of `LLM_NARX_SR` Equations with an
 *           Automatic Faithfulness Check
 *
 *  A fitted equation is handed to an LLM with its coefficients and standardized
 *  effects (coefficient x standard deviation of the term's column on the training
 *  rows, h = 1) and the LLM writes an explanation for a domain reader.  The reply
 *  must begin with one machine-readable CLAIM line per term (sign of the effect,
 *  lag, importance rank) before the narrative, and every claim is checked against
 *  the fitted model: the share of claims that are right is the explanation's
 *  faithfulness, the share of terms claimed is its coverage.  Domain expectations
 *  (@see `DatasetConfig.expectedSigns`) that the fitted signs contradict are put
 *  in the prompt, so the narrative has to address them rather than paper over them.
 *  `Explainer.importance` (no LLM) gives the term-importance table that
 *  `LLM_NARX_SR.showBest` prints under the coefficient tables.
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
/** The `TermEffect` case class is one term of a fitted equation at h = 1.
 *  @param label   the term label (real names)
 *  @param lag     the term's lag
 *  @param coef    the coefficient in original units
 *  @param effect  the standardized effect: coef x std of the term's column (training rows)
 */
case class TermEffect (label: String, lag: Int, coef: Double, effect: Double):

    def sign: Int = if effect < 0.0 then -1 else 1

end TermEffect


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Claim` case class is what the explanation says about one term, checked.
 *  @param term    the term label (real names)
 *  @param sign    the claimed sign of the effect (+1/-1)
 *  @param lag     the claimed lag
 *  @param rank    the claimed importance rank (1 = largest |effect|)
 *  @param signOK  whether the sign agrees with the fitted effect
 *  @param lagOK   whether the lag is the term's lag
 *  @param rankOK  whether the rank is the term's rank by |effect|
 */
case class Claim (term: String, sign: Int, lag: Int, rank: Int, signOK: Boolean, lagOK: Boolean, rankOK: Boolean):

    def ok: Boolean = signOK && lagOK && rankOK

    override def toString: String =
        f"$term%-24s sign ${if sign > 0 then "+" else "-"} ${mark (signOK)}  lag $lag%2d ${mark (lagOK)}  rank $rank%2d ${mark (rankOK)}"

    private def mark (b: Boolean): String = if b then "ok " else "BAD"

end Claim


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Explanation` case class is the LLM's narrative with its checked claims.
 *  @param text    the narrative for the reader
 *  @param claims  the claims found in the reply, matched to terms and checked
 *  @param nTerms  the number of terms of the equation
 */
case class Explanation (text: String, claims: Seq [Claim], nTerms: Int):

    def coverage: Double     = if nTerms == 0 then -0.0 else claims.size.toDouble / nTerms
    def faithfulness: Double = rate (_.ok)                              // claims right in every respect
    def signRate: Double     = rate (_.signOK)
    def lagRate: Double      = rate (_.lagOK)
    def rankRate: Double     = rate (_.rankOK)

    private def rate (p: Claim => Boolean): Double = if claims.isEmpty then -0.0 else claims.count (p).toDouble / claims.size

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return one formatted line for a table of explanations.
     */
    def tableRow: String =
        f"coverage ${coverage}%5.2f  faithful ${faithfulness}%5.2f  (sign ${signRate}%4.2f  lag ${lagRate}%4.2f  rank ${rankRate}%4.2f)  claims ${claims.size}%2d / $nTerms%2d terms"
    end tableRow

end Explanation


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Explainer` class asks an LLM for a grounded explanation of a fitted
 *  equation and checks its claims against the model.
 *  @param llm     the completion client
 *  @param cfg     the dataset configuration (names, domain sentence, expected signs)
 *  @param named   whether the real variable names and the domain sentence are shown
 *                     (the search is over, so naming leaks nothing; blind for the ablation)
 */
class Explainer (llm: LLMClient, cfg: DatasetConfig, named: Boolean = true):

    private val debug   = debugf ("Explainer", false)                   // debug function (prints raw replies)
    private val toShown = if named then Map [String, String] () else cfg.blindNames   // real -> shown name
    private val toReal  = toShown.map (_.swap)                          // shown -> real name
    private val ident   = "[A-Za-z_][A-Za-z0-9_]*".r                   // identifier pattern
    private val claimRx = """(?i)TERM:\s*(.+?)\s*\|\s*SIGN:\s*([+-])\s*\|\s*LAG:\s*(\d+)\s*\|\s*RANK:\s*(\d+)""".r
    private var _calls  = 0                                             // prompts sent
    private var _last   = ""                                            // the last user prompt sent

    private val rn = shown (cfg.resp)                                   // the response as the LLM sees it

    /** The system prompt: the task, the model form and the reply format.
     */
    val system: String =
        val about = if named then cfg.schema else LLMProposer.genericSchema (cfg.exo.size, cfg.freq)
        s"""You explain a fitted symbolic time-series model to a domain reader.
           |$about
           |The model forecasts $rn one step ahead as an intercept plus a weighted sum of terms; VAR~k is the value
           |of VAR k steps back, so a term with lag k is a delayed effect of k steps.  The standardized effect of
           |a term (coefficient x standard deviation of the term's values) says how much the term moves the
           |forecast; its sign says the direction.  Rank the terms by |standardized effect|, rank 1 the largest.
           |Reply in exactly this form and nothing else:
           |TERM: <term as written> | SIGN: <+ or -> | LAG: <k> | RANK: <r>
           |   (one such line per term, all terms, in any order)
           |EXPLANATION: <one paragraph in plain language: what carries the forecast, which effects are delayed and
           |by how many steps, what each exogenous variable does and in which direction, and, where the fitted
           |sign contradicts domain expectation, say so and offer a reason>
           |""".stripMargin

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of prompts sent so far.
     */
    def calls: Int = _calls

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the last user prompt sent (for inspection and the paper's appendix).
     */
    def lastPrompt: String = _last

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the user prompt for a fitted equation: the equation with its h = 1
     *  coefficients, the term table (lag, coefficient, standardized effect) and
     *  any conflict between a fitted net effect and the expected sign.
     *  @param f          the formula
     *  @param effects    the term effects (@see `Explainer.importance`)
     *  @param intercept  the h = 1 intercept
     */
    def prompt (f: ARX_Formula, effects: Seq [TermEffect], intercept: Double): String =
        val eq   = effects.map (e => f"${e.coef}%+.4g*${toShownText (e.label)}").mkString (" ")
        val rows = effects.map (e => f"${toShownText (e.label)}%-28s ${e.lag}%3d   ${e.coef}%+12.4g   ${e.effect}%+12.4g").mkString ("\n")
        val net  = effects.filter (e => e.label.contains ("~") && ! e.label.contains ("*"))   // single-variable terms
                          .groupBy (e => variableOf (e.label)).map ((v, es) => v -> es.map (_.effect).sum)
        val conflicts = cfg.expectedSigns.toSeq.flatMap ((v, s) =>
            net.get (v).filter (e => math.signum (e) != s.toDouble)
               .map (e => s"Domain expectation: ${shown (v)} should have a ${if s > 0 then "positive" else "negative"} net effect; " +
                          f"the fitted net effect is ${e}%+.4g (${if e < 0 then "negative" else "positive"})."))
        s"""Formula: ${toShownText (f.formulaStr)}
           |Fitted model (h = 1):  $rn = ${f"$intercept%.4g"} $eq
           |
           |term                         lag    coefficient   standardized effect
           |$rows
           |${if conflicts.isEmpty then "" else conflicts.mkString ("\n") + "\n"}""".stripMargin
    end prompt

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Parse a reply: the CLAIM lines matched to the terms (by label, shown names
     *  mapped back) and checked, and the narrative after EXPLANATION:.
     *  @param reply    the LLM's reply
     *  @param effects  the term effects the claims are checked against
     */
    def parse (reply: String, effects: Seq [TermEffect]): Explanation =
        val rankOf = effects.sortBy (e => -math.abs (e.effect)).map (_.label).zipWithIndex.map ((l, i) => l -> (i + 1)).toMap
        val byLbl  = effects.map (e => norm (e.label) -> e).toMap
        val claims = VEC [Claim] ()
        for m <- claimRx.findAllMatchIn (reply) do
            val lbl = norm (toRealText (m.group (1)))
            byLbl.get (lbl) match
            case Some (e) =>
                val sign = if m.group (2) == "+" then 1 else -1
                val lag  = m.group (3).toInt
                val rank = m.group (4).toInt
                if ! claims.exists (_.term == e.label) then
                    claims += Claim (e.label, sign, lag, rank, sign == e.sign, lag == e.lag, rank == rankOf (e.label))
            case None => debug ("parse", s"claim for an unknown term ignored: ${m.group (1)}")
        end for
        val i    = reply.toUpperCase.indexOf ("EXPLANATION:")
        val text = if i < 0 then reply.trim else reply.substring (i + "EXPLANATION:".length).trim
        Explanation (toRealText (text), claims.toSeq, effects.size)
    end parse

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Explain a scored equation: refit it through the scorer (training rows),
     *  read its effects, prompt the LLM and check the reply.  A base model (no
     *  formula) gets a note instead.
     *  @param cand    the scored equation
     *  @param scorer  the fit-and-score engine that fitted it
     */
    def explain (cand: Candidate, scorer: FormulaScorer): Explanation =
        val f = cand.formula
        if f == null then return Explanation (s"${cand.formulaStr}: a base model, not explained", Seq (), 0)
        val mod = scorer.fit (f)
        val eff = Explainer.importance (mod, f, scorer.trSize)
        _last   = prompt (f, eff, mod.intercept(0))
        _calls += 1
        val reply = llm.complete (system, _last)
        debug ("explain", s"raw reply:\n$reply")
        parse (reply, eff)
    end explain

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the shown name of a variable (blind unless `named`).
     *  @param v  the real variable name
     */
    private def shown (v: String): String = toShown.getOrElse (v, v)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Map the real names in a text to the shown names.
     *  @param s  the text with real names
     */
    private def toShownText (s: String): String =
        if named then s else ident.replaceAllIn (s, m => toShown.getOrElse (m.matched, m.matched))
    end toShownText

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Map the shown names in a text back to the real names.
     *  @param s  the text with shown names
     */
    private def toRealText (s: String): String =
        if named then s else ident.replaceAllIn (s, m => toReal.getOrElse (m.matched, m.matched))
    end toRealText

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a term label without spaces, for matching.
     *  @param label  the term label
     */
    private def norm (label: String): String = label.replaceAll ("\\s+", "")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the variable of a single-variable term label, e.g., log1p(icu~2) -> icu.
     *  @param label  the term label
     */
    private def variableOf (label: String): String =
        ident.findFirstIn (label.replaceAll ("^(log1p|log|sqrt|sq|dq|exp|expm1|sin|cos|tanh|pow|powr|sat|hinge|trans|ema)\\(", "")
                                .replaceAll ("^\\??[-+.0-9eE]*\\s*[+*]\\s*", "")).getOrElse (label)   // a shift/scale before the variable
    end variableOf

end Explainer


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Explainer` companion object computes the term effects of a fitted model
 *  (no LLM needed) and prints them as the importance table.
 */
object Explainer:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the effect of every term of a fitted formula model at h = 1: the
     *  coefficient in original units and the standardized effect, coefficient x
     *  standard deviation of the term's column on the training rows.
     *  @param mod  the fitted model of the formula
     *  @param f    the formula (its terms match the model's columns in order)
     *  @param tr   the number of training rows
     */
    def importance (mod: ARX_D, f: ARX_Formula, tr: Int): Seq [TermEffect] =
        val b = mod.slopes(?, 0)                                        // h = 1 slopes in original units
        val x = mod.getX
        for (t, j) <- f.preds.toSeq.zipWithIndex yield
            TermEffect (t.label, t.lag, b(j), b(j) * x(0 until tr, j).stdev)
    end importance

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the importance table of a fitted model: terms by decreasing
     *  |standardized effect| with their share of the total.
     *  @param effects  the term effects (@see `importance`)
     */
    def importanceTable (effects: Seq [TermEffect]): String =
        val total = effects.map (e => math.abs (e.effect)).sum
        val hdr   = f"${"term"}%-28s ${"lag"}%3s   ${"coefficient"}%12s   ${"std. effect"}%12s   ${"share"}%6s"
        val rows  = effects.sortBy (e => -math.abs (e.effect)).map (e =>
            f"${e.label}%-28s ${e.lag}%3d   ${e.coef}%+12.4g   ${e.effect}%+12.4g   ${if total > 0 then 100 * math.abs (e.effect) / total else 0.0}%5.1f%%")
        (hdr +: rows).mkString ("\n")
    end importanceTable

end Explainer


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `explainerTest` main function tests the `Explainer` class with a
 *  `MockLLMClient` on the Covid-19 (icu, pRate) data: the importance table of a
 *  fitted formula, the prompt (with a domain-sign conflict line when the fitted
 *  sign disagrees), and the checking of a reply that gets one sign and two ranks
 *  wrong and skips one term -- coverage, sign/lag/rank rates and faithfulness
 *  must come out exactly; then a blind explainer maps x1/x2 back to the names.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.explainerTest
 */
@main def explainerTest (): Unit =

    val cfg = Datasets.COVID2
    given dataset: Map [String, VectorD] = cfg.load ()
    val scorer = new FormulaScorer (6)
    val f      = NARX_FormulaParser.parseFormula ("y ~= y~(2,1) + icu~1 + log1p(pRate~3)", scorer.vars, maxLag = 6)
    val cand   = scorer.score (f)
    val mod    = scorer.fit (f)
    val eff    = Explainer.importance (mod, f, scorer.trSize)

    banner ("term importance at h = 1")
    println (Explainer.importanceTable (eff))
    assert (eff.size == 4 && eff.map (_.label) == Seq ("y~2", "y~1", "icu~1", "log1p(pRate~3)"), "one effect per term, in order")
    assert (eff.forall (e => e.effect == e.coef * mod.getX(0 until scorer.trSize, eff.indexOf (e)).stdev), "effect = coef x std")

    // a reply from a careful reader: every claim right, then two mistakes and a missing term
    val rankOf = eff.sortBy (e => -math.abs (e.effect)).map (_.label).zipWithIndex.toMap
    def line (e: TermEffect, sign: Int = 0, rank: Int = 0): String =
        val s = if sign != 0 then sign else e.sign
        val r = if rank != 0 then rank else rankOf (e.label) + 1
        s"TERM: ${e.label} | SIGN: ${if s > 0 then "+" else "-"} | LAG: ${e.lag} | RANK: $r"
    val good = eff.map (e => line (e)).mkString ("\n") +
               "\nEXPLANATION: Last week's deaths carry the forecast; icu patients a week earlier add to it."
    val r1   = rankOf (eff(0).label) + 1
    val r2   = rankOf (eff(1).label) + 1
    val bad  = Seq (line (eff(0), rank = r2), line (eff(1), rank = r1),          // two ranks swapped
                    line (eff(2), sign = -eff(2).sign)).mkString ("\n") +         // one sign wrong, one term missing
               "\nEXPLANATION: a narrative."

    banner ("a faithful reply")
    val ex1 = new Explainer (new MockLLMClient (VEC (good)), cfg)
    val e1  = ex1.explain (cand, scorer)
    println (ex1.lastPrompt)
    for c <- e1.claims do println (c)
    println (e1.tableRow)
    println (s"narrative: ${e1.text}")
    assert (e1.claims.size == 4 && e1.coverage == 1.0 && e1.faithfulness == 1.0, "every claim right")
    assert (e1.text.startsWith ("Last week"), "the narrative follows EXPLANATION:")

    banner ("a reply with two ranks swapped, one wrong sign and a missing term")
    val ex2 = new Explainer (new MockLLMClient (VEC (bad)), cfg)
    val e2  = ex2.explain (cand, scorer)
    for c <- e2.claims do println (c)
    println (e2.tableRow)
    assert (e2.claims.size == 3 && e2.coverage == 0.75, "three of four terms claimed")
    assert (e2.signRate == 2.0 / 3 && e2.lagRate == 1.0 && e2.rankRate == 1.0 / 3, "sign 2/3, lag 3/3, rank 1/3")
    assert (e2.faithfulness == 0.0, "no claim is right in every respect")

    banner ("blind names: the LLM sees x1, x2 and the claims are mapped back")
    val ex3   = new Explainer (new MockLLMClient (VEC (good.replace ("icu", "x1").replace ("pRate", "x2"))), cfg, named = false)
    val e3    = ex3.explain (cand, scorer)
    println (ex3.lastPrompt)
    assert (! ex3.lastPrompt.contains ("icu") && ex3.lastPrompt.contains ("x1~1"), "the blind prompt shows x1")
    assert (e3.claims.size == 4 && e3.faithfulness == 1.0, "claims in blind names match the real terms")
    assert (e3.text.contains ("icu"), "the narrative is mapped back to real names")
    println ("explainerTest: passed")

end explainerTest

