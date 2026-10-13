
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// YousefChange --------- Note: comment only -- the scaladoc below no longer names the sub-package built on it
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Sun Oct 04 15:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Parser for `NARX_Formula` Specification Strings (free parameters)
 *
 *  Turns formula strings with FREE NONLINEAR PARAMETERS into `NARX_Formula` objects,
 *  so that a structure search can propose, parse, fit (`NARX_SR_D.fromTerms`) and score
 *  nonlinear terms as it does linear ones.  The grammar is `FormulaParser`'s plus the
 *  slot ? (optionally with a start value, ?w0), which `NARX_Formula` prints
 *  (@see `Par.label`, `NARX_Formula.formulaStr`), so parseFormula (f.formulaStr) rebuilds f:
 *
 *      slot    := "?" [ NUM ]                         // a free parameter; ?w0 starts at w0
 *      atom    |= "pow" "(" atom "," slot ")"         // free exponent                 pow(y~1, ?)
 *               | FN "(" slot "+" atom ")"            // free shift inside FN          log(? + x~1)
 *                                                     //   (a free phase inside sin, cos)
 *               | FN "(" slot "*" atom ")"            // free rate inside FN           exp(? * x~1)
 *               | "sat"   "(" atom "," par ")"        // saturation x / (x + k)        sat(x~1, ?)
 *               | "hinge" "(" atom "," par ")"        // max (0, x - c)                hinge(x~1, ?)
 *               | "trans" "(" atom "," par "," par ")"   // 1 / (1 + e^-((x - c)/w))   trans(x~1, ?, ?)
 *               | "ema"   "(" atom "," par ")"        // moving average, decay d       ema(x~1, ?)
 *               | "dq"    "(" atom "," par ")"        // dampened quadratic, cap mx    dq(x~1, ?)
 *               | "wave"  "(" slot ")"                // one sin/cos pair, free frequency   wave(?)
 *      par     := slot | NUM                          // a number in a parameter position is a constant
 *
 *  Each slot is a `Par` created by its ROLE (@see `ParRole`): the exponent of pow is a
 *  `power`, the shift inside log/log1p/pow/sqrt a `shift`, inside sin/cos a `phase`, the
 *  scale a `rate`, and sat/hinge/trans/ema/dq take `half`/`thresh`/`thresh`,`width`/
 *  `decay`/`cap`, so the start value and the bounds of a parameter written as a bare ?
 *  come from the role and the training rows (@see `Par.resolve`).  A slot shared by the
 *  terms of a lag range is ONE parameter, e.g., pow(y~(3,1), ?) (@see `NARX_Formula`).
 *  Everything linear is built by `FormulaParser`'s own builders (`buildLinear`,
 *  `assemble`), so the two parsers cannot drift apart; what `FormulaParser` rejects
 *  ("free parameters (?) need nonlinear.NARX_FormulaParser") this parser builds.
 *  `powr` keeps a fixed exponent (its rounding to a rational is not smooth), so
 *  powr(term, ?) is rejected.
 *  @note    AI Assisted Code
 */

package scalation
package modeling
package forecasting
package nonlinear

import scalation.mathstat._

import FormulaParser.Ast
import ParRole._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NARX_FormulaParser` object parses formula specification strings with free
 *  nonlinear parameters into `NARX_Formula` objects (the inverse of
 *  `NARX_Formula.formulaStr`); a string without free parameters gives the same
 *  terms as `FormulaParser` (as an `NARX_Formula`).
 */
object NARX_FormulaParser:

    private val flaw = flawf ("NARX_FormulaParser")                     // flaw function

    /** The functions with a parameter position after the term and the number of parameters they take.
     */
    val NL_FUNCTIONS: Map [String, Int] = Map ("sat" -> 1, "hinge" -> 1, "trans" -> 2, "ema" -> 1, "dq" -> 1)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Parse a specification into an `NARX_Formula`, throwing `IllegalArgumentException`
     *  with a reason on any syntax or validity violation.  Use `tryParseFormula`
     *  inside search loops.
     *  @param formulaStr  the formula string (formula, optional TFORM directive)
     *  @param vars        the legal variable names (the dataset's keys)
     *  @param maxLag      the maximum allowed lag
     *  @param freqs       the wave frequencies (cycles/step) from the TRAINING-slice
     *                     periodogram; required only if the formula uses wave(k)
     *  @param maxTerms    the maximum number of expanded terms (parsimony guard)
     */
    def parseFormula (formulaStr: String, vars: Set [String], maxLag: Int = 8,
                      freqs: VectorD = null, maxTerms: Int = 40): NARX_Formula =

        val fail = FormulaParser.failer (formulaStr)
        val (lhs, elements, tform) = FormulaParser.syntax (formulaStr, fail)

        def parOf (role: ParRole, a: Ast): Par = a match                 // a parameter position
            case Ast.Slot (w0) => new Par (role, w0)                     // free: start value and bounds from the role
            case Ast.Num (v)   => new Par (Free, v, v, v)                // a constant
            case _             => fail ("a parameter position must hold ? or a number")

        def exponent (e: Double): Double =
            if e < FormulaParser.P_MIN || e > FormulaParser.P_MAX then
                fail (s"exponent $e outside [${FormulaParser.P_MIN},${FormulaParser.P_MAX}]")
            e

        def build (ast: Ast, inProduct: Boolean): Seq [Term] = ast match
            case Ast.WaveP (slot) =>
                if inProduct then fail ("wave(?) not allowed inside interactions")
                wave (new Par (Freq, slot.w0))
            case Ast.Call (fn, arg, num) if NL_FUNCTIONS.contains (fn) && num.isDefined =>
                buildP (fn, arg, Seq (Ast.Num (num.get)), inProduct)     // sat(x~1, 2.0): a constant in the slot
            case Ast.Call (fn, _, None) if NL_FUNCTIONS.contains (fn) && fn != "dq" =>
                fail (s"$fn needs (term, ${if fn == "trans" then "threshold, width" else "parameter"})")
            case Ast.CallP (fn, arg, ps) => buildP (fn, arg, ps, inProduct)
            case _ => FormulaParser.buildLinear (ast, inProduct, vars, maxLag, freqs, fail, build)
        end build

        def buildP (fn: String, arg: Ast, ps: Seq [Ast], inProduct: Boolean): Seq [Term] =
            val ts: Seq [Term] = arg match                               // the argument, shifted or scaled when asked
                case Ast.Shift (slot, a) =>
                    val role = if fn == "sin" || fn == "cos" then Phase else Shift
                    new Par (role, slot.w0) + build (a, inProduct)
                case Ast.Scale (slot, a) => new Par (Rate, slot.w0) * build (a, inProduct)
                case a                   => build (a, inProduct)
            (fn, ps) match
            case ("pow",  Seq (Ast.Slot (w0))) => pow (ts, new Par (Power, w0))
            case ("pow",  Seq (Ast.Num (e)))   => pow (ts, exponent (e))
            case ("powr", Seq (Ast.Num (e)))   => powr (ts, exponent (e))
            case ("powr", Seq (Ast.Slot (_)))  => fail ("powr keeps a fixed exponent (not smooth in it): use pow(term, ?)")
            case ("pow" | "powr", Seq ())      => fail (s"$fn needs (term, exponent)")
            case ("pow" | "powr", _)           => fail (s"$fn takes one exponent")
            case ("sat",   Seq (k))            => sat (ts, parOf (Half, k))
            case ("hinge", Seq (c))            => hinge (ts, parOf (Thresh, c))
            case ("trans", Seq (c, w))         => val (pc, pw) = (parOf (Thresh, c), parOf (Width, w))
                                                  ts.map (TransTerm (_, pc, pw))
            case ("ema",   Seq (d))            => ema (ts, parOf (Decay, d))
            case ("dq",    Seq (mx))           => dq (ts, parOf (Cap, mx))
            case ("dq",    Seq ())             => dq (ts)
            case (f, Seq ()) if ARX_Formula.functions.contains (f) => ts.map (ARX_Formula.functions (f))
            case (f, _) if NL_FUNCTIONS.contains (f) =>
                fail (s"$f takes ${NL_FUNCTIONS (f)} parameter${if NL_FUNCTIONS (f) > 1 then "s" else ""} after the term")
            case (f, _) if ARX_Formula.functions.contains (f) => fail (s"$f takes one argument")
            case (f, _) => fail (s"unknown function '$f'")
        end buildP

        val (resp, preds, tFormT) = FormulaParser.assemble (lhs, elements, tform, vars, maxTerms, build, fail)
        NARX_Formula (resp, preds, tFormT)
    end parseFormula

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Parse a specification, returning either the reason it was rejected (also
     *  logged when `FormulaParser.logRejects`) or the formula -- the entry point for
     *  search loops, where invalid proposals are counted and skipped, never fatal.
     *  @param formulaStr  the specification string (formula, optional TFORM directive)
     *  @param vars        the legal variable names (the dataset's keys)
     *  @param maxLag      the maximum allowed lag
     *  @param freqs       the wave frequencies from the TRAINING-slice periodogram
     *  @param maxTerms    the maximum number of expanded terms (parsimony guard)
     */
    def tryParseFormula (formulaStr: String, vars: Set [String], maxLag: Int = 8,
                         freqs: VectorD = null, maxTerms: Int = 40): Either [String, NARX_Formula] =
        try Right (parseFormula (formulaStr, vars, maxLag, freqs, maxTerms))
        catch case e: IllegalArgumentException =>
            if FormulaParser.logRejects then flaw ("tryParseFormula", e.getMessage)
            Left (e.getMessage)
    end tryParseFormula

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the free-parameter forms of the grammar as text for a prompt or a
     *  report (one line per form), in terms of the shown variable names.
     *  @param x  the name of a variable to use in the examples
     */
    def grammarLines (x: String = "x1"): String =
        s"""A number in a parameter position may be written ? (a FREE parameter, fitted on the training rows) or ?w
           |(fitted, starting at w).  Free-parameter forms:
           |  pow($x~1, ?)            free exponent (1.1 to 4)
           |  log(? + $x~1)           free shift inside log, log1p, sqrt or pow (keeps the argument positive)
           |  exp(? * $x~1)           free rate inside exp (also tanh); sin(? + ...) a free phase
           |  sat($x~1, ?)            saturation x / (x + k), free half-saturation k
           |  hinge($x~1, ?)          max(0, x - c), free threshold c
           |  trans($x~1, ?, ?)       smooth step 1 / (1 + exp(-(x - c) / w)), free threshold c and width w
           |  ema($x~1, ?)            exponentially weighted moving average, free decay in (0, 1)
           |  dq($x~1, ?)             dampened quadratic, free cap
           |  wave(?)                 one seasonal sin/cos pair with a free frequency
           |A lag range shares one parameter: pow($x~(3,1), ?) is three terms with one exponent.  Each free parameter
           |costs a penalty like a term, so use one only where a fixed number would be a guess.""".stripMargin
    end grammarLines

end NARX_FormulaParser


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nARX_FormulaParserTest` main function tests the parsing of free parameters:
 *  roles and labels, one shared parameter per lag range, the canonical key (start
 *  values ignored), the round trip through `formulaStr`/`proposalStr`, constants in
 *  parameter positions, a plain linear string giving the `FormulaParser` terms, and
 *  the rejects (each with a reason).
 *  > runMain scalation.modeling.forecasting.nonlinear.nARX_FormulaParserTest
 */
@main def nARX_FormulaParserTest (): Unit =

    import NARX_FormulaParser._

    val vars  = Set ("y", "icu", "pRate")
    val freqs = VectorD (0.13043, 0.06521)

    banner ("free parameters by role")
    val cases = Seq (
        ("y ~= y~(3,1) + pow(y~1, ?)",                       Seq (Power),         1),
        ("y ~= pow(y~(3,1), ?1.3) + log(? + icu~1)",         Seq (Power, Shift),  2),
        ("y ~= exp(? * icu~1) + sin(? + y~1) + log1p(?2 + pRate~1)", Seq (Rate, Phase, Shift), 3),
        ("y ~= sat(icu~1, ?) + hinge(y~1, ?) + ema(pRate~(2,1), ?)", Seq (Half, Thresh, Decay), 3),
        ("y ~= trans(y~1, ?, ?) * y~(2,1) + dq(icu~1, ?) + wave(?)",  Seq (Thresh, Width, Cap, Freq), 4),
        ("y ~= pow(? + y~1, ?) + pow(? + icu~1, 1.5) TFORM: Log1p", Seq (Shift, Power, Shift), 3))
    for (str, roles, n) <- cases do
        val f = parseFormula (str, vars, 6, freqs)
        println (s"formula:  $str")
        println (s"labels:   ${f.preds.map (_.label).mkString (", ")}")
        println (s"params:   ${f.params.map (_.role).mkString (", ")}  (n = ${f.nParams})")
        println (s"key:      ${f.key}")
        println (s"back:     ${f.formulaStr}\n")
        assert (f.params.map (_.role) == roles, s"roles ${f.params.map (_.role)} != $roles")
        assert (f.nParams == n, s"nParams ${f.nParams} != $n: a lag range shares one parameter")
        val back = parseFormula (f.formulaStr, vars, 6, freqs)
        assert (back.key == f.key, s"round trip by key failed: ${back.key} vs ${f.key}")
        val prop = parseFormula (f.proposalStr, vars, 6, freqs)
        assert (prop.key == f.key, s"proposalStr round trip failed: ${f.proposalStr}")
    end for

    banner ("canonical key: start values and fitted values do not count")
    val f1 = parseFormula ("y ~= pow(y~1, ?) + log(? + icu~1)", vars, 6)
    val f2 = parseFormula ("y ~= log(?3 + icu~1) + pow(y~1, ?1.7)", vars, 6)
    println (s"${f1.key}\n${f2.key}")
    assert (f1.key == f2.key, "keys must not depend on start values or term order")
    f2.params.foreach (_.set (2.0))                                     // as after a fit
    println (s"fitted:   ${f2.formulaStr}")
    println (s"proposal: ${f2.proposalStr}")
    assert (f2.key == f1.key, "the key must not change with the fit")
    assert (f2.formulaStr.contains ("pow(y~1, 2.000)") && f2.proposalStr.contains ("pow(y~1, ?2.000)"),
            "formulaStr shows the fitted value, proposalStr the slot with it")
    assert (parseFormula (f2.proposalStr, vars, 6).params.forall (_.w0 == 2.0), "the proposal's values are read back as start values")

    banner ("constants in parameter positions and a plain linear string")
    val fc = parseFormula ("y ~= sat(icu~1, 2.5) + pow(y~1, 1.5) + dq(y~1) + y~2", vars, 6)
    println (s"labels: ${fc.preds.map (_.label).mkString (", ")}")
    assert (fc.nFree == 0 && fc.nParams == 1, "constants are not free parameters")
    assert (fc.preds(0).label == "sat(icu~1, 2.500)", s"a constant half-saturation: ${fc.preds(0).label}")
    val lin  = "y ~= y~(2,1) + log1p(icu~1) + pow(pRate~1, 1.5) + y~1 * icu~1 + wave(1)"
    val fL   = parseFormula (lin, vars, 6, freqs)
    val fL0  = FormulaParser.parseFormula (lin, vars, 6, freqs)
    assert (fL.key == fL0.key && fL.nFree == 0, "a linear string gives FormulaParser's terms")
    assert (FormulaParser.tryParseFormula ("y ~= pow(y~1, ?)", vars, 6).isLeft, "FormulaParser rejects free parameters")

    banner ("rejects (each must return Left with a reason)")
    val bad = Array (
        "y ~= powr(y~1, ?)",                                            // not smooth in the exponent
        "y ~= hinge(y~1)",                                              // missing threshold
        "y ~= trans(y~1, ?)",                                           // one of two parameters
        "y ~= sat(y~1, ?, ?)",                                          // too many parameters
        "y ~= ? + y~1",                                                 // a shift outside a function
        "y ~= pow(y~1, ?) * wave(?)",                                   // wave inside a product
        "y ~= log(? + z~1)",                                            // unknown variable
        "y ~= cube(? + y~1)")                                           // unknown function
    for f <- bad do
        val r = tryParseFormula (f, vars, 6, freqs)
        assert (r.isLeft, s"should reject: $f")
        println (f"$f%-28s -> ${r.left.getOrElse ("")}")
    end for
    println ("nARX_FormulaParserTest: passed")

end nARX_FormulaParserTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nARX_FormulaParserTest2` main function parses the formula of `nARX_FormulaTest`
 *  from its string and fits it: the toy series y = 1 + 0.5 (x~1 + 2)^1.7 + 3 log (z~1 + 1.5)
 *  (no noise), so `fitNL_D` must recover the shift 2, the exponent 1.7 and the shift 1.5
 *  from a string whose parameters are bare ? slots, and the parsed formula must build the
 *  same design matrix as the DSL formula at the start values (`fit_D`).
 *  > runMain scalation.modeling.forecasting.nonlinear.nARX_FormulaParserTest2
 */
@main def nARX_FormulaParserTest2 (): Unit =

    val m  = 100
    val x_ = VectorD (for t <- 0 until m yield 1.0 + 0.05 * t + 0.6 * math.sin (0.7 * t))
    val z_ = VectorD (for t <- 0 until m yield 0.5 + 2.0 * math.abs (math.cos (0.3 * t)))
    val y_ = new VectorD (m)
    for t <- 1 until m do y_(t) = 1.0 + 0.5 * math.pow (x_(t-1) + 2.0, 1.7) + 3.0 * math.log (z_(t-1) + 1.5)
    y_(0) = y_(1)

    given dataContext: Map [String, VectorD] = Map ("y" -> y_, "x" -> x_, "z" -> z_)
    RidgeRegression.hp("lambda") = 1E-12

    val y = Vars.y; val x = Vars.x; val z = Vars.z
    val fDSL = y ~= pow (shift () + x~1, power ()) + log (shift () + z~1)
    val fStr = NARX_FormulaParser.parseFormula ("y ~= pow(? + x~1, ?) + log(? + z~1)", dataContext.keySet, 4)
    println (s"DSL:    ${fDSL.formulaStr}")
    println (s"parsed: ${fStr.formulaStr}")
    assert (NARX_Formula (fDSL).key == fStr.key, "the parsed formula must equal the DSL formula by key")

    banner ("fit_D (start values): the same design matrix")
    val m0 = fDSL.fit_D (1)
    val m1 = fStr.fit_D (1)
    assert ((m0.getX - m1.getX).normF < 1e-12, "design matrices differ at the start values")

    banner ("fitNL_D: recover shift 2, exponent 1.7, shift 1.5 from a string with bare ? slots")
    val mod = fStr.fitNL_D (1)
    val w   = fStr.params.map (_.value)
    println (s"fitted parameters = $w")
    println (s"fitted formula    = ${fStr.formulaStr}")
    println (s"intercept = ${mod.intercept}, slopes = ${mod.slopes}")
    assert (math.abs (w(0) - 2.0) < 1e-3 && math.abs (w(1) - 1.7) < 1e-3 && math.abs (w(2) - 1.5) < 1e-3,
            s"parameters not recovered: $w")
    println ("nARX_FormulaParserTest2: passed")

end nARX_FormulaParserTest2
