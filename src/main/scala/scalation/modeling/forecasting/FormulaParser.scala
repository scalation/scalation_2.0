
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// YousefChange --------- Note: comment only -- the scaladoc below no longer names the sub-package built on it
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Sat Aug 01 15:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Parser for `ARX_Formula` Specification Strings
 *
 *  Turns formula STRINGS (proposed by an LLM or written by hand) into `ARX_Formula`
 *  objects, so a structure search can run  propose (string) -> parse -> fit -> score.
 *  The grammar is the one `ARX_Formula` prints (@see `Term.label`, `ARX_Formula.formulaStr`),
 *  so parseFormula (f.formulaStr) rebuilds f:
 *
 *      input    := formula [ tform ]                  // tform on the same or the next line
 *      formula  := ID "~=" element { "+" element }
 *      tform    := "TFORM:" ID                        // e.g., TFORM: Log1p (any case)
 *      element  := atom { "*" atom }                  // at most two factors (no wave inside);
 *                                                     // a lag range distributes over the product
 *      atom     := ID "~" INT                         // lagged variable  y~3
 *                | ID "~(" INT ")"                    // same, parenthesized  y~(3)
 *                | ID "~(" INT "," INT ")"            // lag range        y~(5,1)
 *                | FN "(" atom ")"                    // FN in `ARX_Formula.functions`
 *                | "pow"  "(" atom "," NUM ")"        // plain power
 *                | "powr" "(" atom "," NUM ")"        // sign-safe power (@see `powr`)
 *                | "wave" "(" INT ")"                 // k seasonal sin/cos pairs; frequencies
 *                                                     // come from the caller (periodogram)
 *      -- free nonlinear parameters (YousefChange; built by `nonlinear.NARX_FormulaParser`,
 *         rejected here with a reason, @see `NARX_Formula`):
 *      slot     := "?" [ NUM ]                        // a free parameter, optionally with its start value
 *      atom     |= FN "(" slot "+" atom ")"           // free shift inside a function:  log(? + x~1)
 *                | FN "(" slot "*" atom ")"           // free scale inside a function:  exp(? * x~1)
 *                | "pow" "(" atom "," slot ")"        // free exponent
 *                | FNP "(" atom "," (slot | NUM) { "," (slot | NUM) } ")"   // sat, hinge, trans, ema, dq
 *                | "wave" "(" slot ")"                // one sin/cos pair with a free frequency
 *
 *  The syntax is parsed by FastParse (@see `pFormula`); the validity rules for a
 *  proposal (unknown variable, lag or exponent out of range, missing frequencies
 *  for wave, too many terms) are checked afterwards on the syntax tree, and each
 *  reject carries a reason, since the validity rate is itself a reported metric.
 *  Duplicate terms (same label) are collapsed rather than rejected, as they would
 *  make the design matrix singular.
 *  DEPENDENCY: add to build.sbt
 *      libraryDependencies += "com.lihaoyi" %% "fastparse" % "3.1.1"
 *
 *  @note    AI Assisted Code
 */

package scalation
package modeling
package forecasting

import scala.collection.mutable.{ArrayBuffer => VEC}

import fastparse._
import fastparse.MultiLineWhitespace._

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `FormulaParser` object parses formula specification strings into
 *  `ARX_Formula` objects (the inverse of `ARX_Formula.formulaStr`).
 */
object FormulaParser:

    private val flaw = flawf ("FormulaParser")                           // flaw function

    /** Whether `tryParseFormula` logs the reason for each rejected specification
     *  (rejections are expected inside a search loop; switch off for long runs).
     */
    var logRejects = true

    private [forecasting] val P_MIN = 0.2                                // minimum allowed exponent
    private [forecasting] val P_MAX = 3.0                                // maximum allowed exponent

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `Ast` enum is the syntax tree of one atom, before validation.
     */
    private [forecasting] enum Ast:
        case Lag   (name: String, hi: Int, lo: Int)                      // x~k (hi = lo = k) or x~(hi,lo)
        case Call  (fn: String, arg: Ast, num: Option [Double])          // fn(atom) or fn(atom, num)
        case Wave  (k: Int)                                              // wave(k)
        case Slot  (w0: Double)                                          // ? or ?w0 (NaN => no start value): a free parameter
        case Num   (v: Double)                                           // a number in a parameter position
        case Shift (slot: Slot, arg: Ast)                                // ? + atom (inside a function)
        case Scale (slot: Slot, arg: Ast)                                // ? * atom (inside a function)
        case CallP (fn: String, arg: Ast, pars: Seq [Ast])               // fn(arg, p1[, p2]) with a free parameter or a shifted/scaled argument
        case WaveP (slot: Slot)                                          // wave(?) or wave(?w0)
    end Ast

//  S Y N T A X  (FastParse; whitespace and newlines are skipped between tokens)
//  @note:  Context Bound [$: P] indicates that the method requires an implicit parsing context of type `P`
//          to be available in scope.  The compiler desugars this into a traditional implicit parameter list:
//          `[A](implicit $: P[A])`.  The symbol `$` is an anonymous variable name that Fastparse macros look for.

    private def pIdent [$: P]: P [String] =
        P (CharIn ("a-zA-Z_") ~~ CharsWhileIn ("a-zA-Z0-9_", 0)).!

    private def pInt [$: P]: P [Int] = P (CharsWhileIn ("0-9")).!.map (_.toInt)

    private def pNum [$: P]: P [Double] =
        P (CharsWhileIn ("0-9") ~~ ("." ~~ CharsWhileIn ("0-9")).?).!.map (_.toDouble)

    private def pLag [$: P]: P [Ast] =                                   // y~3, y~(3), y~(5,1)
        P (pIdent ~ "~" ~ ("(" ~ pInt ~ ("," ~ pInt).? ~ ")" | pInt.map (k => (k, Option.empty [Int]))))
          .map { case (nm, (hi, lo)) => Ast.Lag (nm, hi, lo.getOrElse (hi)) }

    private def pSlot [$: P]: P [Ast.Slot] =                             // ?, ?1.5, ?-0.5
        P ("?" ~~ ("-".!.? ~~ pNum).?)
          .map { case None => Ast.Slot (Double.NaN)
                 case Some ((sg, v)) => Ast.Slot (if sg.isDefined then -v else v) }

    private def pPar [$: P]: P [Ast] = P (pSlot | pNum.map (Ast.Num (_)))   // a parameter position: ? or a number

    private def pArg [$: P]: P [Ast] =                                   // a function argument: ? + atom, ? * atom or atom
        P ((pSlot ~ "+" ~ pAtom).map (Ast.Shift (_, _)) |
           (pSlot ~ "*" ~ pAtom).map (Ast.Scale (_, _)) |
           pAtom)

    private def pWave [$: P]: P [Ast] =                                  // wave(k) or wave(?)
        P ("wave" ~ "(" ~ (pSlot.map (Ast.WaveP (_)) | pInt.map (Ast.Wave (_))) ~ ")")

    private def pCall [$: P]: P [Ast] =                                  // fn(arg), fn(arg, num), fn(arg, ?), fn(arg, ?, ?)
        P (pIdent ~ "(" ~ pArg ~ ("," ~ pPar).rep (max = 2) ~ ")")
          .map { case (fn, arg, ps) =>
                 val plain = ps.forall (_.isInstanceOf [Ast.Num]) && ps.size <= 1 &&
                             ! arg.isInstanceOf [Ast.Shift] && ! arg.isInstanceOf [Ast.Scale]
                 if plain then Ast.Call (fn, arg, ps.headOption.map (_.asInstanceOf [Ast.Num].v))
                 else Ast.CallP (fn, arg, ps) }

    private def pAtom [$: P]: P [Ast] = P (pWave | pCall | pLag)

    private def pElement [$: P]: P [Seq [Ast]] = P (pAtom.rep (1, sep = "*"))

    private def pTForm [$: P]: P [String] = P (IgnoreCase ("tform") ~ ":" ~ pIdent)

    private def pFormula [$: P]: P [(String, Seq [Seq [Ast]], Option [String])] =
        P (pIdent ~ "~=" ~ pElement.rep (1, sep = "+") ~ pTForm.? ~ End)

//  V A L I D A T I O N   A N D   T E R M   B U I L D I N G

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Parse a specification into an `ARX_Formula`, throwing `IllegalArgumentException`
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
                      freqs: VectorD = null, maxTerms: Int = 40): ARX_Formula =
        val fail = failer (formulaStr)
        val (lhs, elements, tform) = syntax (formulaStr, fail)
        def build (ast: Ast, inProduct: Boolean): Seq [Term] = buildLinear (ast, inProduct, vars, maxLag, freqs, fail, build)
        val (resp, preds, tFormT) = assemble (lhs, elements, tform, vars, maxTerms, build, fail)
        ARX_Formula (resp, preds, tFormT)
    end parseFormula

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the function that rejects a specification with a reason.
     *  @param formulaStr  the specification string (quoted in the reason)
     */
    private [forecasting] def failer (formulaStr: String): String => Nothing =
        (reason: String) => throw new IllegalArgumentException (s"parse: $reason in formula: ${formulaStr.replace ('\n', ' ')}")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Parse the syntax of a specification into its response name, elements (each a
     *  product of atoms) and optional TFORM name; a syntax error is a reject.
     *  @param formulaStr  the specification string
     *  @param fail        the reject function
     */
    private [forecasting] def syntax (formulaStr: String, fail: String => Nothing): (String, Seq [Seq [Ast]], Option [String]) =
        fastparse.parse (formulaStr.trim, p => pFormula (using p)) match
        case Parsed.Success (v, _) => v
        case f: Parsed.Failure     => fail (s"syntax error (${f.msg})")
    end syntax

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build the terms of an atom of the LINEAR grammar (lagged variables, lag ranges,
     *  functions, powers, waves) with the validity rules; the free-parameter forms are
     *  rejected here with a reason (`nonlinear.NARX_FormulaParser` builds them).  The
     *  argument of a function is built by `recur`, so a parser that extends the grammar
     *  passes its own builder and still gets the linear forms from here.
     *  @param ast        the syntax tree of the atom
     *  @param inProduct  whether the atom is a factor of a product (no wave allowed)
     *  @param vars       the legal variable names
     *  @param maxLag     the maximum allowed lag
     *  @param freqs      the wave frequencies (null => no wave)
     *  @param fail       the reject function
     *  @param recur      the builder for the argument of a function (the caller's own)
     */
    private [forecasting] def buildLinear (ast: Ast, inProduct: Boolean, vars: Set [String], maxLag: Int, freqs: VectorD,
                                           fail: String => Nothing, recur: (Ast, Boolean) => Seq [Term]): Seq [Term] =
        ast match
        case Ast.Lag (nm, hi, lo) =>
            if ! vars.contains (nm) then fail (s"variable '$nm' not in dataset ${vars.mkString ("{", ",", "}")}")
            if lo < 1 || hi > maxLag || hi < lo then
                if hi == lo then fail (s"lag $hi outside [1,$maxLag]")
                else fail (s"lag range ($hi,$lo) outside [1,$maxLag]")
            FormulaVar (nm) ~ ((hi, lo))
        case Ast.Wave (k) =>
            if inProduct then fail ("wave(k) not allowed inside interactions")
            if freqs == null then fail ("wave(k) used but no frequencies supplied")
            if k < 1 || k > freqs.dim then fail (s"wave($k) outside [1,${freqs.dim}]")
            wave (freqs, k)
        case Ast.Call (fn, arg, num) => (fn, num) match
            case ("pow" | "powr", None)     => fail (s"$fn needs (term, exponent)")
            case ("pow" | "powr", Some (e)) =>
                if e < P_MIN || e > P_MAX then fail (s"$fn exponent $e outside [$P_MIN,$P_MAX]")
                val ts = recur (arg, inProduct)
                if fn == "pow" then pow (ts, e) else powr (ts, e)
            case (f, Some (_)) if ARX_Formula.functions.contains (f) => fail (s"$f takes one argument")
            case (f, None)     if ARX_Formula.functions.contains (f) =>
                recur (arg, inProduct).map (ARX_Formula.functions (f))
            case (f, _) => fail (s"unknown function '$f'")
        case Ast.Slot (_) | Ast.Num (_) | Ast.Shift (_, _) | Ast.Scale (_, _) | Ast.CallP (_, _, _) | Ast.WaveP (_) =>
            fail ("free parameters (?) need nonlinear.NARX_FormulaParser")
    end buildLinear

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Assemble a parsed specification: check the response and the TFORM, build the
     *  terms of every element (a product of two atoms distributes a lag range), apply
     *  the term cap and return the response, the distinct terms and the transform type.
     *  @param lhs       the response name
     *  @param elements  the elements, each a product of atoms
     *  @param tform     the TFORM name, if any
     *  @param vars      the legal variable names
     *  @param maxTerms  the maximum number of expanded terms
     *  @param build     the builder of an atom's terms
     *  @param fail      the reject function
     */
    private [forecasting] def assemble (lhs: String, elements: Seq [Seq [Ast]], tform: Option [String], vars: Set [String],
                                        maxTerms: Int, build: (Ast, Boolean) => Seq [Term], fail: String => Nothing):
                                       (FormulaVar, VEC [Term], TransformT) =
        if ! vars.contains (lhs) then fail (s"response '$lhs' not a dataset variable")
        val tFormT = tform match
            case None      => null
            case Some (nm) => TransformT.values.find (_.toString.equalsIgnoreCase (nm))
                                        .getOrElse (fail (s"unknown TFORM '$nm'"))

        val preds = VEC [Term] ()
        for el <- elements do
            el.size match
            case 1 => preds ++= build (el(0), false)
            case 2 => val (as, bs) = (build (el(0), true), build (el(1), true))
                      for a <- as; b <- bs do preds += ProductTerm (a, b)   // a range distributes over the product
            case n => fail (s"interaction with $n factors (at most two)")
        end for

        if preds.size > maxTerms then fail (s"${preds.size} terms exceeds maxTerms = $maxTerms")
        (FormulaVar (lhs), ARX_Formula.distinct (preds), tFormT)
    end assemble

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Parse a specification, returning either the reason it was rejected (also
     *  logged when `logRejects`) or the formula -- the entry point for search
     *  loops, where invalid proposals are counted and skipped, never fatal.
     *  @param formulaStr  the specification string (formula, optional TFORM directive)
     *  @param vars        the legal variable names (the dataset's keys)
     *  @param maxLag      the maximum allowed lag
     *  @param freqs       the wave frequencies from the TRAINING-slice periodogram
     *  @param maxTerms    the maximum number of expanded terms (parsimony guard)
     */
    def tryParseFormula (formulaStr: String, vars: Set [String], maxLag: Int = 8,
                         freqs: VectorD = null, maxTerms: Int = 40): Either [String, ARX_Formula] =
        try Right (parseFormula (formulaStr, vars, maxLag, freqs, maxTerms))
        catch case e: IllegalArgumentException =>
            if logRejects then flaw ("tryParseFormula", e.getMessage)
            Left (e.getMessage)
    end tryParseFormula

end FormulaParser


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `formulaParserTest` main function tests parsing, the parse/serialize
 *  round-trip, the lenient forms and the validation rejects on Covid-19 style
 *  specifications.
 *  > runMain scalation.modeling.forecasting.formulaParserTest
 */
@main def formulaParserTest (): Unit =

    import FormulaParser._

    val vars  = Set ("y", "hosp_patients", "icu_patients")
    val freqs = VectorD (0.13043, 0.06521)                              // stand-in periodogram output

    banner ("valid formulas")
    val formulas = Array (
        "y ~= y~(5,1) + hosp_patients~1 + icu_patients~(2,1)",
        "y ~= y~(5,1) + pow(y~(5,1), 1.5) + log1p(hosp_patients~1) + icu_patients~2",
        "y ~= y~(3,1) + y~1 * hosp_patients~1 + wave(2)",
        "y ~= powr(y~2, 0.67) + expm1(icu_patients~1) TFORM: Norm")

    for f <- formulas do
        val fP = parseFormula (f, vars, maxLag = 8, freqs = freqs)
        println (s"formula: $f")
        println (s"resp:    ${fP.resp.name}, nTerms: ${fP.nTerms}, tFormT: ${fP.tFormT}")
        println (s"labels:  ${fP.preds.map (_.label)}")
        println (s"back:    ${fP.formulaStr}")
        println (s"key:     ${fP.key}\n")
    end for

    banner ("round-trip: parseFormula (f.formulaStr) == f (by key)")
    for f <- formulas do
        val f1 = parseFormula (f, vars, freqs = freqs)
        val f2 = parseFormula (f1.formulaStr, vars, freqs = freqs)
        assert (f1.key == f2.key, s"round-trip failed for: $f")
    end for
    println ("round-trip: all passed")

    banner ("lenient forms: y~(k) == y~k, duplicates collapse, TFORM case-insensitive or on a new line, spaces")
    val len = parseFormula ("y ~= y~(3,1) + y~2 + icu_patients~(2)\nTFORM: log1p", vars, freqs = freqs)
    assert (len.nTerms == 4, s"expected 4 distinct terms, got ${len.nTerms}")
    assert (len.tFormT == TransformT.Log1p, s"expected Log1p, got ${len.tFormT}")

    val inl = parseFormula ("y ~= y~1 + hosp_patients~2 TFORM: Log1p", vars, freqs = freqs)
    assert (inl.nTerms == 2 && inl.tFormT == TransformT.Log1p, "inline TFORM not accepted")

    val spc = parseFormula ("  y ~=  y ~ 1 +log1p( hosp_patients~2 )  ", vars, freqs = freqs)
    assert (spc.nTerms == 2 && spc.formulaStr == "y ~= y~1 + log1p(hosp_patients~2)", s"spacing not tolerated: ${spc.formulaStr}")

    val prd = parseFormula ("y ~= y~(2,1) * hosp_patients~1", vars, freqs = freqs)
    assert (prd.nTerms == 2 && prd.preds.forall (_.isInstanceOf [ProductTerm]),
            "range inside a product did not expand to two products")
    println (s"product: ${prd.formulaStr}")
    println (s"lenient: ${len.formulaStr}")

    banner ("rejects (each must return Left with a reason)")
    val bad = Array (
        "y ~= z~1",                                                     // unknown variable
        "y ~= y~9",                                                     // lag out of range
        "y ~= y~0",                                                     // lag 0 (DSL only)
        "y ~= pow(y~1, 9.0)",                                           // exponent out of range
        "y ~= pow(y~1)",                                                // missing exponent
        "y ~= log1p(y~1, 2)",                                           // unary function with two arguments
        "y ~= cube(y~1)",                                               // unknown function
        "y ~= y~1 * y~2 * y~3",                                         // 3-factor interaction
        "y ~= wave(1) * y~2",                                           // wave inside a product
        "y ~= wave(5)",                                                 // k exceeds freqs.dim
        "q ~= y~1",                                                     // unknown response
        "y ~= y~(1,5)",                                                 // reversed range
        "y ~= y~1 TFORM: Cube",                                         // unknown TFORM
        "y ~= y~1 + ",                                                  // dangling +
        "y ~= ")                                                        // empty rhs
    for f <- bad do
        val r = tryParseFormula (f, vars, freqs = freqs)
        assert (r.isLeft, s"should reject: $f")
        println (f"$f%-28s -> ${r.left.getOrElse ("")}")
    end for
    println ("rejects: all correctly refused")

end formulaParserTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `formulaParserTest2` main function tests that every grammar production
 *  is the `ARX_Formula` DSL: each specification string, parsed, must build the
 *  same `ARX_D` design matrix (`ARX_D.fromTerms`) as the DSL formula written by
 *  hand -- the parser calls the DSL's own term constructors -- and each DSL
 *  formula must serialize (`formulaStr`) back to a string that parses to the same key,
 *  so DSL formulas can seed the search.  Uses the Covid-19 `BEST2_EXO` data (icu
 *  patients, test positivity rate).
 *  > runMain scalation.modeling.forecasting.formulaParserTest2
 */
@main def formulaParserTest2 (): Unit =

    import Example_Covid._
    import TransformT.Log1p

    val (xe, y_) = clip (loadData (BEST2_EXO))                          // icu_patients, positive_rate

    given dataContext: Map [String, VectorD] = Map (
        "y"     -> y_,
        "icu"   -> xe.col(0),
        "pRate" -> xe.col(1))

    val vars    = dataContext.keySet
    val hh      = 4
    val tr_size = Model.trSize (y_.dim)
    val pg      = new Periodogram (y_(0 until tr_size), "y")
    val freqs   = pg.topKFrequencies (pg.amplitudes, 2)

    val y     = Vars.y
    val icu   = Vars.icu
    val pRate = Vars.pRate

    val cases: Seq [(String, ARX_Formula)] = Seq (
        ("y ~= y~(3,1) + icu~2",
         y ~= y~(3,1) + icu~2),
        ("y ~= y~1 + log1p(icu~1) + sq(pRate~2) + exp(pRate~1)",
         y ~= y~1 + log1p (icu~1) + sq (pRate~2) + exp (pRate~1)),
        ("y ~= pow(y~(2,1), 1.5) + powr(icu~1, 0.5) + log(y~2)",     // log (icu) would hit icu = 0 early on
         y ~= pow (y~(2,1), 1.5) + powr (icu~1, 0.5) + log (y~2)),
        ("y ~= y~1 + y~1 * icu~1 + log1p(icu~1) * pRate~2",
         y ~= y~1 + (y~1 * icu~1) + (log1p (icu~1) * pRate~2)),
        ("y ~= y~(2,1) * icu~1",                                        // a range distributes over the product
         y ~= y~(2,1) * icu~1),
        ("y ~= y~(2,1) + wave(2)",
         y ~= y~(2,1) + wave (freqs, 2)),
        ("y ~= y~(4,1) + icu~(2,1) + sq(y~1)\nTFORM: Log1p",
         (y ~= y~(4,1) + icu~(2,1) + sq (y~1)).copy (tFormT = Log1p)),
        ("y ~= pow(y~(2,1), 1.8) + icu~1",                               // the DSL's power operator
         y ~= y~(2,1) ↑ 1.8 + icu~1))

    def maxAbs (a: MatrixD): Double = (0 until a.dim).map (i => a(i).abs.max).max
    def maxAbsDiff (a: MatrixD, b: MatrixD): Double = (0 until a.dim).map (i => (a(i) - b(i)).abs.max).max

    banner ("each grammar production: parsed string vs hand-written DSL formula")
    for (formulaStr, f) <- cases do
        val fP   = FormulaParser.parseFormula (formulaStr, vars, maxLag = 6, freqs = freqs)
        val modP = ARX_D.fromTerms (fP.resp, fP.preds, hh, tFormT = fP.tFormT)
        val modF = ARX_D.fromTerms (f.resp, f.preds, hh, tFormT = f.tFormT)
        val (xP, xF) = (modP.getX, modF.getX)
        val label    = formulaStr.replace ('\n', ' ')
        println (f"$label%-60s  columns = ${xP.dim2}")
        assert (fP.key == f.key, s"parsed and DSL formulas differ:\n  ${fP.key}\n  ${f.key}")
        assert (fP.tFormT == f.tFormT, s"tFormT ${fP.tFormT} != ${f.tFormT}")
        assert (xP.dims == xF.dims, s"design matrix dims ${xP.dims} != ${xF.dims}")
        assert (maxAbsDiff (xP, xF) <= 1e-9 * (1.0 + maxAbs (xF)), "design matrices differ")
        assert ((modP.getY - modF.getY).abs.max <= 1e-9 * (1.0 + modF.getY.abs.max), "responses differ")
        assert (xP.dim2 == fP.nTerms, s"one column per term expected: ${xP.dim2} vs ${fP.nTerms}")
        assert (modF.getFname.toSeq == f.preds.map (_.label), "fname must be the grammar labels")
        val back = FormulaParser.parseFormula (f.formulaStr, vars, maxLag = 6, freqs = freqs)
        assert (back.key == f.key, s"DSL round trip key mismatch:\n  ${back.key}\n  ${f.key}")
    end for
    println ("every production builds the DSL's terms and the same ARX_D design matrix")

end formulaParserTest2

