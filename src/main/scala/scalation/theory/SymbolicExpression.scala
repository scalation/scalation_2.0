//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Wed May 06 18:35:00 CDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Symbolic Equation DSL for General Symbolic Regression and Time Series Forecasting
 *
 *  The `SymbolicExpression` file provides a compact abstract syntax tree (AST),
 *  embedded Scala DSL, parser for portable S-expression strings, validator,
 *  evaluator, complexity scorer, and LaTeX renderer.
 */

package scalation
package theory

import scala.math.{abs, cos, exp, log, log1p, max, min, pow, sin, sqrt, tanh}

import scalation.mathstat.VectorD

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `flaw` function reports fatal errors for this standalone DSL module.
 *  It is intentionally local to avoid depending on another ScalaTion utility.
 *  @param caller  the method/object where the flaw occurred
 *  @param msg     the diagnostic message
 */
private def flaw (caller: String, msg: String): Nothing =
    throw new IllegalArgumentException (s"$caller: $msg")

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymExpr` enum defines the symbolic expression AST used by the symbolic
 *  regression DSL.  It supports both ordinary symbolic regression variables and
 *  time-series feature constructors such as lags, differences, and rolling means.
 */
enum SymExpr:

    case Const (value: Double)                                          // numeric constant
    case Coef  (index: Int)                                             // fitted coefficient c_i
    case Var   (name: String)                                           // ordinary variable x

    case Add   (left: SymExpr, right: SymExpr)                          // left + right
    case Sub   (left: SymExpr, right: SymExpr)                          // left - right
    case Mul   (left: SymExpr, right: SymExpr)                          // left * right
    case Div   (left: SymExpr, right: SymExpr)                          // protected left / right
    case Neg   (expr: SymExpr)                                          // -expr
    case Pow   (base: SymExpr, power: Double)                           // base^power

    case Abs   (expr: SymExpr)                                          // abs (expr)
    case Sqrt  (expr: SymExpr)                                          // protected sqrt (expr)
    case Log   (expr: SymExpr)                                          // protected log (expr)
    case Log1p (expr: SymExpr)                                          // protected log1p (expr)
    case Exp   (expr: SymExpr)                                          // clipped exp (expr)
    case Sin   (expr: SymExpr)                                          // sin (expr)
    case Cos   (expr: SymExpr)                                          // cos (expr)
    case Tanh  (expr: SymExpr)                                          // tanh (expr)

    case Lag          (variable: String, k: Int)                        // variable_{t-k}
    case Diff         (variable: String, k: Int)                        // variable_{t-k} - variable_{t-k-1}
    case SeasonalDiff (variable: String, season: Int)                   // variable_{t-1} - variable_{t-season-1}
    case RollMean     (variable: String, window: Int)                   // rolling mean using past window
    case RollStd      (variable: String, window: Int)                   // rolling stdev using past window
    case Ewma         (variable: String, alpha: Double)                 // exponentially weighted moving average

end SymExpr


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymDSL` object provides an embedded Scala DSL for building symbolic
 *  expressions naturally, while still producing the canonical `SymExpr` tree.
 */
object SymDSL:

    import SymExpr._

    def x (name: String): SymExpr = Var (name)
    def c (index: Int): SymExpr = Coef (index)
    def const (value: Double): SymExpr = Const (value)

    def lag (variable: String, k: Int): SymExpr = Lag (variable, k)
    def diff (variable: String, k: Int): SymExpr = Diff (variable, k)
    def seasonal_diff (variable: String, season: Int): SymExpr = SeasonalDiff (variable, season)
    def rollmean (variable: String, window: Int): SymExpr = RollMean (variable, window)
    def rollstd (variable: String, window: Int): SymExpr = RollStd (variable, window)
    def ewma (variable: String, alpha: Double): SymExpr = Ewma (variable, alpha)

    def abs_ (expr: SymExpr): SymExpr = Abs (expr)
    def sqrt_ (expr: SymExpr): SymExpr = Sqrt (expr)
    def log_ (expr: SymExpr): SymExpr = Log (expr)
    def log1p_ (expr: SymExpr): SymExpr = Log1p (expr)
    def exp_ (expr: SymExpr): SymExpr = Exp (expr)
    def sin_ (expr: SymExpr): SymExpr = Sin (expr)
    def cos_ (expr: SymExpr): SymExpr = Cos (expr)
    def tanh_ (expr: SymExpr): SymExpr = Tanh (expr)

    extension (left: SymExpr)
        def + (right: SymExpr): SymExpr = Add (left, right)
        def - (right: SymExpr): SymExpr = Sub (left, right)
        def * (right: SymExpr): SymExpr = Mul (left, right)
        def / (right: SymExpr): SymExpr = Div (left, right)
        def ~^ (power: Double): SymExpr = Pow (left, power)

    extension (value: Double)
        def + (right: SymExpr): SymExpr = Add (Const (value), right)
        def - (right: SymExpr): SymExpr = Sub (Const (value), right)
        def * (right: SymExpr): SymExpr = Mul (Const (value), right)
        def / (right: SymExpr): SymExpr = Div (Const (value), right)

end SymDSL


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TimeSeriesFrame` trait provides a minimal time-series data interface.
 *  Adapters can later be added for ScalaTion tables, matrices, or forecasting
 *  datasets without changing the symbolic expression evaluator.
 */
trait TimeSeriesFrame:

    def apply (variable: String, t: Int): Double
    def length: Int
    def variables: Set [String]

end TimeSeriesFrame


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VectorFrame` class provides a simple `TimeSeriesFrame` backed by a map
 *  from variable names to `VectorD` time series.
 *  @param data  the named time series vectors
 */
class VectorFrame (data: Map [String, VectorD]) extends TimeSeriesFrame:

    private val n = if data.isEmpty then 0 else data.head._2.dim

    def apply (variable: String, t: Int): Double =
        if ! data.contains (variable) then flaw ("apply", s"unknown variable $variable")
        if t < 0 || t >= n then flaw ("apply", s"time index $t outside [0, ${n-1}]")
        data(variable)(t)
    end apply

    def length: Int = n
    def variables: Set [String] = data.keySet

end VectorFrame


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymEval` object evaluates symbolic expressions against a `TimeSeriesFrame`.
 *  Protected operators are used by default to avoid numerical crashes during
 *  symbolic search.
 */
object SymEval:

    import SymExpr._

    private val EPS = 1E-8

    def apply (expr: SymExpr, frame: TimeSeriesFrame, t: Int, coef: VectorD = null): Double =
        expr match
            case Const (value) => value
            case Coef (index)  => if coef == null then 1.0 else coef(index)
            case Var (name)    => frame(name, t)

            case Add (a, b) => apply (a, frame, t, coef) + apply (b, frame, t, coef)
            case Sub (a, b) => apply (a, frame, t, coef) - apply (b, frame, t, coef)
            case Mul (a, b) => apply (a, frame, t, coef) * apply (b, frame, t, coef)
            case Div (a, b) => apply (a, frame, t, coef) / safeDenom (apply (b, frame, t, coef))
            case Neg (a)    => -apply (a, frame, t, coef)
            case Pow (a, p) => pow (apply (a, frame, t, coef), p)

            case Abs (a)   => abs (apply (a, frame, t, coef))
            case Sqrt (a)  => sqrt (abs (apply (a, frame, t, coef)))
            case Log (a)   => log (abs (apply (a, frame, t, coef)) + EPS)
            case Log1p (a) => log1p (max (apply (a, frame, t, coef), -1.0 + EPS))
            case Exp (a)   => exp (max (-50.0, min (50.0, apply (a, frame, t, coef))))
            case Sin (a)   => sin (apply (a, frame, t, coef))
            case Cos (a)   => cos (apply (a, frame, t, coef))
            case Tanh (a)  => tanh (apply (a, frame, t, coef))

            case Lag (v, k)                 => frame(v, t-k)
            case Diff (v, k)                => frame(v, t-k) - frame(v, t-k-1)
            case SeasonalDiff (v, season)   => frame(v, t-1) - frame(v, t-season-1)
            case RollMean (v, window)       => rollingMean (frame, v, t, window)
            case RollStd (v, window)        => rollingStd (frame, v, t, window)
            case Ewma (v, alpha)            => ewma (frame, v, t, alpha)
    end apply

    def evalVector (expr: SymExpr, frame: TimeSeriesFrame, from: Int, until: Int,
                    coef: VectorD = null): VectorD =
        VectorD ((for t <- from until until yield apply (expr, frame, t, coef)).toArray)
    end evalVector

    private def rollingMean (frame: TimeSeriesFrame, variable: String, t: Int, window: Int): Double =
        var sum = 0.0
        for i <- 1 to window do sum += frame(variable, t-i)
        sum / window
    end rollingMean

    private def rollingStd (frame: TimeSeriesFrame, variable: String, t: Int, window: Int): Double =
        val mean = rollingMean (frame, variable, t, window)
        var sum2 = 0.0
        for i <- 1 to window do
            val z = frame(variable, t-i) - mean
            sum2 += z * z
        sqrt (sum2 / window)
    end rollingStd

    private def ewma (frame: TimeSeriesFrame, variable: String, t: Int, alpha: Double): Double =
        val depth = min (20, t)
        var acc   = 0.0
        var weight = alpha
        for i <- 1 to depth do
            acc += weight * frame(variable, t-i)
            weight *= 1.0 - alpha
        acc
    end ewma

    private def safeDenom (x: Double): Double =
        if abs (x) < EPS then if x >= 0.0 then EPS else -EPS else x

end SymEval


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymLatex` object renders symbolic expressions as LaTeX strings.
 */
object SymLatex:

    import SymExpr._

    def apply (expr: SymExpr): String = render (expr)

    def render (expr: SymExpr): String =
        expr match
            case Const (value) => formatDouble (value)
            case Coef (index)  => s"c_{$index}"
            case Var (name)    => latexVar (name)

            case Add (a, b) => s"${render (a)} + ${render (b)}"
            case Sub (a, b) => s"${render (a)} - ${render (b)}"
            case Mul (a, b) => s"${wrap (a)} ${wrap (b)}"
            case Div (a, b) => s"\\frac{${render (a)}}{${render (b)}}"
            case Neg (a)    => s"-${wrap (a)}"
            case Pow (a, p) => s"${wrap (a)}^{${formatPower (p)}}"

            case Abs (a)   => s"\\left|${render (a)}\\right|"
            case Sqrt (a)  => s"\\sqrt{${render (a)}}"
            case Log (a)   => s"\\log\\left(${render (a)}\\right)"
            case Log1p (a) => s"\\log\\left(1 + ${render (a)}\\right)"
            case Exp (a)   => s"e^{${render (a)}}"
            case Sin (a)   => s"\\sin\\left(${render (a)}\\right)"
            case Cos (a)   => s"\\cos\\left(${render (a)}\\right)"
            case Tanh (a)  => s"\\tanh\\left(${render (a)}\\right)"

            case Lag (v, k)               => s"${latexVar (v)}_{t-$k}"
            case Diff (v, k)              => s"\\Delta_{$k} ${latexVar (v)}_{t-1}"
            case SeasonalDiff (v, season) => s"${latexVar (v)}_{t-1} - ${latexVar (v)}_{t-${season+1}}"
            case RollMean (v, window)     => s"\\operatorname{mean}_{$window}\\left(${latexVar (v)}\\right)_{t-1}"
            case RollStd (v, window)      => s"\\operatorname{std}_{$window}\\left(${latexVar (v)}\\right)_{t-1}"
            case Ewma (v, alpha)          => s"\\operatorname{EWMA}_{${formatDouble (alpha)}}\\left(${latexVar (v)}\\right)_{t-1}"
    end render

    private def wrap (expr: SymExpr): String =
        expr match
            case Add (_, _) | Sub (_, _) => s"\\left(${render (expr)}\\right)"
            case _                       => render (expr)

    private def latexVar (name: String): String =
        if name.matches ("[a-zA-Z]") then name else s"\\mathrm{${name.replace ("_", "\\_")}}"

    private def formatPower (p: Double): String =
        if p.isWhole then p.toInt.toString else formatDouble (p)

    private def formatDouble (x: Double): String =
        if x.isWhole then x.toInt.toString else f"$x%.4g"

end SymLatex


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymParser` object parses the portable LLM-facing S-expression DSL into
 *  the canonical `SymExpr` AST.
 */
object SymParser:

    import SymExpr._

    def parse (input: String): SymExpr =
        val tokens       = tokenize (input)
        val (expr, rest) = parseExpr (tokens)
        if rest.nonEmpty then flaw ("parse", s"unexpected tokens: ${rest.mkString (" ")}")
        expr
    end parse

    private def tokenize (input: String): List [String] =
        input.replace ("(", " ( ").replace (")", " ) ").split ("\\s+").filter (_.nonEmpty).toList

    private def parseExpr (tokens: List [String]): (SymExpr, List [String]) =
        tokens match
            case Nil => flaw ("parseExpr", "unexpected end of input")

            case "(" :: op :: rest => parseCompound (op.toLowerCase, rest)

            case token :: rest if token.matches ("c\\d+") =>
                (Coef (token.drop (1).toInt), rest)

            case token :: rest if token.matches ("-?\\d+(\\.\\d+)?([eE]-?\\d+)?") =>
                (Const (token.toDouble), rest)

            case token :: rest =>
                (Var (token), rest)
    end parseExpr

    private def parseCompound (op: String, tokens: List [String]): (SymExpr, List [String]) =
        op match
            case "add" => parseBinary (Add.apply, tokens)
            case "sub" => parseBinary (Sub.apply, tokens)
            case "mul" => parseBinary (Mul.apply, tokens)
            case "div" => parseBinary (Div.apply, tokens)

            case "neg"   => parseUnary (Neg.apply, tokens)
            case "abs"   => parseUnary (Abs.apply, tokens)
            case "sqrt"  => parseUnary (Sqrt.apply, tokens)
            case "log"   => parseUnary (Log.apply, tokens)
            case "log1p" => parseUnary (Log1p.apply, tokens)
            case "exp"   => parseUnary (Exp.apply, tokens)
            case "sin"   => parseUnary (Sin.apply, tokens)
            case "cos"   => parseUnary (Cos.apply, tokens)
            case "tanh"  => parseUnary (Tanh.apply, tokens)

            case "pow" =>
                val (base, r1) = parseExpr (tokens)
                r1 match
                    case p :: ")" :: rest => (Pow (base, p.toDouble), rest)
                    case _                 => flaw ("parseCompound", "expected: (pow expr power)")

            case "lag" => parseVarInt (Lag.apply, tokens, "lag")
            case "diff" => parseVarInt (Diff.apply, tokens, "diff")
            case "seasonal_diff" => parseVarInt (SeasonalDiff.apply, tokens, "seasonal_diff")
            case "rollmean" => parseVarInt (RollMean.apply, tokens, "rollmean")
            case "rollstd" => parseVarInt (RollStd.apply, tokens, "rollstd")

            case "ewma" =>
                tokens match
                    case variable :: alpha :: ")" :: rest => (Ewma (variable, alpha.toDouble), rest)
                    case _ => flaw ("parseCompound", "expected: (ewma variable alpha)")

            case other => flaw ("parseCompound", s"unknown operator $other")
    end parseCompound

    private def parseBinary (build: (SymExpr, SymExpr) => SymExpr,
                             tokens: List [String]): (SymExpr, List [String]) =
        val (a, r1) = parseExpr (tokens)
        val (b, r2) = parseExpr (r1)
        expectClose (build (a, b), r2)
    end parseBinary

    private def parseUnary (build: SymExpr => SymExpr,
                            tokens: List [String]): (SymExpr, List [String]) =
        val (a, r1) = parseExpr (tokens)
        expectClose (build (a), r1)
    end parseUnary

    private def parseVarInt (build: (String, Int) => SymExpr, tokens: List [String], name: String): (SymExpr, List [String]) =
        tokens match
            case variable :: k :: ")" :: rest => (build (variable, k.toInt), rest)
            case _ => flaw ("parseVarInt", s"expected: ($name variable integer)")
    end parseVarInt

    private def expectClose (expr: SymExpr, tokens: List [String]): (SymExpr, List [String]) =
        tokens match
            case ")" :: rest => (expr, rest)
            case _           => flaw ("expectClose", "expected closing parenthesis")

end SymParser


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymConstraints` case class gives validation limits for generated symbolic
 *  expressions.
 *  @param variables       the legal variables
 *  @param maxDepth        the maximum expression depth
 *  @param maxLag          the maximum allowed lag or seasonal lag
 *  @param maxRollWindow   the maximum rolling-window size
 *  @param powers          the legal powers
 */
case class SymConstraints (variables: Set [String],
                           maxDepth: Int = 6,
                           maxLag: Int = 52,
                           maxRollWindow: Int = 52,
                           powers: Set [Double] = Set (-2.0, -1.0, 0.5, 1.0, 2.0, 3.0))


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymValidator` object validates generated symbolic expressions.
 */
object SymValidator:

    import SymExpr._

    def validate (expr: SymExpr, cons: SymConstraints): Either [List [String], SymExpr] =
        val errors = collectErrors (expr, cons, 1)
        if errors.isEmpty then Right (expr) else Left (errors)
    end validate

    def isValid (expr: SymExpr, cons: SymConstraints): Boolean = validate (expr, cons).isRight

    private def collectErrors (expr: SymExpr, cons: SymConstraints, depth: Int): List [String] =
        val depthErrors = if depth > cons.maxDepth then List (s"expression exceeds maxDepth = ${cons.maxDepth}") else Nil
        depthErrors ++ nodeErrors (expr, cons) ++ children (expr).flatMap (collectErrors (_, cons, depth+1))
    end collectErrors

    private def nodeErrors (expr: SymExpr, cons: SymConstraints): List [String] =
        expr match
            case Var (name) => checkVar (name, cons)
            case Lag (v, k) => checkVar (v, cons) ++ checkRange ("lag", k, 1, cons.maxLag)
            case Diff (v, k) => checkVar (v, cons) ++ checkRange ("diff", k, 1, cons.maxLag)
            case SeasonalDiff (v, season) => checkVar (v, cons) ++ checkRange ("seasonal_diff", season, 2, cons.maxLag)
            case RollMean (v, window) => checkVar (v, cons) ++ checkRange ("rollmean", window, 2, cons.maxRollWindow)
            case RollStd (v, window) => checkVar (v, cons) ++ checkRange ("rollstd", window, 2, cons.maxRollWindow)
            case Ewma (v, alpha) => checkVar (v, cons) ++ { if alpha > 0.0 && alpha <= 1.0 then Nil else List ("ewma alpha must be in (0, 1]") }
            case Pow (_, power) => if cons.powers.contains (power) then Nil else List (s"power $power is not allowed")
            case _ => Nil
    end nodeErrors

    private def children (expr: SymExpr): List [SymExpr] =
        expr match
            case Add (a, b) => List (a, b)
            case Sub (a, b) => List (a, b)
            case Mul (a, b) => List (a, b)
            case Div (a, b) => List (a, b)
            case Neg (a) => List (a)
            case Pow (a, _) => List (a)
            case Abs (a) => List (a)
            case Sqrt (a) => List (a)
            case Log (a) => List (a)
            case Log1p (a) => List (a)
            case Exp (a) => List (a)
            case Sin (a) => List (a)
            case Cos (a) => List (a)
            case Tanh (a) => List (a)
            case _ => Nil
    end children

    private def checkVar (v: String, cons: SymConstraints): List [String] =
        if cons.variables.contains (v) then Nil else List (s"unknown variable $v")

    private def checkRange (name: String, k: Int, lo: Int, hi: Int): List [String] =
        if k >= lo && k <= hi then Nil else List (s"$name requires value in [$lo, $hi], got $k")

end SymValidator


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymComplexity` object gives a simple expression complexity score.
 */
object SymComplexity:

    import SymExpr._

    def apply (expr: SymExpr): Int =
        expr match
            case Const (_) | Coef (_) | Var (_) => 1
            case Lag (_, _) | Diff (_, _) | SeasonalDiff (_, _) | RollMean (_, _) | RollStd (_, _) | Ewma (_, _) => 2
            case Add (a, b) => 1 + apply (a) + apply (b)
            case Sub (a, b) => 1 + apply (a) + apply (b)
            case Mul (a, b) => 1 + apply (a) + apply (b)
            case Div (a, b) => 1 + apply (a) + apply (b)
            case Pow (a, _) => 2 + apply (a)
            case Neg (a)    => 1 + apply (a)
            case Abs (a)    => 1 + apply (a)
            case Sqrt (a)   => 1 + apply (a)
            case Log (a)    => 1 + apply (a)
            case Log1p (a)  => 1 + apply (a)
            case Exp (a)    => 1 + apply (a)
            case Sin (a)    => 1 + apply (a)
            case Cos (a)    => 1 + apply (a)
            case Tanh (a)   => 1 + apply (a)
    end apply

end SymComplexity


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `symbolicExpressionTest` tests the symbolic DSL, parser, validator,
 *  evaluator, complexity scorer, and LaTeX renderer.
 *  > runMain scalation.theory.symbolicExpressionTest
 */
@main def symbolicExpressionTest (): Unit =

    import SymDSL._

    banner ("Symbolic DSL Test")

    val y     = VectorD (1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0)
    val cases = VectorD (2.0, 3.0, 4.0, 7.0, 9.0, 10.0, 13.0, 20.0)
    val frame = new VectorFrame (Map ("y" -> y, "cases" -> cases))

    val expr = c(0) * lag ("y", 1) + c(1) * log1p_ (lag ("cases", 2)) + c(2) * rollmean ("y", 3)
    val coef = VectorD (0.60, 0.25, 0.15)

    println (s"expr       = $expr")
    println (s"latex      = ${SymLatex (expr)}")
    println (s"complexity = ${SymComplexity (expr)}")
    println (s"value t=6  = ${SymEval (expr, frame, 6, coef)}")

    val dsl = "(add (mul c0 (lag y 1)) (mul c1 (log1p (lag cases 2))))"
    val parsed = SymParser.parse (dsl)
    println (s"parsed     = $parsed")
    println (s"latex      = ${SymLatex (parsed)}")

    val cons = SymConstraints (variables = Set ("y", "cases"), maxDepth = 6, maxLag = 8)
    println (s"valid      = ${SymValidator.validate (parsed, cons)}")

end symbolicExpressionTest

