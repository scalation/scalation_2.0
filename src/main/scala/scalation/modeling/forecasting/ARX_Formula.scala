
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Thu Jun 25 19:23:06 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Allows ARX* Models to be Entered using Formulas
 *           via `ARX_Formula`
 *
 *  A formula is a response variable and a list of predictor terms, e.g.,
 *
 *      y ~= y~(3,1) + log1p (hosp~1) + y~1 * hosp~1 + wave (freqs, 2)
 *
 *  Each `Term` (lagged variable, transformed term, power term, product term or
 *  wave term) can produce its own column of the data/input matrix (@see
 *  `Term.column`), with the first lag values padded as in `MakeMatrix4TS.makeMatrix4L`
 *  (@see `VectorD.lagv`), so `ARX_D.fromTerms` builds a direct forecasting model
 *  from any formula.
 *
 *  ONE GRAMMAR: the label of every term (@see `Term.label`) is written in the
 *  formula grammar that `FormulaParser` reads, e.g., y~3, log1p(y~1), pow(y~1, 1.5),
 *  y~1 * hosp~1, so the feature names printed by ScalaTion, the specification
 *  strings exchanged with an LLM (@see `ARX_Formula.formulaStr`) and the parser share one
 *  notation; `↑` and `↟` are Scala sugar that build the same pow/powr terms.
 *  @note    AI Assisted Code
 */

package scalation
package modeling
package forecasting

import scala.collection.mutable.{ArrayBuffer => VEC}
import scala.language.{dynamics, implicitConversions}
import scala.math.{max, Pi}
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

import TransformT._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Term` trait is for any term on the right hand side of a formula.
 */
trait Term:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the name of the underlying variable (the dataset key).
     */
    def name: String

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the DISPLAY label of this term in the formula grammar (name with lag,
     *  transform, power and interaction), used as the feature name and in
     *  specification strings (@see `ARX_Formula.formulaStr`).
     */
    def label: String = name

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the lag applied (the maximum lag for a product term).
     */
    def lag: Int

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the names of the dataset variables this term depends on.
     */
    def vars: Set [String] = Set (name)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the column of the data/input matrix for this term, one value per
     *  time point (the first lag values are padded, @see `VectorD.lagv`).
     *  @param dataset  the map of named data columns
     */
    def column (dataset: Map [String, VectorD]): VectorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply this term by another term to form an interaction (product) term.
     *  @param other  the other term
     */
    def * (other: Term): Term = ProductTerm (this, other)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply this term by each term of a lag range, e.g., hosp~1 * y~(3,1)
     *  gives the three interactions (as the grammar's range distribution does).
     *  @param ts  the terms to multiply by
     */
    def * (ts: Seq [Term]): Seq [Term] = ts.map (ProductTerm (this, _))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add another term to this term to start a list of terms.
     *  @param other  the other term
     */
    def + (other: Term): VEC [Term] = VEC (this, other)

end Term


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `FormulaVar` case class represents a variable at a given lag.
 *  @param name  the variable name
 *  @param lag   the lag applied (e.g., y~2 is lag 2), 0 => current value
 */
case class FormulaVar (name: String, lag: Int = 0) extends Term:

    require (lag >= 0, s"FormulaVar: lag $lag for variable '$name' must be >= 0")

    override def label: String = if lag == 0 then name else s"$name~$lag"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return this variable at lag l, e.g., y~2.
     *  @param l  the lag
     */
    def ~ (l: Int): FormulaVar = FormulaVar (name, l)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return this variable at each lag in the given inclusive range, e.g., y~(3,1) = y~3, y~2, y~1.
     *  @param range  the (maximum lag, minimum lag) pair
     */
    def ~ (range: (Int, Int)): Seq [FormulaVar] =
        val (maxLag, minLag) = range
        require (maxLag >= minLag, s"FormulaVar: lag range ($maxLag,$minLag) for '$name' must be decreasing")
        (maxLag to minLag by -1).map (l => FormulaVar (name, l))
    end ~

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make this variable the response of a formula whose predictors were gathered
     *  by a builder.
     *  @param builder  the builder holding the predictor terms
     */
    def ~= (builder: ARX_FormulaBuilder): ARX_Formula =
        ARX_Formula (this, ARX_Formula.distinct (builder.preds))
    end ~=

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the column of the dataset for this term, one value per time point
     *  (the first lag values are padded, @see `VectorD.lagv`).
     *  @param dataset  the map of named data columns
     */
    def column (dataset: Map [String, VectorD]): VectorD = dataset(name).lagv (lag)

end FormulaVar


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TransformedTerm` case class represents a term transformed by a unary
 *  function, e.g., log1p (x~1); its label is the grammar form op(source).
 *  @param source     the term being transformed
 *  @param transform  the scalar transformation function
 *  @param op         the function name (e.g., "log1p"), used in the label
 */
case class TransformedTerm (source: Term, transform: Double => Double, op: String)
     extends Term:

    def name: String = source.name

    override def label: String = s"$op(${source.label})"

    def lag: Int = source.lag

    override def vars: Set [String] = source.vars

    def column (dataset: Map [String, VectorD]): VectorD = source.column (dataset).map (transform)

end TransformedTerm


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PowerTerm` case class represents a term raised to a real power, plain
 *  x^p or sign-safe (@see `powr`); its label is the grammar form pow(source, p)
 *  or powr(source, p), so the exponent is part of the notation.
 *  @param source  the term being raised
 *  @param p       the exponent
 *  @param safe    whether the power is sign-safe: sign (x) |x|^r with r the nearby
 *                 rational with an odd denominator (@see `PowRForm`, `Rat.fromDouble3`)
 */
case class PowerTerm (source: Term, p: Double, safe: Boolean = false)
     extends Term:

    private val r = if safe then Rat.fromDouble3 (p).toDouble else p    // the exponent applied

    def name: String = source.name

    override def label: String = s"${if safe then "powr" else "pow"}(${source.label}, $p)"

    def lag: Int = source.lag

    override def vars: Set [String] = source.vars

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Apply the power to one value.
     *  @param x  the value
     */
    def f (x: Double): Double =
        if safe then math.signum (x) * math.pow (math.abs (x), r) else math.pow (x, r)
    end f

    def column (dataset: Map [String, VectorD]): VectorD = source.column (dataset).map (f)

end PowerTerm


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ProductTerm` case class represents an interaction term, e.g., x~1 * y~2.
 *  @param left   the left hand side term
 *  @param right  the right hand side term
 */
case class ProductTerm (left: Term, right: Term)
     extends Term:

    def name: String = left.name

    override def label: String = s"${left.label} * ${right.label}"

    def lag: Int = max (left.lag, right.lag)

    override def vars: Set [String] = left.vars ++ right.vars

    def column (dataset: Map [String, VectorD]): VectorD = left.column (dataset) * right.column (dataset)

end ProductTerm


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `WaveTerm` case class represents a seasonal term that is a function of the
 *  time index t (not of a data column): sin (w t + phase).  Use `wave` to build
 *  the sin/cos pair for a frequency, so the fitted coefficients capture any phase.
 *  @param w      the angular frequency (2 Pi times the frequency in cycles per time unit)
 *  @param phase  the phase: 0 for the sine component, Pi/2 for the cosine component
 *  @param idx    the 1-based rank of the frequency in the periodogram list it came
 *                from (0 => not from a list), so a formula can write wave(k)
 */
case class WaveTerm (w: Double, phase: Double, idx: Int = 0) extends Term:

    def name: String =
        val kind = if phase == 0.0 then "sin" else "cos"
        f"$kind(w=$w%.4f)"
    end name

    def lag: Int = 0                                                    // no history needed

    override def vars: Set [String] = Set ()                            // no data column

    def column (dataset: Map [String, VectorD]): VectorD =
        val m = dataset.head._2.dim                                     // all columns have the same length
        VectorD (for t <- 0 until m yield math.sin (w * t + phase))
    end column

end WaveTerm


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Vars` object turns an unquoted name (e.g., `Vars.y`) into a `FormulaVar`.
 */
object Vars extends Dynamic:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a variable for the given name (called by the compiler for `Vars.name`).
     *  @param name  the variable name
     */
    def selectDynamic (name: String): FormulaVar = FormulaVar (name)

end Vars


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Unary Functions that wrap a term (or each term in a sequence) into a `TransformedTerm`.
 *  Their names are the grammar's function names (@see `ARX_Formula.functions`).
 */
def log (t: Term): Term   = TransformedTerm (t, math.log, "log")
def log1p (t: Term): Term = TransformedTerm (t, math.log1p, "log1p")
def sqrt (t: Term): Term  = TransformedTerm (t, math.sqrt, "sqrt")
def sq (t: Term): Term    = TransformedTerm (t, x => x * x, "sq")
def dq (t: Term): Term    = TransformedTerm (t, x => DampenedQuad.f(x, Transform.hp("mx").toDouble), "dq")
def exp (t: Term): Term   = TransformedTerm (t, math.exp, "exp")
def expm1 (t: Term): Term = TransformedTerm (t, math.expm1, "expm1")
def sin (t: Term): Term   = TransformedTerm (t, math.sin, "sin")
def cos (t: Term): Term   = TransformedTerm (t, math.cos, "cos")
def tanh (t: Term): Term  = TransformedTerm (t, math.tanh, "tanh")

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Unary Functions that apply on each term in a sequence.
 */
def log (ts: Seq [Term]): Seq [Term]   = ts.map (log (_))
def log1p (ts: Seq [Term]): Seq [Term] = ts.map (log1p (_))
def sqrt (ts: Seq [Term]): Seq [Term]  = ts.map (sqrt (_))
def sq (ts: Seq [Term]): Seq [Term]    = ts.map (sq (_))
def dq (ts: Seq [Term]): Seq [Term]    = ts.map (dq (_))
def exp (ts: Seq [Term]): Seq [Term]   = ts.map (exp (_))
def expm1 (ts: Seq [Term]): Seq [Term] = ts.map (expm1 (_))
def sin (ts: Seq [Term]): Seq [Term]   = ts.map (sin (_))
def cos (ts: Seq [Term]): Seq [Term]   = ts.map (cos (_))
def tanh (ts: Seq [Term]): Seq [Term]  = ts.map (tanh (_))

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Raise a term (or each term in a sequence) to a real power, e.g., pow (y~1, 1.5).
 *  @param t  the term to raise
 *  @param p  the exponent
 */
def pow (t: Term, p: Double): Term = PowerTerm (t, p)
def pow (ts: Seq [Term], p: Double): Seq [Term] = ts.map (pow (_, p))

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Raise a term (or each term in a sequence) to a sign-safe power (@see `PowRForm`):
 *  x -> sign (x) |x|^r where r is the nearby rational with an odd denominator, so
 *  the result is real for negative x.
 *  @param t  the term to raise
 *  @param q  the power (converted to a nearby rational with an odd denominator)
 */
def powr (t: Term, q: Double): Term = PowerTerm (t, q, safe = true)
def powr (ts: Seq [Term], q: Double): Seq [Term] = ts.map (powr (_, q))

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Power operators on terms: ↑ (power) and ↟ (sign-safe power), e.g., y~1 ↑ 0.67.
 *  They are Scala sugar for `pow` and `powr` (same terms, same labels) and bind
 *  tighter than +, so no parentheses are needed in a formula.
 */
extension (t: Term)
    def ↑ (p: Double): Term = pow (t, p)
    def ↟ (q: Double): Term = powr (t, q)

extension (ts: Seq [Term])
    def ↑ (p: Double): Seq [Term] = ts.map (_ ↑ p)
    def ↟ (q: Double): Seq [Term] = ts.map (powr (_, q))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply each term of a lag range by another term, e.g., y~(3,1) * hosp~1
     *  gives the three interactions (as the grammar's range distribution does).
     *  @param other  the other term
     */
    def * (other: Term): Seq [Term] = ts.map (ProductTerm (_, other))

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Build the sin/cos pair of wave terms for a frequency given in cycles per time
 *  unit (as returned by `Periodogram.topKFrequencies`).
 *  @param freq  the frequency in cycles per time unit (0 < freq <= 0.5)
 *  @param idx   the 1-based rank of the frequency in its periodogram list (0 => none)
 */
def wave (freq: Double, idx: Int = 0): Seq [Term] =
    val w = _2Pi * freq
    Seq (WaveTerm (w, 0.0, idx), WaveTerm (w, Pi / 2.0, idx))
end wave

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Build the wave terms for the first k frequencies found by a `Periodogram`
 *  (the grammar's wave(k)).
 *  @param freqs  the frequencies in cycles per time unit
 *  @param k      the number of frequencies (sin/cos pairs) to include
 */
def wave (freqs: VectorD, k: Int): Seq [Term] =
    (0 until math.min (k, freqs.dim)).flatMap (i => wave (freqs(i), i + 1))
end wave

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Build a `TransformedTerm` from a `Transform`, using its scalar function f_.
 *  @param source  the term being transformed
 *  @param tf      the transform (e.g., Log1pForm (), PowForm ())
 *  @param label   the operation name used in the label
 */
def transformed (source: Term, tf: Transform, label: String): TransformedTerm =
    TransformedTerm (source, tf.f_, label)
end transformed


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Binding` case class names a derived series (@see `letIn`).
 *  @param name  the name of the new series
 *  @param defn  the term defining it from existing columns
 *  @param tf    the transform used (None if defn is not a transform), whose inverse
 *               applies when this binding is the response of the formula
 */
case class Binding (name: String, defn: Term, tf: Option [Transform] = None)

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Bind a name to a variable transformed by a `TransformT` (e.g., Log1p, Norm,
 *  MinMax), fit on the TRAINING slice of the variable's column (as in `rescale`).
 *  @param name     the name of the new series
 *  @param v        the variable to transform
 *  @param tFormT   the transform type
 *  @param label    the operation name used in the label (defaults to the type's name)
 *  @param dataset  the map of named data columns
 */
def bind (name: String, v: FormulaVar, tFormT: TransformT, label: String = null)
         (using dataset: Map [String, VectorD]): Binding =
    val col = dataset (v.name)
    val tf  = tFormT.form (col(0 until Model.trSize (col.dim)))
    val lbl = if label == null then tFormT.name.stripSuffix ("Form") else label
    Binding (name, transformed (v, tf, lbl), Some (tf))
end bind

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Evaluate the formula in body with the bound series added to the dataset (as in
 *  ML's let bindings in body).  If a binding is the formula's response, its
 *  transform is recorded in the formula (respTForm) for inverting forecasts.
 *  @param bindings  the bindings (name -> derived series)
 *  @param body      the formula, a context function of the extended dataset
 *  @param base      the map of named data columns to extend
 */
def letIn (bindings: Binding*)(body: Map [String, VectorD] ?=> ARX_Formula)
          (using base: Map [String, VectorD]): ARX_Formula =
    var extended = base
    for b <- bindings do
        if base.contains (b.name) then
            throw new IllegalArgumentException (s"letIn: binding '${b.name}' shadows an existing column")
        extended += b.name -> b.defn.column (base)
    end for

    val formula = body (using extended)
    bindings.find (_.name == formula.resp.name) match
    case Some (b) if b.tf.isDefined => formula.copy (respTForm = b.tf)
    case Some (b) => throw new IllegalArgumentException (s"letIn: response binding '${b.name}' has no transform")
    case None     => formula
end letIn


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Conversions that let a formula be written naturally:  a string is a variable;
 *  a term or a lag range starts the right hand side; + appends to a term list.
 */
//implicit def stringToFormulaVar (name: String): FormulaVar = FormulaVar (name)

implicit def termToBuilder (term: Term): ARX_FormulaBuilder = ARX_FormulaBuilder (Seq (term))

implicit def seqToBuilder (terms: Seq [Term]): ARX_FormulaBuilder = ARX_FormulaBuilder (terms)

extension (terms: VEC [Term])
    def + (other: Term): VEC [Term] = { terms += other; terms }

extension (lhs: FormulaVar)
    def ~= (rhs: Term): ARX_Formula = ARX_Formula (lhs, VEC (rhs))
    def ~= (rhs: VEC [Term]): ARX_Formula = ARX_Formula (lhs, ARX_Formula.distinct (rhs))


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARX_FormulaBuilder` case class gathers the predictor terms on the right
 *  hand side of a formula.
 *  @param preds  the predictor terms gathered so far
 */
case class ARX_FormulaBuilder (preds: Seq [Term] = Seq ()):

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Append a term, e.g., + hosp~1.
     *  @param term  the term to append
     */
    def + (term: Term): ARX_FormulaBuilder = ARX_FormulaBuilder (preds :+ term)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Append the terms of a lag range, e.g., + y~(3,1).
     *  @param terms  the terms to append
     */
    def + (terms: Seq [Term]): ARX_FormulaBuilder = ARX_FormulaBuilder (preds ++ terms)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply the last term gathered by another term, e.g., ... + y~1 * hosp~1.
     *  @param other  the other term
     */
    def * (other: Term): Term =
        require (preds.nonEmpty, "ARX_FormulaBuilder: no term to multiply")
        ProductTerm (preds.last, other)
    end *

end ARX_FormulaBuilder


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARX_Formula` case class represents a formula: a response variable, its
 *  predictor terms and, optionally, the transform type applied to every base
 *  column before the terms are evaluated (the grammar's TFORM directive).
 *  @param resp       the response variable
 *  @param preds      the predictor terms (distinct labels, @see `ARX_Formula.distinct`)
 *  @param tFormT     the transform type for every base column, fit on the training
 *                    slice by `ARX_D.fromTerms` (null => none), e.g., Log1p, Norm
 *  @param respTForm  the transform already applied to the response by `letIn` (None => none)
 */
case class ARX_Formula (resp: FormulaVar, preds: VEC [Term],
                        tFormT: TransformT = null,
                        respTForm: Option [Transform] = None):

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return this formula as a specification string in the grammar, e.g.,
     *  y ~= y~3 + y~1 + log1p(icu~1) + wave(2) TFORM: Log1p, so that
     *  FormulaParser.parseFormula (f.formulaStr, ...) rebuilds f.  Wave terms that came
     *  from a periodogram list (idx 1 .. k, the grammar's wave(k)) are written as
     *  wave(k); other wave terms keep their labels, which the parser does not read.
     */
    def formulaStr: String =
        val (waves, rest) = preds.partition (_.isInstanceOf [WaveTerm])
        val idx   = waves.map (_.asInstanceOf [WaveTerm].idx).distinct.sorted
        val wpart = if waves.isEmpty then Seq ()
                    else if idx.head >= 1 && idx == (1 to idx.last) then Seq (s"wave(${idx.last})")
                    else waves.map (_.label)
        val head  = s"${resp.name} ~= ${(rest.map (_.label) ++ wpart).mkString (" + ")}"
        if tFormT == null then head else s"$head TFORM: $tFormT"
    end formulaStr

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a canonical key for de-duplication (labels sorted, so that
     *  y~1 + hosp~2 and hosp~2 + y~1 collide as intended, plus the transform type).
     */
    def key: String =
        s"${resp.name} ~= ${preds.map (_.label).sorted.mkString (" + ")}" +
        (if tFormT == null then "" else s" TFORM: $tFormT")
    end key

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of predictor terms.
     */
    def nTerms: Int = preds.size

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the largest lag used by any term (0 if none).
     */
    def maxLag: Int = preds.map (_.lag).maxOption.getOrElse (0)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Fit an `ARX` recursive forecasting model with one column per term of this
     *  formula (@see `ARX.fromTerms`), in-sample (trained on every row).
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param tFormT   the transform type applied to every base column (defaults to
     *                  this formula's; null => none)
     *  @param trSize   the training slice the base-column transforms are fit on
     *                  (-1 => `Model.trSize`, @see `ARX.fromTerms`)
     *  @param scaled   whether the regression standardizes the term columns
     *  @param dataset  the map of named data columns
     */
    def fit (hh: Int = 1, tFormT: TransformT = this.tFormT, trSize: Int = -1, scaled: Boolean = false)
            (using dataset: Map [String, VectorD]): ARX =
        val mod = ARX.fromTerms (resp, preds, hh, tFormT = tFormT, trSize = trSize, scaled = scaled)
        mod.trainNtest ()()
        mod
    end fit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Fit an `ARX_D` direct forecasting model with one column per term of this
     *  formula (@see `ARX_D.fromTerms`), in-sample (trained on every row).
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param tFormT   the transform type applied to every base column (defaults to
     *                  this formula's; null => none)
     *  @param trSize   the training slice the base-column transforms are fit on
     *                  (-1 => `Model.trSize`, @see `ARX_D.fromTerms`)
     *  @param scaled   whether the regression standardizes the term columns
     *  @param dataset  the map of named data columns
     */
    def fit_D (hh: Int = 1, tFormT: TransformT = this.tFormT, trSize: Int = -1, scaled: Boolean = false)
              (using dataset: Map [String, VectorD]): ARX_D =
        val mod = ARX_D.fromTerms (resp, preds, hh, tFormT = tFormT, trSize = trSize, scaled = scaled)
        val (x, yy) = (mod.getX, mod.getYy)
        mod.trainNtest (x, yy)(x, yy)
        mod
    end fit_D

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the string representaion for this object.
     */
    override def toString: String = formulaStr 

end ARX_Formula


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARX_Formula` companion object holds the grammar's function registry and
 *  helpers shared by the DSL and `FormulaParser`.
 */
object ARX_Formula:

    /** The Unary Functions of the grammar, by name: adding one here adds it to the
     *  DSL, the parser and the grammar text at once.
     */
    val functions: Map [String, Term => Term] = Map ("log" -> log,
                                                     "log1p" -> log1p,
                                                     "sqrt" -> sqrt,
                                                     "sq" -> sq,
                                                     "dq" -> dq,
                                                     "exp" -> exp,
                                                     "expm1" -> expm1,
                                                     "sin" -> sin,
                                                     "cos" -> cos,
                                                     "tanh" -> tanh)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the grammar as text (for prompts and documentation).
     *  @param maxLag  the maximum lag allowed
     *  @param pMin    the minimum exponent allowed
     *  @param pMax    the maximum exponent allowed
     */
    def grammar (maxLag: Int = 8, pMin: Double = 0.2, pMax: Double = 3.0): String =

        s"""formula := y ~= element { + element } [ TFORM: Log1p | Norm ]
           |element := term | term * term                      (two factors, no wave inside)
           |term    := x~k | x~(h,l) | FN(term) | pow(term, p) | powr(term, p) | wave(k)
           |            x~k is x lagged k steps (1 <= k <= $maxLag), x~(h,l) is x~h, ..., x~l
           |            FN in {${functions.keys.toSeq.sorted.mkString (", ")}}, $pMin <= p <= $pMax
           |            wave(k) adds sin/cos pairs for the k strongest seasonal frequencies""".stripMargin
    end grammar

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Collapse duplicate terms (same label), which would make the design matrix
     *  singular, keeping the first occurrence in order.
     *  @param preds  the predictor terms (any sequence, e.g., a range or a `VEC`)
     */
    def distinct (preds: scala.collection.Seq [Term]): VEC [Term] = preds.distinctBy (_.label).to (VEC)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `TermData` case class holds a formula's dataset as the terms see it.
     *  @param data     the dataset with every base column transformed (when a tFormT was given)
     *  @param y        the (transformed) response column
     *  @param tForm_y  the transform applied to the response (null => none), for inverting forecasts
     *  @param trSize   the number of leading rows the transforms were fit on
     */
    case class TermData (data: Map [String, VectorD], y: VectorD, tForm_y: Transform, trSize: Int)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Throw an `IllegalArgumentException` for a formula that cannot be fit.
     *  @param msg  the reason
     */
    private def fail (msg: String): Nothing = throw new IllegalArgumentException (s"fromTerms: $msg")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check a formula against the dataset (variables present, lengths equal, lags
     *  inside the series) and transform every base column on its own TRAINING
     *  slice (as in `rescale`), before the terms are evaluated; the response's
     *  transform is kept for inverting forecasts (the regression itself never
     *  rectifies in a centered/standardized space, @see `MakeMatrix4TS.centered`).
     *  @param resp     the response/endogenous variable
     *  @param preds    the predictor terms from the right hand side of the formula
     *  @param tFormT   the transform type applied to every base column (null => none)
     *  @param trSize   the number of leading rows the transforms are fit on
     *                      (-1 => `Model.trSize (y.dim)`, as in `rescale`); pass the size
     *                      of the training window so its statistics never include
     *                      validation or testing rows
     *  @param bakcast  whether a backcasted value is prepended (not supported, must be false)
     *  @param dataset  the map of named data columns (given)
     */
    def prepare (resp: FormulaVar, preds: VEC [Term], tFormT: TransformT = null, trSize: Int = -1,
                 bakcast: Boolean = false)(using dataset: Map [String, VectorD]): TermData =
        if bakcast then fail ("bakcast not supported")
        if preds.isEmpty then fail ("no predictor terms (the mean model has no design matrix)")
        val y = dataset.getOrElse (resp.name, fail (s"response '${resp.name}' not in dataset"))
        for pred <- preds; nm <- pred.vars do
            if ! dataset.contains (nm) then fail (s"variable '$nm' not in dataset")
            if dataset(nm).dim != y.dim then fail (s"'$nm' length ${dataset(nm).dim} != ${y.dim}")
        for pred <- preds if pred.lag >= y.dim do fail (s"lag ${pred.lag} >= series length ${y.dim}")

        val tr_size = if trSize > 0 then trSize else Model.trSize (y.dim)
        if tr_size > y.dim then fail (s"trSize $tr_size > series length ${y.dim}")
        val tData   = if tFormT == null then dataset
                      else dataset.map ((nm, col) => nm -> tFormT.form (col(0 until tr_size)).f (col))
        val tForm_y = if tFormT == null then null else tFormT.form (y(0 until tr_size))
        TermData (tData, tData(resp.name), tForm_y, tr_size)
    end prepare

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the data/input matrix of a formula: one column per term, full length
     *  (first lag values padded, @see `VectorD.lagv`) and NO intercept column (the
     *  regression centers, so feature selection on it must use first = 0).  With
     *  check, a term whose column is not finite (e.g., log of 0, exp of a large
     *  value) is rejected here, before the regression sees it.
     *  @param preds  the predictor terms from the right hand side of the formula
     *  @param data   the dataset as the terms see it (@see `prepare`)
     *  @param check  whether to reject a column that is not finite (defaults to true)
     */
    def matrix (preds: VEC [Term], data: Map [String, VectorD], check: Boolean = true): MatrixD =
        val xy = MatrixD (preds.map (_.column (data))).ᵀ
        if check then
            for j <- preds.indices if xy(?, j).exists (v => v.isNaN || v.isInfinite) do
                fail (s"term '${preds(j).label}' is not finite (e.g., log of 0 or exp of a large value)")
        xy
    end matrix

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the feature names of a formula's matrix: the terms' labels.
     *  @param preds  the predictor terms from the right hand side of the formula
     */
    def names (preds: VEC [Term]): Array [String] = preds.map (_.label).toArray

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the hyper-parameters for a model built from a formula -- p and px the
     *  largest lags of the response and of the exogenous variables, spec = 0 (the
     *  given hparam is never modified) -- and the number of distinct exogenous
     *  variables in the terms.
     *  @param resp    the response/endogenous variable
     *  @param preds   the predictor terms from the right hand side of the formula
     *  @param hparam  the hyper-parameters to start from
     */
    def lagHP (resp: FormulaVar, preds: VEC [Term], hparam: HyperParameter): (HyperParameter, Int) =
        val pY    = preds.filter (_.vars == Set (resp.name)).map (_.lag).maxOption.getOrElse (0)
        val pX    = preds.filter (t => t.vars.nonEmpty && t.vars != Set (resp.name)).map (_.lag).maxOption.getOrElse (0)
        val n_exo = (preds.flatMap (_.vars).toSet - resp.name).size
        (hparam.updateReturn (("p", pY), ("px", pX), ("spec", 0)), n_exo)
    end lagHP

end ARX_Formula


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_FormulaTest` main function tests the `ARX_Formula` class on a toy
 *  dataset: a valid formula, and one using a column of the wrong length.
 *  > runMain scalation.modeling.forecasting.aRX_FormulaTest
 */
@main def aRX_FormulaTest (): Unit =

    given dataContext: Map [String, VectorD] = Map (
        "y" -> VectorD (12.0, 15.0, 14.0, 16.0, 18.0, 21.0, 19.0, 22.0),
        "x" -> VectorD (4.0,   5.2,  4.8,  5.5,  6.0,  6.2,  5.9,  6.5),
        "z" -> VectorD (1.0,   2.0,  3.0))                                // wrong length

    val y = Vars.y
    val x = Vars.x
    val z = Vars.z

    banner ("valid formula")
    val formula = y ~= y~1 + log (x~1) + sq (x~2) + y~1 * x~1
    println (s"formula   = $formula")

    val mod  = formula.fit ()
    println (s"ARX   parameter = ${mod.parameter}")
    val modD = formula.fit_D ()
    println (s"ARX_D parameter = ${modD.parameter}")
    assert ((mod.parameter - modD.parameter).norm < 1e-8, "ARX and ARX_D must agree at h = 1")

    banner ("formula using z (wrong length)")
    try
        (y ~= y~1 + z~1).fit ()
    catch
        case ex: IllegalArgumentException => println (s"rejected: ${ex.getMessage}")

end aRX_FormulaTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_FormulaTest2` main function tests the `ARX_Formula` class on the
 *  Covid-19 dataset, comparing the formula's exact model (one column per term)
 *  with the classical `ARX_D` built from contiguous lags, on a formula with a
 *  skipped lag and an interaction term.
 *  > runMain scalation.modeling.forecasting.aRX_FormulaTest2
 */
@main def aRX_FormulaTest2 (): Unit =

    import Example_Covid._

    val exoColumns = Array ("icu_patients", "hosp_patients")
    val (xe, y_)   = clip (loadData (exoColumns))

    given dataContext: Map [String, VectorD] = Map (
        "y"             -> y_,
        "icu_patients"  -> xe.col(0),
        "hosp_patients" -> xe.col(1),
        "positive_rate" -> xe.col(2))

    val y     = Vars.y
    val icu   = Vars.icu_patients
//  val hosp  = Vars.hosp_patients
    val pRate = Vars.positive_rate

    val hh = 6
    banner ("formula: lags 4, 3, 2 of y, hosp~1, the interaction y~1 * hosp~1, icu~2")
//  val formula = y ~= y~(4,2) + hosp~1 + (y~1 * hosp~1) + icu~2
    val formula = y ~= y~1 + icu~2 + icu~1 + pRate~2 + pRate~1

    val modF = formula.fit_D (hh)
    println (s"modelName = ${modF.modelName}")
    println (s"fname     = ${stringOf (modF.getFname)}")
    println (s"parameter = ${modF.parameter}")
    modF.forecastAll ()
    modF.diagnoseAll (modF.getY, modF.getYf)
    banner("Roll Validate")
    modF.tnT_Test()

    banner ("the same formula as a recursive ARX (fit)")
    val modR = formula.fit (hh)
    println (s"modelName = ${modR.modelName}")
    println (s"parameter = ${modR.parameter}")
    modR.tnT_Test ()

end aRX_FormulaTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_FormulaTest3` main function tests the `ARX_Formula` class on the
 *  Covid-19 dataset, writing `ARX_SR_D`'s fEndo/fExo transform sets as explicit
 *  terms, e.g., fEndo = LSET (PowForm ()) with p = 3 becomes y~(3,1) ↑ 1.8.
 *  > runMain scalation.modeling.forecasting.aRX_FormulaTest3
 */
@main def aRX_FormulaTest3 (): Unit =

    import Example_Covid._

    val exoColumns = Array ("icu_patients", "hosp_patients")
    val (xe, y_)   = clip (loadData (exoColumns))

    given dataContext: Map [String, VectorD] = Map (
        "y"             -> y_,
        "icu_patients"  -> xe.col(0),
        "hosp_patients" -> xe.col(1))

    val y    = Vars.y
    val icu  = Vars.icu_patients
    val hosp = Vars.hosp_patients

    def show (mod: ARX_D): Unit =
        println (s"modelName = ${mod.modelName}")
        println (s"fname     = ${stringOf (mod.getFname)}")
        println (s"parameter = ${mod.parameter}")
        mod.forecastAll ()
        mod.diagnoseAll (mod.getY, mod.getYf)
    end show

    banner ("endogenous transforms (fEndo)")
    show ((y ~= y~(3,1) + y~(3,1) ↑ 0.67).fit_D ())

    banner ("endogenous and per-variable exogenous transforms (fEndo + fExo)")
    show ((y ~= y~(3,1) + y~(3,1) ↑ 1.8 + hosp~(2,1) + hosp~(2,1) ↑ 1.8 + icu~2 + sq (icu~2)).fit_D ())

    banner ("interaction of a transformed endogenous lag with an exogenous lag")
    show ((y ~= y~(2,1) + y~(2,1) ↑ 1.8 + hosp~1 + (log1p (y~1) * hosp~1)).fit_D ())

end aRX_FormulaTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_FormulaTest4` main function tests the `ARX_Formula` class on the
 *  Covid-19 dataset, using `letIn` and `bind` to model a transformed response.
 *  > runMain scalation.modeling.forecasting.aRX_FormulaTest4
 */
@main def aRX_FormulaTest4 (): Unit =

    import Example_Covid._

    val exoColumns = Array ("icu_patients", "hosp_patients")
    val (xe, y_)   = clip (loadData (exoColumns))

    given dataContext: Map [String, VectorD] = Map (
        "y"             -> y_,
        "icu_patients"  -> xe.col(0),
        "hosp_patients" -> xe.col(1))

    val u    = Vars.u                                                   // u = log1p (y), bound below
    val y    = Vars.y
    val hosp = Vars.hosp_patients
    val icu  = Vars.icu_patients

    val hh = 1
    var mod: ARX_D = null
    letIn (bind ("u", y, Log1p)) {
        val f = u ~= u~6 + u~6 ↑ 1.1 + hosp~4 + hosp~4 ↑ 1.1 + icu~4
        mod   = f.fit_D (hh)                                            // fit in the extended context
        f
    }

    println (s"modelName = ${mod.modelName}")
    println (s"fname     = ${stringOf (mod.getFname)}")
    println (s"parameter = ${mod.parameter}")
    mod.forecastAll ()
    mod.diagnoseAll (mod.getY, mod.getYf)

end aRX_FormulaTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_FormulaTest5` main function tests the `ARX_Formula` class on the
 *  Covid-19 dataset with seasonal wave terms: the dominant frequencies are found
 *  by a `Periodogram` on the training slice, then sin/cos pairs are added to the
 *  formula, followed by feature selection and rolling validation.
 *  > runMain scalation.modeling.forecasting.aRX_FormulaTest5
 */
@main def aRX_FormulaTest5 (): Unit =

    import Example_Covid._

    val exoColumns = Array ("icu_patients", "hosp_patients")
    val (xe, y_)   = clip (loadData (exoColumns))

    given dataContext: Map [String, VectorD] = Map (
        "y"    -> y_,
        "icu"  -> xe.col(0),
        "hosp" -> xe.col(1))

    val y    = Vars.y
//  val hosp = Vars.hosp
//  val icu  = Vars.icu

    val hh      = 6
    val tr_size = Model.trSize (y_.dim)
    val pg      = new Periodogram (y_(0 until tr_size), "new_deaths")   // training slice only
    val freqs   = pg.topKFrequencies (pg.amplitudes, 3)
    println (s"frequencies = $freqs")
    println (s"periods     = ${pg.topKPeriods (pg.amplitudes, 3)}")

    val mx = y_.max * 0.6
    Transform.hp("mx") = mx

    banner ("lags + exogenous + top-2 seasonal sin/cos pairs")
//  val formula = y ~= y~(5,1) + log (y~2) + y~(5,1) ↑ 1.5 + hosp~1 + icu~(2,1) + wave (freqs, 2)
    val formula = y ~= y~(5,1) + y~(5,1)↟0.2 + dq (y~(5,1))        // ARX_SR_D_5_0_0

    println (s"formula = $formula")
//  val mod = formula.fit_D (hh)
    val mod = formula.fit_D (hh, tFormT = MinMax)

    println (s"modelName = ${mod.modelName}")
    println (s"fname     = ${stringOf (mod.getFname)}")
    println (s"parameter = ${mod.parameter}")
    mod.tnT_Test ()

    import SelectionTech._
    val tech = Forward
//  val tech = Backward
//  val tech = Beam
//  val tech = CEM

    for h <- 1 to hh do
        banner (s"Feature Selection using $tech for horizon $h")
        val (colsArr, rSqArr) = mod.selectFeatures (tech, "none", first = 0, h = h)   // R^2, R^2 bar, sMAPE, sMAPEC
        mod.reportSelection (colsArr, rSqArr, inSample = false)         // TnT of the multi-horizon model on horizon h's columns
    end for

    banner (s"Feature Selection using $tech over all horizons")
    val (colsArr, rSqArr) = mod.selectFeaturesPerH (tech, "none", first = 0)
    mod.reportSelection (colsArr, rSqArr, inSample = false)             // k, features, trace per horizon; TnT of the hh models

end aRX_FormulaTest5


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_FormulaTest6` main function tests the `ARX_Formula` class on the
 *  Covid-19 dataset with feature selection for EVERY horizon (`featureSelection`):
 *  starting from lags, exogenous variables and seasonal wave terms, the selected
 *  features and the rolling-validation QoF are reported per horizon.
 *  > runMain scalation.modeling.forecasting.aRX_FormulaTest6
 */
@main def aRX_FormulaTest6 (): Unit =

    import Example_Covid._

    val exoColumns = Array ("icu_patients", "hosp_patients")
    val (xe, y_)   = clip (loadData (exoColumns))

    given dataContext: Map [String, VectorD] = Map (
        "y"    -> y_,
        "icu"  -> xe.col(0),
        "hosp" -> xe.col(1))

    val y    = Vars.y
    val hosp = Vars.hosp
    val icu  = Vars.icu

    val hh      = 6
    val tr_size = Model.trSize (y_.dim)
    val pg      = new Periodogram (y_(0 until tr_size), "new_deaths")   // training slice only
    val freqs   = pg.topKFrequencies (pg.amplitudes, 3)
    println (s"frequencies = $freqs")

    banner ("lags + exogenous + top-2 seasonal sin/cos pairs")
    val formula = y ~= y~(5,1) + hosp~1 + icu~2 + wave (freqs, 2)
//  val formula = y ~= y~(6,1) + y~(6,1)↑1.1 + hosp~(4, 1) + hosp~(4, 1)↑1.12 + icu~(4,1)
    val mod     = formula.fit_D (hh)
    println (s"modelName = ${mod.modelName}")
    println (s"fname     = ${stringOf (mod.getFname)}")
    println (s"parameter = ${mod.parameter}")

    import SelectionTech._
//  val tech = Forward
    val tech = Backward
//  val tech = Beam
    banner (s"feature selection for each horizon using $tech")
    val (colsArr, rSqArr) = mod.selectFeaturesPerH (tech, "none", first = 0)
    mod.reportSelection (colsArr, rSqArr, inSample = false)             // k, features, trace per horizon; TnT of the hh models

end aRX_FormulaTest6


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_FormulaTest7` main function tests the `ARX_Formula` class for a
 *  Two-Stage Linear (2SL, Durbin / Hannan-Rissanen) approximation of an ARMA model:
 *  stage 1 fits a high-order AR model to estimate the shocks; stage 2 includes the
 *  estimated shocks as a variable in the formula.
 *  > runMain scalation.modeling.forecasting.aRX_FormulaTest7
 */
@main def aRX_FormulaTest7 (): Unit =

    import Example_Covid._

    val exoColumns = Array ("icu_patients", "hosp_patients")
    val (_, y_)    = clip (loadData (exoColumns))

    val hh = 6
    val p  = 4
    val q  = 1
    val L  = p + q + 2                                                  // order of the stage 1 AR model

    banner (s"stage 1: AR ($L) to estimate the shocks")
    AR.hp("p") = L
    val mod1 = new AR (y_, 1)
    mod1.trainNtest ()()
    val e_sh_ = (y_ - mod1.predictAll ()).shift (1)                     // estimated shocks
    e_sh_(1)  = 0.0                                                     // no prediction at t = 0
    println (s"e_sh_ = $e_sh_")

    given dataContext: Map [String, VectorD] = Map (
        "y"    -> y_,
        "e_sh" -> e_sh_)

    val y    = Vars.y
    val e_sh = Vars.e_sh

    banner (s"stage 2: AR ($p) lags plus the current shock")
    val formula = y ~= y~(p,1) + e_sh~0
    val mod2    = formula.fit_D (hh)
    mod2.forecastAll ()
    mod2.diagnoseAll (mod2.getY, mod2.getYf)

end aRX_FormulaTest7


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_FormulaTest8` main function tests the grammar side of `ARX_Formula`:
 *  labels are written in the formula grammar (pow(y~1, 1.5), y~1 * hosp~1), ↑ and
 *  pow build the same term, a lag range distributes over an interaction, duplicate
 *  terms collapse, wave terms from a periodogram list serialize as wave(k), and
 *  `key` is order-free.  (The parse/serialize round trip is `formulaParserTest2`.)
 *  > runMain scalation.modeling.forecasting.aRX_FormulaTest8
 */
@main def aRX_FormulaTest8 (): Unit =

    val y     = Vars.y
    val hosp  = Vars.hosp
    val freqs = VectorD (0.13043, 0.06521)

    banner ("labels in grammar form")
    val f1 = y ~= y~(3,1) + log1p (hosp~1) + pow (y~1, 1.5) + y~1 ↑ 1.5 + powr (hosp~2, 0.67) + y~1 * hosp~1
    println (s"f1.formulaStr = ${f1.formulaStr}")
    println (s"labels        = ${f1.preds.map (_.label)}")
    assert (f1.preds.map (_.label) == Seq ("y~3", "y~2", "y~1", "log1p(hosp~1)", "pow(y~1, 1.5)",
                                            "powr(hosp~2, 0.67)", "y~1 * hosp~1"),
            "labels are not in grammar form (or ↑ did not collapse with pow)")

    banner ("a lag range distributes over an interaction")
    val f2 = y ~= y~(3,1) * hosp~1 + hosp~1 * y~(2,1)
    println (s"f2.formulaStr = ${f2.formulaStr}")
    assert (f2.nTerms == 5 && f2.preds(0).label == "y~3 * hosp~1" && f2.preds(4).label == "hosp~1 * y~1")

    banner ("wave terms from a periodogram list serialize as wave(k); key and formulaStr are order-free")
    val f3 = y ~= y~1 + wave (freqs, 2)
    val f4 = y ~= wave (freqs, 2) + y~1
    println (s"f3.formulaStr = ${f3.formulaStr}")
    println (s"f3.key        = ${f3.key}")
    assert (f3.formulaStr == "y ~= y~1 + wave(2)" && f3.key == f4.key && f3.formulaStr == f4.formulaStr)
    val f5 = (y ~= y~1 + wave (0.25)).copy (tFormT = Log1p)               // a wave not from a list keeps its labels
    println (s"f5.formulaStr = ${f5.formulaStr}")
    assert (f5.formulaStr.endsWith ("TFORM: Log1p") && f5.formulaStr.contains ("sin(w="))

    banner ("grammar text from the function registry")
    println (ARX_Formula.grammar ())
    assert (ARX_Formula.functions.keySet == Set ("log", "log1p", "sqrt", "sq", "dq", "exp", "expm1", "sin", "cos", "tanh"))

end aRX_FormulaTest8

