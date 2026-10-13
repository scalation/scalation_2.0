
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Sun Oct  4 12:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Formulas with NONLINEAR Parameters for `NARX_SR_D`
 *           via `NARX_Formula` (extends `ARX_Formula`)
 *           AI Assisted Code
 *
 *  An `ARX_Formula` fixes every number inside a term: pow (y~1, 1.5) is y^1.5.  Here a
 *  number may be a free PARAMETER, created by its ROLE (@see `ParRole`), e.g.,
 *
 *      y ~= y~(3,1) + y~(3,1) ↑ power () + log (shift () + hosp~1) + exp (rate () * hosp~1)
 *
 *  that `fitNL_D` fits on the training rows with the nonlinear optimizer of `NARX_SR_D`
 *  (`OPTIMIZER_NL`, Variable Projection by default), while the coefficients of the
 *  terms stay linear and are fitted by the regression.  The role gives a parameter its
 *  start value and bounds: unit-free roles (exponent, frequency, decay) from the table
 *  `Par.defaults`, the others (shift, rate, threshold, width, half-saturation, cap) from
 *  the TRAINING range of the column they act on, when the column is first evaluated;
 *  any of the three numbers may be given instead, e.g., power (lb = 1.1, ub = 2.5).
 *  A parameter object used in several terms (e.g., the exponent of a lag range) is
 *  SHARED by them, as the parameters of a transform are shared by its lags in
 *  `NARX_SR_D.rescale`.
 *  A formula written with `~=` is an `ARX_Formula`: `fit_D` gives the `ARX_D` model with
 *  the parameters at their current (start) values (`fit` the recursive `ARX`), `fitNL_D`
 *  the `NARX_SR_D` model with the fitted ones (@see `NARX_Formula`).
 *
 *  TERMS with free parameters: pow (power), shift and rate inside any function,
 *  wave (frequency), and sat (half-saturation), hinge (threshold), trans (threshold,
 *  width), ema (decay), dq (cap) -- @see the term classes below.
 *
 *  GRAMMAR: a free parameter is written ?, or ?w0 once its start value is known, e.g.,
 *  pow(y~1, ?1.500), log(? + icu~1), exp(? * icu~1), sat(icu~1, ?), sin(f=?0.1000).
 *  After `fitNL_D` the labels show the fitted values in the plain grammar, e.g.,
 *  pow(y~1, 1.995), so a fitted formula can be re-entered as an `ARX_Formula`
 *  (@see `ARX_Formula.formulaStr`).
 */

package scalation
package modeling
package forecasting
package nonlinear

import scala.collection.mutable.{ArrayBuffer => VEC}
import scala.language.implicitConversions
import scala.math.{abs, max, Pi}
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ParRole` enum lists the roles a free parameter can play; the role decides
 *  its default start value and bounds (@see `Par.resolve`).
 *  Free     fully given by the user (`par`), no defaults
 *  Shift    x + s inside log, pow, ...: keeps the argument positive on the training rows
 *  Phase    x + s inside sin or cos: in [-Pi, Pi]
 *  Rate     a x inside exp, sin, cos: 1 / max |x| and [0.01, 20] / max |x| (no overflow)
 *  Power    the exponent of pow: table (1.5 in [1.1, 4])
 *  Freq     the frequency of wave in cycles per time unit: table (0.1 in [0.01, 0.5])
 *  Decay    the decay of ema in (0, 1): table (0.5 in [0.01, 0.99])
 *  Thresh   the threshold of hinge and trans: the middle of the training range, in it
 *  Width    the width of trans: a tenth of the training range, in [1/100, 1] of it
 *  Half     the half-saturation of sat: the middle of the training range, in [1/100, 10] of it
 *  Cap      the cap of dq: half the largest |x|, in [0.05, 5] of it
 */
enum ParRole:
    case Free, Shift, Phase, Rate, Power, Freq, Decay, Thresh, Width, Half, Cap
end ParRole

import ParRole._


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Par` class is a free nonlinear parameter of a formula term: a role, a start
 *  value, bounds, and the current value, which the optimizer sets while fitting and
 *  which holds the fitted value afterwards.  A start value or bound left unset (NaN)
 *  is RESOLVED from the role: at once for the unit-free roles, from the training
 *  range of the column the parameter acts on for the others (@see `resolve`).
 *  Two `Par` objects are distinct parameters even with the same numbers (identity, not
 *  equality); one object used in several terms is one shared parameter.
 *  @param role  the role of the parameter (@see `ParRole`)
 *  @param w0_   the start value (NaN => from the role)
 *  @param lb_   the lower bound (NaN => from the role, -infinity for none); lb == ub fixes the parameter
 *  @param ub_   the upper bound (NaN => from the role, +infinity for none)
 */
class Par (val role: ParRole, w0_ : Double = Double.NaN, lb_ : Double = Double.NaN, ub_ : Double = Double.NaN):

    private var w0v     = w0_                                             // the start value
    private var lbv     = lb_                                             // the lower bound
    private var ubv     = ub_                                             // the upper bound
    private var value_  = w0_                                             // the current value
    private var fitted_ = false                                           // whether `set` has been called (plain label)

    if role == Free then
        require (resolved, s"Par: a Free parameter needs its start value and bounds")
    else if Par.defaults.contains (role) then
        val (w, l, u) = Par.defaults (role)                               // unit-free role: resolve from the table
        fill (w, l, u)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the start value, the lower bound and the upper bound (NaN until resolved).
     */
    def w0: Double = w0v
    def lb: Double = lbv
    def ub: Double = ubv

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether the start value and the bounds are all known.
     */
    def resolved: Boolean = ! (w0v.isNaN || lbv.isNaN || ubv.isNaN)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the current value (the start value, or the fitted value after `set`).
     */
    def value: Double = value_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether this parameter is free: its bounds differ, or are still to be
     *  resolved from the data (only equal bounds make a constant).
     */
    def free: Boolean = if resolved then lbv < ubv else true            // unresolved => free

// Note: `fixed` (equal bounds: a constant written in a parameter slot), `fresh` (a copy with the
//       numbers as given, for a formula built from another formula's terms, @see `NARX_Formula.fresh`)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether this parameter is fixed (resolved, equal bounds): a constant.
     */
    def fixed: Boolean = resolved && lbv == ubv

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a new parameter with the same role and the numbers as given (unset
     *  ones resolve again), independent of this one.
     */
    def fresh: Par = new Par (role, w0_, lb_, ub_)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Fill the unset start value and bounds with the given ones, keep the start value
     *  inside the bounds and make it the current value (unless already fitted).
     *  @param w  the default start value
     *  @param l  the default lower bound
     *  @param u  the default upper bound
     */
    private def fill (w: Double, l: Double, u: Double): Unit =
        if lbv.isNaN then lbv = l
        if ubv.isNaN then ubv = u
        if w0v.isNaN then w0v = w
        require (lbv <= ubv, s"Par ($role): lower bound $lbv > upper bound $ubv")
        w0v = max (lbv, math.min (ubv, w0v))                              // keep the start inside the box
        if ! fitted_ then value_ = w0v
    end fill

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Resolve the unset start value and bounds from the role and the column x the
     *  parameter acts on, using its training rows (@see `ParRole` for the rules);
     *  no effect when already resolved.
     *  @param x       the column of the term the parameter is applied to
     *  @param trSize  the number of training rows (-1 => `Model.trSize (x.dim)`)
     */
    def resolve (x: VectorD, trSize: Int = -1): Unit =
        if ! resolved then
            val m   = if trSize > 0 then math.min (trSize, x.dim) else Model.trSize (x.dim)
            val xt  = x(0 until m)
            val (lo, hi) = (xt.min, xt.max)
            val rng = if hi > lo then hi - lo else max (abs (hi), 1.0)
            val big = if max (abs (lo), abs (hi)) > 0.0 then max (abs (lo), abs (hi)) else 1.0
            role match
            case Shift  => val l = -lo + Par.SHIFT_MARGIN * rng            // argument stays positive
                           fill (max (0.0, l), l, max (hi, l + rng))
            case Phase  => fill (0.0, -Pi, Pi)
            case Rate   => fill (1.0 / big, 0.01 / big, 20.0 / big)
            case Thresh => fill (0.5 * (lo + hi), lo, hi)
            case Width  => fill (0.1 * rng, 0.01 * rng, rng)
            case Half   => fill (max (0.5 * (lo + hi), 0.01 * rng), 0.01 * rng, 10.0 * rng)
            case Cap    => fill (0.5 * big, 0.05 * big, 5.0 * big)
            case _      => fill (0.0, Double.NegativeInfinity, Double.PositiveInfinity)
        end if
    end resolve

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the current value during the search (the label keeps its ? marker).
     *  @param v  the value
     */
    def update (v: Double): Unit = value_ = v

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the fitted value: from now on the label shows it in the plain grammar.
     *  @param v  the fitted value
     */
    def set (v: Double): Unit = { value_ = v; fitted_ = true }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return to the start value (before a new fit).
     */
    def reset (): Unit = { value_ = w0v; fitted_ = false }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the label: ? while unresolved, ?w (w the start value) while free, the
     *  plain value once fitted (or when fixed) -- in the `Par.LabelMode.Value` mode;
     *  in the `Slot` mode a free parameter is always ?, in the `SlotValue` mode ?w with
     *  its current value (@see `Par.withMode`).
     */
    def label: String =
        // the label modes `Slot` (canonical keys) and `SlotValue` (proposals shown to a proposer)
        Par.mode match
        case Par.LabelMode.Slot      => if fixed then Par.fmt (value_) else "?"
        case Par.LabelMode.SlotValue => if fixed then Par.fmt (value_)
                                        else if value_.isNaN then "?" else "?" + Par.fmt (value_)
        case Par.LabelMode.Value     =>
            if fitted_ || fixed then Par.fmt (value_)
            else if value_.isNaN then "?"
            else "?" + Par.fmt (value_)
    end label

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Shift a term, or each term of a lag range, by this parameter: s + x, e.g.,
     *  log (shift () + icu~(4,1)) (the terms share the parameter).
     *  @param t   the term to shift
     *  @param ts  the terms to shift
     */
    def + (t: Term): Term = ShiftTerm (t, this)
    def + (ts: Seq [Term]): Seq [Term] = ts.map (ShiftTerm (_, this))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Scale a term, or each term of a lag range, by this parameter: a x, e.g.,
     *  exp (rate () * icu~1) (the terms share the parameter).
     *  @param t   the term to scale
     *  @param ts  the terms to scale
     */
    def * (t: Term): Term = ScaleTerm (t, this)
    def * (ts: Seq [Term]): Seq [Term] = ts.map (ScaleTerm (_, this))

    override def toString: String = label

end Par


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Par` companion object holds the default (start, lower, upper) of the
 *  unit-free roles -- settable, e.g., Par.defaults += Power -> (1.5, 1.1, 2.5) --
 *  the margin of the shift rule, and the label format.
 */
object Par:

    /** The defaults of the unit-free roles: role -> (start value, lower bound, upper bound).
     */
    var defaults: Map [ParRole, (Double, Double, Double)] = Map (
        Power -> (1.5, 1.1, 4.0),
        Freq  -> (0.1, 0.01, 0.5),
        Decay -> (0.5, 0.01, 0.99))

    /** The lower bound of a shift keeps the argument at least this fraction of the
     *  training range above zero.
     */
    var SHIFT_MARGIN = 0.01

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Format a parameter value for a label (4 significant digits).
     *  @param v  the value
     */
    def fmt (v: Double): String = f"$v%.4g"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `LabelMode` enum says how a free parameter is written in a label:
     *  Label modes, so that a formula's key never depends on the state of its parameters.
     *  `Value` -- ?w0 before the fit, the fitted value after it (the default);
     *  `Slot`  -- ? always (canonical keys: formulas that differ only in start or
     *             fitted values collide); `SlotValue` -- ?w with the current value
     *             (a fitted equation shown back to a proposer as a template).
     */
    enum LabelMode:
        case Value, Slot, SlotValue
    end LabelMode

    private var mode_ : LabelMode = LabelMode.Value                       // the current label mode

    def mode: LabelMode = mode_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Evaluate the body with the labels of free parameters written in the given mode.
     *  @param m     the label mode
     *  @param body  the computation (e.g., a key or a formula string)
     */
    def withMode [T] (m: LabelMode)(body: => T): T =
        val old = mode_
        mode_ = m
        try body finally mode_ = old
    end withMode

end Par

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Create a free parameter by its role.  Every number may be left out to take the
 *  role's default (@see `ParRole`), e.g., power (), power (1.3), power (lb = 1.1, ub = 2.5);
 *  `par` gives a parameter with no role, so all its numbers must be given.
 *  @param w0  the start value
 *  @param lb  the lower bound
 *  @param ub  the upper bound
 */
def par (w0: Double, lb: Double = Double.NegativeInfinity, ub: Double = Double.PositiveInfinity): Par =
    new Par (Free, w0, lb, ub)
end par

def shift  (w0: Double = Double.NaN, lb: Double = Double.NaN, ub: Double = Double.NaN): Par = new Par (Shift, w0, lb, ub)
def phase  (w0: Double = Double.NaN, lb: Double = Double.NaN, ub: Double = Double.NaN): Par = new Par (Phase, w0, lb, ub)
def rate   (w0: Double = Double.NaN, lb: Double = Double.NaN, ub: Double = Double.NaN): Par = new Par (Rate, w0, lb, ub)
def power  (w0: Double = Double.NaN, lb: Double = Double.NaN, ub: Double = Double.NaN): Par = new Par (Power, w0, lb, ub)
def freq   (w0: Double = Double.NaN, lb: Double = Double.NaN, ub: Double = Double.NaN): Par = new Par (Freq, w0, lb, ub)
def decay  (w0: Double = Double.NaN, lb: Double = Double.NaN, ub: Double = Double.NaN): Par = new Par (Decay, w0, lb, ub)
def thresh (w0: Double = Double.NaN, lb: Double = Double.NaN, ub: Double = Double.NaN): Par = new Par (Thresh, w0, lb, ub)
def width  (w0: Double = Double.NaN, lb: Double = Double.NaN, ub: Double = Double.NaN): Par = new Par (Width, w0, lb, ub)
def half   (w0: Double = Double.NaN, lb: Double = Double.NaN, ub: Double = Double.NaN): Par = new Par (Half, w0, lb, ub)
def cap    (w0: Double = Double.NaN, lb: Double = Double.NaN, ub: Double = Double.NaN): Par = new Par (Cap, w0, lb, ub)


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ParTerm` trait is for a term built from a source term and free parameters.
 *  Evaluating its column resolves the parameters that are still unset from the
 *  source column (@see `Par.resolve`).
 */
trait ParTerm extends Term:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the term the parameters act on.
     */
    def source: Term

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the free parameters of this term (not of its source).
     */
    def pars: Seq [Par]

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the column from the source column and the parameter values.
     *  @param x  the source column
     */
    def apply (x: VectorD): VectorD

    def name: String = source.name

    def lag: Int = source.lag

    override def vars: Set [String] = source.vars

    def column (dataset: Map [String, VectorD]): VectorD =
        val x = source.column (dataset)
        for p <- pars do p.resolve (x)
        apply (x)
    end column

end ParTerm


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ShiftTerm` case class shifts a term by a free parameter, s + x, meant for
 *  use inside a nonlinear function, e.g., log (shift () + icu~1); alone, or inside
 *  exp, the shift is covered by the intercept or the coefficient
 *  (@see `NARX_Formula.redundant`).  Its label is s + x in the grammar, e.g.,
 *  log(?1.000 + icu~1).
 *  @param source  the term being shifted
 *  @param s       the shift parameter
 */
case class ShiftTerm (source: Term, s: Par) extends ParTerm:

    override def label: String = s"${s.label} + ${source.label}"

    def pars: Seq [Par] = Seq (s)

    def apply (x: VectorD): VectorD = x + s.value

end ShiftTerm


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ScaleTerm` case class scales a term by a free parameter, a x, meant for use
 *  inside exp, sin or cos, e.g., exp (rate () * icu~1) (a rate); alone, or inside
 *  log or pow, the scale is covered by the coefficient (@see `NARX_Formula.redundant`).
 *  Its label is a * x in the grammar, e.g., exp(?0.1000 * icu~1).
 *  @param source  the term being scaled
 *  @param a       the scale parameter
 */
case class ScaleTerm (source: Term, a: Par) extends ParTerm:

    override def label: String = s"${a.label} * ${source.label}"

    def pars: Seq [Par] = Seq (a)

    def apply (x: VectorD): VectorD = x * a.value

end ScaleTerm


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ParPowerTerm` case class raises a term to a free power, x^p, e.g.,
 *  pow (y~1, power ()); its label is pow(source, ?p) and, once fitted, pow(source, p)
 *  as for `PowerTerm`.  The source may be shifted, (s + x)^p.
 *  @param source  the term being raised
 *  @param p       the exponent parameter
 */
case class ParPowerTerm (source: Term, p: Par) extends ParTerm:

    override def label: String = s"pow(${source.label}, ${p.label})"

    def pars: Seq [Par] = Seq (p)

    def apply (x: VectorD): VectorD = x.map (math.pow (_, p.value))

end ParPowerTerm


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SatTerm` case class is a SATURATING term, x / (x + k): linear for small x
 *  and flat for large x, with the half-saturation k free, e.g., sat (icu~1, half ()).
 *  @param source  the term being saturated (non-negative values)
 *  @param k       the half-saturation parameter (positive)
 */
case class SatTerm (source: Term, k: Par) extends ParTerm:

    override def label: String = s"sat(${source.label}, ${k.label})"

    def pars: Seq [Par] = Seq (k)

    def apply (x: VectorD): VectorD = x.map (v => v / (v + k.value))

end SatTerm


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `HingeTerm` case class is a HINGE term, max (0, x - c): zero below the
 *  threshold c and linear above it (the MARS basis function), with c free, e.g.,
 *  hinge (y~1, thresh ()).  It is piecewise linear in c, so the fit of c is exact
 *  only up to the spacing of the data around it.
 *  @param source  the term being thresholded
 *  @param c       the threshold parameter
 */
case class HingeTerm (source: Term, c: Par) extends ParTerm:

    override def label: String = s"hinge(${source.label}, ${c.label})"

    def pars: Seq [Par] = Seq (c)

    def apply (x: VectorD): VectorD = x.map (v => max (0.0, v - c.value))

end HingeTerm


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TransTerm` case class is a smooth TRANSITION term, 1 / (1 + exp (-(x - c) / w)):
 *  0 well below the threshold c, 1 well above it, with the switch spread over a width
 *  w, e.g., trans (y~1, thresh (), width ()).  Multiplied by other terms it gives a
 *  smooth-transition (STAR) model: trans (y~1, thresh (), width ()) * y~(3,1).
 *  @param source  the transition variable
 *  @param c       the threshold parameter
 *  @param w       the width parameter (positive)
 */
case class TransTerm (source: Term, c: Par, w: Par) extends ParTerm:

    override def label: String = s"trans(${source.label}, ${c.label}, ${w.label})"

    def pars: Seq [Par] = Seq (c, w)

    def apply (x: VectorD): VectorD = x.map (v => 1.0 / (1.0 + math.exp (-(v - c.value) / w.value)))

end TransTerm


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `EmaTerm` case class is an EXPONENTIALLY WEIGHTED MOVING AVERAGE of a term,
 *  e_t = d x_t + (1 - d) e_t-1 (e_0 = x_0): a geometric distributed lag of x in one
 *  parameter, the decay d, e.g., ema (icu~1, decay ()).  Use a lagged source so that
 *  only past values enter the average.
 *  @param source  the term being averaged
 *  @param d       the decay parameter in (0, 1): 1 => the source itself, small => long memory
 */
case class EmaTerm (source: Term, d: Par) extends ParTerm:

    override def label: String = s"ema(${source.label}, ${d.label})"

    def pars: Seq [Par] = Seq (d)

    def apply (x: VectorD): VectorD =
        val e = new VectorD (x.dim)
        if x.dim > 0 then e(0) = x(0)
        for t <- 1 until x.dim do e(t) = d.value * x(t) + (1.0 - d.value) * e(t-1)
        e
    end apply

end EmaTerm


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `DqTerm` case class is the DAMPENED QUADRATIC of a term, x |x| / sqrt (1 + x^2 / mx^2)
 *  (@see `DampenedQuad`): quadratic for |x| well below the cap mx and linear above it,
 *  with the cap free, e.g., dq (y~1, cap ()) (`dq (y~1)` keeps the cap of `Transform.hp`).
 *  @param source  the term being dampened
 *  @param mx      the cap parameter (positive)
 */
case class DqTerm (source: Term, mx: Par) extends ParTerm:

    override def label: String = s"dq(${source.label}, ${mx.label})"

    def pars: Seq [Par] = Seq (mx)

    def apply (x: VectorD): VectorD = DampenedQuad.f (x, mx.value)

end DqTerm


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ParWaveTerm` case class is a seasonal term with a FREE frequency f (cycles
 *  per time unit): sin (2 Pi f t + phase).  Use `wave (freq ())` to build the
 *  sin/cos pair sharing f (@see `WaveTerm` for a fixed frequency).
 *  @param freq   the frequency parameter (cycles per time unit)
 *  @param phase  the phase: 0 for the sine component, Pi/2 for the cosine component
 *  @param idx    the 1-based rank of the frequency in a periodogram list (0 => none)
 */
case class ParWaveTerm (freq: Par, phase: Double, idx: Int = 0) extends Term:

    def name: String =
        val kind = if phase == 0.0 then "sin" else "cos"
        s"$kind(f=${freq.label})"
    end name

    def lag: Int = 0                                                      // no history needed

    override def vars: Set [String] = Set ()                              // no data column

    def column (dataset: Map [String, VectorD]): VectorD =
        val m = dataset.head._2.dim                                       // all columns have the same length
        val w = _2Pi * freq.value
        VectorD (for t <- 0 until m yield math.sin (w * t + phase))
    end column

end ParWaveTerm


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Functions that build the terms with free parameters, on a term or on each term
 *  of a lag range (which then share the parameter).  The fixed-number versions of
 *  `ARX_Formula`'s `pow`, `wave` and `dq` are repeated here, as a definition in this
 *  package hides them.
 *  @param t   the term
 *  @param ts  the terms of a lag range
 */
def pow (t: Term, p: Par): Term = ParPowerTerm (t, p)
def pow (ts: Seq [Term], p: Par): Seq [Term] = ts.map (ParPowerTerm (_, p))
def pow (t: Term, p: Double): Term = PowerTerm (t, p)
def pow (ts: Seq [Term], p: Double): Seq [Term] = ts.map (PowerTerm (_, p))

def sat (t: Term, k: Par): Term = SatTerm (t, k)
def sat (ts: Seq [Term], k: Par): Seq [Term] = ts.map (SatTerm (_, k))

def hinge (t: Term, c: Par): Term = HingeTerm (t, c)
def hinge (ts: Seq [Term], c: Par): Seq [Term] = ts.map (HingeTerm (_, c))

def trans (t: Term, c: Par, w: Par): Term = TransTerm (t, c, w)

def ema (t: Term, d: Par): Term = EmaTerm (t, d)
def ema (ts: Seq [Term], d: Par): Seq [Term] = ts.map (EmaTerm (_, d))

def dq (t: Term, mx: Par): Term = DqTerm (t, mx)
def dq (ts: Seq [Term], mx: Par): Seq [Term] = ts.map (DqTerm (_, mx))
def dq (t: Term): Term = forecasting.dq (t)
def dq (ts: Seq [Term]): Seq [Term] = forecasting.dq (ts)

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Build the sin/cos pair of wave terms for a FREE frequency (cycles per time unit),
 *  e.g., wave (freq ()); the two terms share the parameter.  The fixed-frequency
 *  versions of `ARX_Formula` are repeated here.
 *  @param freq   the frequency: a free parameter, or a number in cycles per time unit
 *  @param freqs  the frequencies in cycles per time unit
 *  @param idx    the 1-based rank of the frequency in its periodogram list (0 => none)
 *  @param k      the number of frequencies (sin/cos pairs) to include
 */
def wave (freq: Par): Seq [Term] = Seq (ParWaveTerm (freq, 0.0), ParWaveTerm (freq, Pi / 2.0))
def wave (freq: Double, idx: Int = 0): Seq [Term] = forecasting.wave (freq, idx)
def wave (freqs: VectorD, k: Int): Seq [Term] = forecasting.wave (freqs, k)

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The power operator with a free exponent, t ↑ p, on a term or on each term of a
 *  lag range (which then share the parameter), e.g., y~(3,1) ↑ power ().
 *  The fixed-exponent version of `ARX_Formula` is repeated here, as a definition in
 *  this package hides it (shift and scale are `Par`'s + and *).
 */
extension (t: Term)
    def ↑ (p: Par): Term = ParPowerTerm (t, p)
    def ↑ (p: Double): Term = PowerTerm (t, p)

extension (ts: Seq [Term])
    def ↑ (p: Par): Seq [Term] = ts.map (ParPowerTerm (_, p))
    def ↑ (p: Double): Seq [Term] = ts.map (PowerTerm (_, p))

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Fit the `NARX_SR_D` model of any formula, (y ~= ...).fitNL_D (hh), with its free
 *  parameters fitted (@see `NARX_Formula.fitNL_D`); .fit_D (hh) gives the `ARX_D` model
 *  with the parameters at their current values (.fit (hh) the recursive `ARX`).
 */
extension (f: ARX_Formula)
    def fitNL_D (hh: Int = 1, tFormT: TransformT = f.tFormT, trSize: Int = -1, scaled: Boolean = false)
              (using dataset: Map [String, VectorD]): NARX_SR_D =
        NARX_Formula (f).fitNL_D (hh, tFormT, trSize, scaled)


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NARX_Formula` class is an `ARX_Formula` whose terms may hold free nonlinear
 *  parameters (@see `Par`).  `NARX_Formula (f)` wraps any `ARX_Formula` f, and the
 *  extension `fitNL_D` lets (y ~= ...).fitNL_D (hh) fit the `NARX_SR_D` model with the
 *  parameters fitted, while .fit_D (hh) fits the `ARX_D` model with them at their
 *  current values.
 *  @param resp_       the response variable
 *  @param preds_      the predictor terms (distinct labels, @see `ARX_Formula.distinct`)
 *  @param tFormT_     the transform type for every base column (null => none)
 *  @param respTForm_  the transform already applied to the response by `letIn` (None => none)
 */
class NARX_Formula (resp_ : FormulaVar, preds_ : VEC [Term],
                    tFormT_ : TransformT = null,
                    respTForm_ : Option [Transform] = None)
      extends ARX_Formula (resp_, preds_, tFormT_, respTForm_):

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the distinct free parameters of the terms, in the order they appear in the labels.
     */
    def params: Seq [Par] = NARX_Formula.params (preds)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of distinct parameters (free ones and constants written in
     *  parameter positions).
     */
    def nParams: Int = params.size

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of distinct FREE parameters (the ones a fit searches; a
     *  parameter with equal bounds is a constant).
     *  @note:  `nFree` -- the count the DF penalty of `NARX_SR_D` and the `llmsr` scorer use.
     */
    def nFree: Int = params.count (_.free)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a canonical key for de-duplication: as `ARX_Formula.key`, with every
     *  free parameter written as ? (pow(y~1, ?1.500) -> pow(y~1, ?), before and after
     *  the fit), so formulas that differ only in start or fitted values collide as intended.
     *  @note:  the key is independent of the parameters' state (@see `Par.LabelMode.Slot`); `proposalStr`
     *  shows the current values as start values; `formulaStr` writes a free-frequency pair as wave(?w).
     */
    override def key: String = Par.withMode (Par.LabelMode.Slot) { super.key }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the formula as a specification string with every free parameter written
     *  as ?w, w its current (start or fitted) value -- the form a proposer is shown, which
     *  `NARX_FormulaParser` reads back as a start value.
     */
    def proposalStr: String = Par.withMode (Par.LabelMode.SlotValue) { formulaStr }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the formula as a specification string in the grammar (@see
     *  `ARX_Formula.formulaStr`), a free-frequency sin/cos pair written as wave(?),
     *  wave(?w) or wave(w) (its parameter's label), so `NARX_FormulaParser` reads it back.
     */
    override def formulaStr: String =
        val (pw, rest) = preds.partition (_.isInstanceOf [ParWaveTerm])
        if pw.isEmpty then super.formulaStr
        else
            val base  = new ARX_Formula (resp, rest, null).formulaStr      // the other terms (transform added below)
            val pairs = pw.map (_.asInstanceOf [ParWaveTerm].freq).distinct.map (f => s"wave(${f.label})")
            val head  = if rest.isEmpty then s"${resp.name} ~= ${pairs.mkString (" + ")}"
                        else s"$base + ${pairs.mkString (" + ")}"
            if tFormT == null then head else s"$head TFORM: $tFormT"
    end formulaStr

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Fit an `NARX_SR_D` direct forecasting model with one column per term of this
     *  formula (@see `NARX_SR_D.fromTerms`): the free parameters are fitted on the
     *  training rows by `OPTIMIZER_NL`, then the coefficients in-sample (trained on
     *  every row), as `ARX_Formula.fit_D` does for `ARX_D`.
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param tFormT   the transform type applied to every base column (defaults to
     *                  this formula's; null => none)
     *  @param trSize   the training rows the base-column transforms and the free
     *                  parameters are fit on (-1 => `Model.trSize`)
     *  @param scaled   whether the regression (and the fit of the parameters)
     *                  standardizes the term columns
     *  @param dataset  the map of named data columns
     */
    def fitNL_D (hh: Int = 1, tFormT: TransformT = this.tFormT, trSize: Int = -1, scaled: Boolean = false)
              (using dataset: Map [String, VectorD]): NARX_SR_D =
        val mod = NARX_SR_D.fromTerms (resp, preds, hh, tFormT = tFormT, trSize = trSize, scaled = scaled)
        val (x, yy) = (mod.getX, mod.getYy)
        mod.trainNtest (x, yy)(x, yy)
        mod
    end fitNL_D

end NARX_Formula


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NARX_Formula` companion object wraps formulas, collects their free
 *  parameters and extends the grammar text.
 */
object NARX_Formula:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the formula as an `NARX_Formula` (the same object when it already is one).
     *  @param f  the formula
     */
    def apply (f: ARX_Formula): NARX_Formula = f match
        case nf: NARX_Formula => nf
        case _                => new NARX_Formula (f.resp, f.preds, f.tFormT, f.respTForm)
    end apply

// Note: a formula from a response and terms (the search builds formulas this way), `fresh` copies of
//       terms (own parameters) and `distinct` by structure (start values ignored)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a formula from a response and predictor terms (distinct by structure).
     *  @param resp    the response variable
     *  @param preds   the predictor terms
     *  @param tFormT  the transform type for every base column (null => none)
     */
    def apply (resp: FormulaVar, preds: scala.collection.Seq [Term], tFormT: TransformT = null): NARX_Formula =
        new NARX_Formula (resp, distinct (preds), tFormT)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return copies of the terms whose free parameters are NEW `Par` objects (one copy
     *  per parameter, so a parameter shared by several terms stays shared), with the
     *  numbers as given, so that a formula built from the terms of other formulas
     *  (a crossover, a library selection) fits its own parameters without touching
     *  theirs.  Terms without parameters are returned as they are.
     *  @param preds  the predictor terms
     */
    def fresh (preds: scala.collection.Seq [Term]): VEC [Term] =
        val copies = scala.collection.mutable.HashMap [Par, Par] ()         // identity map (Par has no equals)
        def cp (p: Par): Par = copies.getOrElseUpdate (p, p.fresh)
        def walk (t: Term): Term = t match
            case ShiftTerm (s, p)          => ShiftTerm (walk (s), cp (p))
            case ScaleTerm (s, a)          => ScaleTerm (walk (s), cp (a))
            case ParPowerTerm (s, p)       => ParPowerTerm (walk (s), cp (p))
            case SatTerm (s, k)            => SatTerm (walk (s), cp (k))
            case HingeTerm (s, c)          => HingeTerm (walk (s), cp (c))
            case TransTerm (s, c, w)       => TransTerm (walk (s), cp (c), cp (w))
            case EmaTerm (s, d)            => EmaTerm (walk (s), cp (d))
            case DqTerm (s, mx)            => DqTerm (walk (s), cp (mx))
            case ParWaveTerm (f, ph, idx)  => ParWaveTerm (cp (f), ph, idx)
            case TransformedTerm (s, fn, op) => TransformedTerm (walk (s), fn, op)
            case PowerTerm (s, p, safe)    => PowerTerm (walk (s), p, safe)
            case ProductTerm (l, r)        => ProductTerm (walk (l), walk (r))
            case other                     => other
        VEC.from (preds.map (walk))
    end fresh

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the terms distinct by STRUCTURE: as `ARX_Formula.distinct` (by label),
     *  with every free parameter written as ?, so that two copies of a term that differ
     *  only in start or fitted values count once (the first is kept).
     *  @param preds  the predictor terms
     */
    def distinct (preds: scala.collection.Seq [Term]): VEC [Term] =
        Par.withMode (Par.LabelMode.Slot) { ARX_Formula.distinct (preds) }
    end distinct

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the distinct free parameters of the terms (by identity), in the order
     *  they appear in the labels, looking inside transformed, power and product terms.
     *  @param preds  the predictor terms
     */
    def params (preds: scala.collection.Seq [Term]): Seq [Par] =
        val found = VEC [Par] ()
        def add (p: Par): Unit = if ! found.exists (_ eq p) then found += p
        def walk (t: Term): Unit = t match
            case ShiftTerm (s, p)          => add (p); walk (s)
            case ScaleTerm (s, a)          => add (a); walk (s)
            case pt: ParTerm               => walk (pt.source); pt.pars.foreach (add)
            case ParWaveTerm (f, _, _)     => add (f)
            case TransformedTerm (s, _, _) => walk (s)
            case PowerTerm (s, _, _)       => walk (s)
            case ProductTerm (l, r)        => walk (l); walk (r)
            case _                         =>
        preds.foreach (walk)
        found.toSeq
    end params

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Resolve the free parameters of the terms that are still unset, from the
     *  training rows of the columns they act on (@see `Par.resolve`); the terms do
     *  this themselves when first evaluated, this call lets `fromTerms` use its own
     *  number of training rows.
     *  @param preds    the predictor terms
     *  @param dataset  the map of named data columns
     *  @param trSize   the number of training rows
     */
    def resolve (preds: scala.collection.Seq [Term], dataset: Map [String, VectorD], trSize: Int): Unit =
        def walk (t: Term): Unit = t match
            case pt: ParTerm =>
                if pt.pars.exists (! _.resolved) then
                    val x = pt.source.column (dataset)
                    for p <- pt.pars do p.resolve (x, trSize)
                walk (pt.source)
            case TransformedTerm (s, _, _) => walk (s)
            case PowerTerm (s, _, _)       => walk (s)
            case ProductTerm (l, r)        => walk (l); walk (r)
            case _                         =>
        preds.foreach (walk)
    end resolve

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the free parameters of a term that the LINEAR fit already covers, so
     *  that searching them would make the problem singular (they are kept at their
     *  start values by `NARX_SR_D.fromTerms`):
     *      x + s or a x alone              the intercept / the coefficient covers s / a
     *      exp (x + s) = e^s exp (x)       the coefficient covers e^s (also expm1)
     *      log (a x) = log a + log (x)     the intercept covers log a (also with a shift)
     *      (a x)^p = a^p x^p               the coefficient covers a^p (also with a shift)
     *      (a x) * t                       the coefficient covers a
     *      sat (a x, k) = sat (x, k / a)   k covers a
     *      hinge (a x + s, c), trans (a x + s, c, w): c and w cover s and a
     *      ema (a x + s) = a ema (x) + s   the coefficient and the intercept cover a and s
     *  @param t  the term
     */
    def redundant (t: Term): Seq [Par] = t match
        case ShiftTerm (_, s)                                            => Seq (s)
        case ScaleTerm (_, a)                                            => Seq (a)
        case TransformedTerm (ShiftTerm (_, s), _, "exp" | "expm1")      => Seq (s)
        case TransformedTerm (ScaleTerm (_, a), _, "log")                => Seq (a)
        case TransformedTerm (ShiftTerm (ScaleTerm (_, a), _), _, "log") => Seq (a)
        case ParPowerTerm (ScaleTerm (_, a), _)                          => Seq (a)
        case ParPowerTerm (ShiftTerm (ScaleTerm (_, a), _), _)           => Seq (a)
        case ProductTerm (ScaleTerm (_, a), _)                           => Seq (a)
        case ProductTerm (_, ScaleTerm (_, a))                           => Seq (a)
        case SatTerm (ScaleTerm (_, a), _)                               => Seq (a)
        case HingeTerm (s, _)                                            => affine (s)
        case TransTerm (s, _, _)                                         => affine (s)
        case EmaTerm (s, _)                                              => affine (s)
        case _                                                           => Seq ()
    end redundant

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the shift and scale parameters at the top of a term (a x + s or s + a x).
     *  @param t  the term
     */
    private def affine (t: Term): Seq [Par] = t match
        case ShiftTerm (s, p) => p +: affine (s)
        case ScaleTerm (s, a) => a +: affine (s)
        case _                => Seq ()
    end affine

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Remove the start values from the free-parameter slots of a label or key,
     *  e.g., pow(y~1, ?1.500) -> pow(y~1, ?).
     *  @param s  the label or key
     */
    def slots (s: String): String = s.replaceAll ("\\?[-+.0-9eE]*", "?")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the grammar as text: `ARX_Formula.grammar` plus the free parameters.
     *  @param maxLag  the maximum lag allowed
     *  @param pMin    the minimum exponent allowed
     *  @param pMax    the maximum exponent allowed
     */
    def grammar (maxLag: Int = 8, pMin: Double = 0.2, pMax: Double = 3.0): String =
        ARX_Formula.grammar (maxLag, pMin, pMax) +
        s"""
           |free parameters: a number in a term may be ?, a parameter fitted on the training rows (?w0: from the start value w0)
           |            pow(term, ?)          free exponent, $pMin <= p <= $pMax
           |            FN(? + term)          free shift inside log, log1p, pow, sin, cos (not exp: e^s is a coefficient)
           |            exp(? * term)         free rate (not inside log or pow: a is a coefficient there)
           |            sat(term, ?)          saturation x / (x + k), free half-saturation k
           |            hinge(term, ?)        max(0, x - c), free threshold c
           |            trans(term, ?, ?)     smooth step 1 / (1 + exp(-(x - c) / w)), free threshold c and width w
           |            ema(term, ?)          exponentially weighted moving average, free decay in (0, 1)
           |            dq(term, ?)           dampened quadratic, free cap
           |            wave(?)               free frequency in cycles per time unit (one sin/cos pair)
           |            the same parameter may be shared by the terms of a lag range, e.g., pow(y~(3,1), ?)""".stripMargin
    end grammar

end NARX_Formula


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nARX_FormulaTest` main function tests the `NARX_Formula` class on a toy
 *  dataset generated WITHOUT noise from
 *
 *      y_t = 1 + 0.5 (x_t-1 + 2)^1.7 + 3 log (z_t-1 + 1.5),
 *
 *  so the free parameters (shift 2, exponent 1.7, shift 1.5), created by their roles
 *  with no numbers given, and the coefficients must be recovered by `fitNL_D`, while
 *  `fit_D` keeps the start values.  The same formula object serves both, as in
 *  `aRX_FormulaTest`.
 *  > runMain scalation.modeling.forecasting.nonlinear.nARX_FormulaTest
 */
@main def nARX_FormulaTest (): Unit =

    val m  = 100
    val x_ = VectorD (for t <- 0 until m yield 1.0 + 0.05 * t + 0.6 * math.sin (0.7 * t))
    val z_ = VectorD (for t <- 0 until m yield 0.5 + 2.0 * math.abs (math.cos (0.3 * t)))
    val y_ = new VectorD (m)
    for t <- 1 until m do y_(t) = 1.0 + 0.5 * math.pow (x_(t-1) + 2.0, 1.7) + 3.0 * math.log (z_(t-1) + 1.5)
    y_(0) = y_(1)

    given dataContext: Map [String, VectorD] = Map ("y" -> y_, "x" -> x_, "z" -> z_)

    val y = Vars.y
    val x = Vars.x
    val z = Vars.z

    RidgeRegression.hp("lambda") = 1E-12                                // ~OLS, so the parameters can be recovered exactly

    banner ("formula with three free parameters: shift and exponent of x~1, shift of z~1")
    val formula = y ~= pow (shift () + x~1, power ()) + log (shift () + z~1)
    println (s"formula    = $formula")
    println (s"key        = ${NARX_Formula (formula).key}")
    println (s"nParams    = ${NARX_Formula (formula).nParams}")

    banner ("fit_D: ARX_D with the parameters at their start values (resolved from the training rows)")
    val mod0 = formula.fit_D ()
    println (s"formula    = $formula")
    println (s"bounds     = ${NARX_Formula (formula).params.map (p => s"[${p.lb}, ${p.ub}]")}")
    println (s"fname      = ${stringOf (mod0.getFname)}")
    println (s"intercept  = ${mod0.intercept}, slopes = ${mod0.slopes(?, 0)}")
    mod0.forecastAll ()
    mod0.diagnoseAll (mod0.getY, mod0.getYf)

    banner (s"fitNL_D: NARX_SR_D with the parameters fitted by $OPTIMIZER_NL")
    val mod = formula.fitNL_D ()
    println (s"formula    = $formula")
    println (s"fname      = ${stringOf (mod.getFname)}")
    println (s"intercept  = ${mod.intercept}, slopes = ${mod.slopes(?, 0)}")
    println (s"param_nl   = ${mod.parameter_nl}")
    mod.forecastAll ()
    mod.diagnoseAll (mod.getY, mod.getYf)

    val w = mod.parameter_nl
    assert ((w - VectorD (2.0, 1.7, 1.5)).norm < 1e-3, s"fitNL_D must recover (2, 1.7, 1.5), got $w")
    assert (math.abs (mod.intercept(0) - 1.0) < 1e-3 && (mod.slopes(?, 0) - VectorD (0.5, 3.0)).norm < 1e-3,
            s"fitNL_D must recover the intercept 1 and the slopes (0.5, 3), got ${mod.intercept}, ${mod.slopes(?, 0)}")

end nARX_FormulaTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nARX_FormulaTest2` main function tests the `NARX_Formula` class on the
 *  Covid-19 dataset, writing `nARX_SR_DTest6`'s transform sets as explicit terms with
 *  free parameters: a shifted power of the y lags (shared shift and exponent, as a
 *  `Pow` transform shares them) and a shifted log of each exogenous lag range.  The
 *  formula is fitted with the start values (`fit_D`, as `tune = false`) and with the
 *  fitted parameters (`fitNL_D`), each followed by rolling validation on the testing
 *  range, as in `aRX_FormulaTest5`.
 *  > runMain scalation.modeling.forecasting.nonlinear.nARX_FormulaTest2
 */
@main def nARX_FormulaTest2 (): Unit =

    import Example_Covid._

    val exoColumns = Array ("icu_patients", "positive_rate")
    val (xe, y_)   = clip (loadData (exoColumns))

    given dataContext: Map [String, VectorD] = Map (
        "y"     -> y_,
        "icu"   -> xe.col(0),
        "pRate" -> xe.col(1))

    val y     = Vars.y
    val icu   = Vars.icu
    val pRate = Vars.pRate

    val hh = 6
    RidgeRegression.hp("lambda") = 1.0

    banner ("lags, a shifted power of the y lags and a shifted log of each exogenous lag range")
    val formula = y ~= y~(6,1) + pow (shift () + y~(6,1), power (lb = 1.1, ub = 2.5)) +
                       icu~(4,1) + log (shift () + icu~(4,1)) +
                       pRate~(4,1) + log (shift () + pRate~(4,1))
    println (s"formula   = $formula")
    println (s"nParams   = ${NARX_Formula (formula).nParams}")

    banner ("fit_D: ARX_D with the parameters at their start values")
    val mod0 = formula.fit_D (hh, scaled = true)
    println (s"modelName = ${mod0.modelName}")
    println (s"formula   = $formula")
    println (s"fname     = ${stringOf (mod0.getFname)}")
    mod0.forecastAll ()
    mod0.diagnoseAll (mod0.getY, mod0.getYf)
    mod0.setSkip ()                                                     // don't skip any rows
    mod0.rollValidate (rc = 2)
    mod0.diagnoseAll (mod0.getY, mod0.getYf, Forecaster.teRng (y_.dim))

    banner (s"fitNL_D: NARX_SR_D with the parameters fitted by $OPTIMIZER_NL")
    val mod = formula.fitNL_D (hh, scaled = true)
    println (s"modelName = ${mod.modelName}")
    println (s"formula   = $formula")
    println (s"fname     = ${stringOf (mod.getFname)}")
    println (s"param_nl  = ${mod.parameter_nl}")
    mod.forecastAll ()
    mod.diagnoseAll (mod.getY, mod.getYf)
    mod.setSkip ()                                                      // don't skip any rows
    mod.rollValidate (rc = 2)
    mod.diagnoseAll (mod.getY, mod.getYf, Forecaster.teRng (y_.dim))

end nARX_FormulaTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nARX_FormulaTest3` main function tests the grammar side of `NARX_Formula`:
 *  labels with ? slots (bare until the parameter is resolved), one parameter shared
 *  by a lag range, `key` without start values, a redundant (linear-in-disguise)
 *  parameter that is reported and kept at its start value, labels in the plain
 *  grammar after `fitNL_D`, and a free wave frequency recovered from a noise-free
 *  seasonal series.
 *  > runMain scalation.modeling.forecasting.nonlinear.nARX_FormulaTest3
 */
@main def nARX_FormulaTest3 (): Unit =

    val y    = Vars.y
    val hosp = Vars.hosp

    banner ("labels with free-parameter slots; a lag range shares one parameter")
    val p  = power ()
    val f1 = y ~= y~(3,1) + y~(3,1) ↑ p + log (shift () + hosp~1) + exp (rate () * hosp~1)
    println (s"f1.formulaStr = ${f1.formulaStr}")
    println (s"f1.key        = ${NARX_Formula (f1).key}")
    println (s"f1.params     = ${NARX_Formula (f1).params.map (p => s"${p.role}: $p")}")
    assert (f1.preds.map (_.label) == Seq ("y~3", "y~2", "y~1", "pow(y~3, ?1.500)", "pow(y~2, ?1.500)", "pow(y~1, ?1.500)",
                                            "log(? + hosp~1)", "exp(? * hosp~1)"))
    assert (NARX_Formula (f1).nParams == 3, "the three power terms share one parameter")
    assert (NARX_Formula (f1).key == "y ~= exp(? * hosp~1) + log(? + hosp~1) + pow(y~1, ?) + pow(y~2, ?) + pow(y~3, ?) + y~1 + y~2 + y~3")

    banner ("grammar text")
    println (NARX_Formula.grammar ())

    banner ("a redundant parameter (shift inside exp) is reported and kept; the rate is recovered; plain labels after fitNL_D")
    val m  = 80
    val x_ = VectorD (for t <- 0 until m yield 0.5 + 0.02 * t + 0.3 * math.sin (0.5 * t))
    val y_ = new VectorD (m)
    for t <- 1 until m do y_(t) = 2.0 + 1.5 * math.exp (0.8 * x_(t-1)) + 0.4 * math.pow (x_(t-1), 1.3)
    y_(0) = y_(1)
    given dataContext: Map [String, VectorD] = Map ("y" -> y_, "x" -> x_)
    val x = Vars.x
    RidgeRegression.hp("lambda") = 1E-12

    val s  = shift ()                                                   // redundant: exp (x + s) = e^s exp (x)
    val f2 = y ~= exp (s + rate () * x~1) + pow (x~1, 1.3)
    println (s"f2 before fitNL_D = $f2")
    val mod = f2.fitNL_D ()
    println (s"f2 after fitNL_D  = $f2")
    println (s"param_nl        = ${mod.parameter_nl}")
    assert (s.value == s.w0, "the redundant shift must keep its start value")
    assert (math.abs (mod.parameter_nl(1) - 0.8) < 1e-4, s"fitNL_D must recover the rate 0.8, got ${mod.parameter_nl}")
    assert (f2.formulaStr == s"y ~= exp(${Par.fmt (s.w0)} + 0.8000 * x~1) + pow(x~1, 1.3)", s"plain labels expected, got ${f2.formulaStr}")

    banner ("a free wave frequency is recovered from a noise-free seasonal series (start 0.12 given, bounds from the role)")
    val w_ = VectorD (for t <- 0 until m yield 5.0 + 2.0 * math.sin (_2Pi * 0.13 * t) + math.cos (_2Pi * 0.13 * t))
    val f3 = Vars.w ~= wave (freq (0.12))
    val mod3 = f3.fitNL_D ()(using Map ("w" -> w_))
    println (s"f3 after fitNL_D  = $f3")
    println (s"intercept       = ${mod3.intercept}, slopes = ${mod3.slopes(?, 0)}")
    assert (math.abs (mod3.parameter_nl(0) - 0.13) < 1e-4, s"fitNL_D must recover the frequency 0.13, got ${mod3.parameter_nl}")

end nARX_FormulaTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nARX_FormulaTest4` main function tests the terms with free parameters
 *  beyond shift, power and rate -- sat, hinge, trans, ema and dq -- each on a toy
 *  series generated WITHOUT noise from the term with a known parameter, so `fitNL_D`
 *  (parameters created by their roles, no numbers given) must recover it:
 *      y = 1 + 4 sat (x~1, 1.5)             half-saturation 1.5
 *      y = 0.5 + 2 hinge (x~1, 2.0)         threshold 2.0
 *      y = 1 + 2 trans (x~1, 2.0, 0.3)      threshold 2.0, width 0.3
 *      y = 1 + 2 ema (x~1, 0.3)             decay 0.3
 *      y = 1 + 2 dq (x~1, 3.0)              cap 3.0
 *  > runMain scalation.modeling.forecasting.nonlinear.nARX_FormulaTest4
 */
@main def nARX_FormulaTest4 (): Unit =

    val m  = 120
    val x_ = VectorD (for t <- 0 until m yield 2.5 + 2.2 * math.sin (0.37 * t) + 0.004 * t)   // in [0.3, 5.2]
    val xl = x_.lagv (1)                                                // the x~1 column
    val x  = Vars.x
    val y  = Vars.y
    RidgeRegression.hp("lambda") = 1E-12

    val e = new VectorD (m)                                             // ema of x~1 with decay 0.3
    e(0)  = xl(0)
    for t <- 1 until m do e(t) = 0.3 * xl(t) + 0.7 * e(t-1)

    val cases = Seq (                                                   // (name, series, formula, truth, tolerance)
        ("sat",   xl.map (v => 1.0 + 4.0 * v / (v + 1.5)),                        y ~= sat (x~1, half ()),               VectorD (1.5), 1e-3),
        ("hinge", xl.map (v => 0.5 + 2.0 * max (0.0, v - 2.0)),                   y ~= hinge (x~1, thresh ()),           VectorD (2.0), 1e-2),
        ("trans", xl.map (v => 1.0 + 2.0 / (1.0 + math.exp (-(v - 2.0) / 0.3))),  y ~= trans (x~1, thresh (), width ()), VectorD (2.0, 0.3), 1e-3),
        ("ema",   e * 2.0 + 1.0,                                                  y ~= ema (x~1, decay ()),              VectorD (0.3), 1e-3),
        ("dq",    DampenedQuad.f (xl, 3.0) * 2.0 + 1.0,                           y ~= dq (x~1, cap ()),                 VectorD (3.0), 1e-3))

    for (name, y_, f, truth, tol) <- cases do
        banner (s"$name: ${NARX_Formula (f).key}")
        given dataContext: Map [String, VectorD] = Map ("y" -> y_, "x" -> x_)
        val mod = f.fitNL_D ()
        println (s"fitted formula = $f")
        println (s"param_nl       = ${mod.parameter_nl} (truth $truth)")
        println (s"intercept      = ${mod.intercept}, slopes = ${mod.slopes(?, 0)}")
        assert ((mod.parameter_nl - truth).norm < tol, s"$name: fitNL_D must recover $truth, got ${mod.parameter_nl}")
    end for

end nARX_FormulaTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nARX_FormulaTest5` main function tests the new terms on the Covid-19
 *  dataset: a saturating effect of icu, a geometric distributed lag of the positive
 *  rate, and a smooth-transition (STAR) switch of the y lags on the level of y~1,
 *  with `fit_D` (start values) and `fitNL_D` (fitted), each followed by rolling
 *  validation on the testing range.
 *  > runMain scalation.modeling.forecasting.nonlinear.nARX_FormulaTest5
 */
@main def nARX_FormulaTest5 (): Unit =

    import Example_Covid._

    val exoColumns = Array ("icu_patients", "positive_rate")
    val (xe, y_)   = clip (loadData (exoColumns))

    given dataContext: Map [String, VectorD] = Map (
        "y"     -> y_,
        "icu"   -> xe.col(0),
        "pRate" -> xe.col(1))

    val y     = Vars.y
    val icu   = Vars.icu
    val pRate = Vars.pRate

    val hh = 6
    RidgeRegression.hp("lambda") = 1.0

    banner ("lags, a saturating icu effect, a distributed lag of pRate, and a smooth-transition switch of the y lags")
    val formula = y ~= y~(3,1) + sat (icu~1, half ()) + ema (pRate~1, decay ()) +
                       trans (y~1, thresh (), width ()) * y~(3,1)
    println (s"formula   = $formula")
    println (s"nParams   = ${NARX_Formula (formula).nParams}")

    banner ("fit_D: ARX_D with the parameters at their start values")
    val mod0 = formula.fit_D (hh, scaled = true)
    println (s"formula   = $formula")
    println (s"fname     = ${stringOf (mod0.getFname)}")
    mod0.forecastAll ()
    mod0.diagnoseAll (mod0.getY, mod0.getYf)
    mod0.setSkip ()                                                     // don't skip any rows
    mod0.rollValidate (rc = 2)
    mod0.diagnoseAll (mod0.getY, mod0.getYf, Forecaster.teRng (y_.dim))

    banner (s"fitNL_D: NARX_SR_D with the parameters fitted by $OPTIMIZER_NL")
    val mod = formula.fitNL_D (hh, scaled = true)
    println (s"formula   = $formula")
    println (s"fname     = ${stringOf (mod.getFname)}")
    println (s"param_nl  = ${mod.parameter_nl}")
    mod.forecastAll ()
    mod.diagnoseAll (mod.getY, mod.getYf)
    mod.setSkip ()                                                      // don't skip any rows
    mod.rollValidate (rc = 2)
    mod.diagnoseAll (mod.getY, mod.getYf, Forecaster.teRng (y_.dim))

end nARX_FormulaTest5

