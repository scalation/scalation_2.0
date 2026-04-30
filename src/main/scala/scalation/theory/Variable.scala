
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat May 18 14:57:50 EDT 2019
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Theory/Model Support: Meta-data about a Variable
 */

package scalation
package theory

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VariableKind` enumeration indicates the kind of variable.
 *  @param name  the name of the variable kind
 */
enum VariableKind (val name: String):

    case Categorical extends VariableKind ("Categorical")
    case Ordinal     extends VariableKind ("Ordinal")
    case Continuous  extends VariableKind ("Continuous")

end VariableKind

import VariableKind._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Variable` class provides meta-data for a variable including its kind,
 *  distinct values, name and optional ontological concept.  The variable may
 *  be an input variable (feature) or an output variable (response).  Typically,
 *  it represents a column 'xj' in a data matrix.
 *------------------------------------------------------------------------------
 *  Several modeling techniques such as decision trees need to divide the values
 *  into groups, e.g., for branch values:
 *  When 'xj' is categorical, these will be all its distinct values.
 *  Otherwise, these will be 0 (up to threshold) or 1 (above threshold).
 *  @see `classifier.Node` for 'threshold' 
 *  @param xj       the column vector (feature/response)
 *  @param j        the index position within the relevant data matrix
 *  @param kind     indication of the variable kind
 *  @param name     the name of column (feature or response)
 *  @param concept  an optional URI for an optological concept
 */
case class Variable (xj: VectorD, j: Int, kind: VariableKind = Continuous,
                     name_ : String = null, concept: String = null):

    val name = if name_ == null then s"x$j" else name_

    val values = kind match
    case Categorical => xj.toInt.distinct.sorted             // distinct values
    case _           => VectorI (0, 1)                       // 0 => below, 1 => above threshold

end Variable


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Variable` companion object provides utilities for variables.
 *  Pick either COMMON or ROBUST values to the dummy variables.
 */
object Variable:

//  Common values for dummy variables
    val VAL0 = 0.0                                           // first value  (e.g., 0 or 1)
    val VAL1 = 1.0                                           // second value (e.g., 1 or 2)

//  Robust values for dummy variables
//  val VAL0 = 1.0                                           // first value  (e.g., 0 or 1)
//  val VAL1 = 2.0                                           // second value (e.g., 1 or 2)

    private val debug = debugf ("Variable", false)           // debug function
    private val flaw  = flawf ("Variable")                   // flaw function
    private var shift = 0                                    // shift values to start at 0
    private var tmax  = 0                                    // the maximum value after shifting

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the shift in categorical/treatment variable to make it start at zero
     *  as well as the maximum value after shifting.  Must call 'dummyVars' first
     */
    def get_shift_tmax: (Int, Int) = (shift, tmax)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Assign values for the dummy variables based on the categorical/treatment
     *  vector 'tt'.  A single categorical variable 'tt' with values 'ttmin' to 'ttmax'
     *  will be (1) shifted to the range 0 to 'tmax' and then replace by 'tmax'
     *  dummy variables/columns as follows:
     *      0  =>  0, 0, 0  OR  1, 1, 1
     *      1  =>  1, 0, 0  OR  2, 1, 1
     *      2  =>  0, 1, 0  OR  1, 2, 1
     *      3  =>  0, 0, 1  OR  1, 1, 2
     *  Using (0, 1) for (VAL0, VAL1) is conventional, but using (1, 2) reduces
     *  collinearity, for example in `QuadRegression`.
     *  Note: one-hot encoding using 3 dummy variable leads to singular matrices.
     *  @param tt  the categorical/treatment vector
     */
    def dummyVars (tt: VectorI): MatrixD =
        shift  = tt.min                                      // record shift
        val t  = if shift != 0 then tt - shift else tt
        tmax   = t.max
        val xd = MatrixD.fill (t.dim, tmax, VAL0)
        for i <- t.indices do
            val ti = t(i)                                    // treatment level for ith item
            if ti > 0 then xd(i, ti-1) = VAL1 
        end for
        xd
    end dummyVars

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Assign values for dummy variables based on a single categorical/treatment
     *  value tt.
     *  @param tt   the categorical/treatment value 
     *  @param sht  the amount to shift the value
     *  @param tmx  the maximum categorical/treatment after shifting
     */
    def dummyVar (tt: Int, shf: Int = shift, tmx: Int = tmax): VectorD =
        if tmx < 1 then flaw ("dummyVar", s"requires maximum categorical value $tmx >= 1")
        val xd = new VectorD (tmx); xd.set (VAL0)

        val t  = if shf != 0 then tt - shf else tt
        debug ("dummyVar", s"shf = $shf, original tt = $tt, shifted t = $t")
        if t > 0 then xd(t-1) = VAL1
        xd
    end dummyVar

end Variable

import Variable._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `variableTest` the conversion of a categorical variable into multiple
 *  dummy variables.
 *  > runMain scalation.modeling.variableTest
 */
@main def variableTest (): Unit =

     val t1 = VectorI (0, 1, 2, 3, 3, 2, 1, 0)
     val t2 = VectorI (1, 2, 3, 4, 4, 3, 2, 1)

     banner (s"Encoding Categorical Variables - Base Values: VAL0 = $VAL0, VAL1 = $VAL1")

     banner ("Conversion of variable/vector t1 to multiple dummy variables/vectors td1")
     println ("t1  = " + t1)
     println ("td1 = " + dummyVars (t1))

     banner ("Conversion of variable/vector t2 to multiple dummy variables/vectors td2")
     println ("t2  = " + t2)
     println ("td2 = " + dummyVars (t2))

end variableTest

