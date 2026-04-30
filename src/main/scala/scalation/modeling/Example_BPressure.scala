
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miler
 *  @version 2.0
 *  @date    Thu Jan 31 15:08:27 EST 2019
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Example Dataset: Blood Pressure
 *
 *-----------------------------------------------------------------------------
 *  Convention:
 *      'x'    - data/input matrix (may have an intercept column)
 *      'y'    - response/output vector
 *      'ox'   - data matrix with an intercept column of ones prepended
 *      'xy'   - combined data matrix
 *      'oxy'  - combined data matrix with an intercept column of ones prepended
 *      't'    - index for the data points (instances 0, 1, ... m-1)
 *
 *  Note: for studies where only models with intercepts are considered,
 *  ox and oxy are not needed (the ones column will already be in x).
 */

package scalation
package modeling

import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Example_BPressure` object stores the Blood Pressure dataset in a matrix.
 *  @see online.stat.psu.edu/online/development/stat501/data/bloodpress.txt
 */
object Example_BPressure:

    /** the variable/feature names
     */
    val x_fname = Array ("age", "weight", "dur", "stress")

    /** the variable/feature names including intercept (column of ones)
     */
    val ox_fname = Array ("intercept") ++ x_fname

    /** Blood Pressure (BP) data/input matrix
     *  20 data points:        x_1     x_2    x_3      x_4
     *                         Age  Weight    Dur   Stress
     */
    val x = MatrixD ((20, 4), 47.0,   85.4,   5.1,    33.0,     // 1
                              49.0,   94.2,   3.8,    14.0,     // 2
                              49.0,   95.3,   8.2,    10.0,     // 3
                              50.0,   94.7,   5.8,    99.0,     // 4
                              51.0,   89.4,   7.0,    95.0,     // 5
                              48.0,   99.5,   9.3,    10.0,     // 6
                              49.0,   99.8,   2.5,    42.0,     // 7
                              47.0,   90.9,   6.2,     8.0,     // 8
                              49.0,   89.2,   7.1,    62.0,     // 9
                              48.0,   92.7,   5.6,    35.0,     // 10

                              47.0,   94.4,   5.3,    90.0,     // 11
                              49.0,   94.1,   5.6,    21.0,     // 12
                              50.0,   91.6,  10.2,    47.0,     // 13
                              45.0,   87.1,   5.6,    80.0,     // 14
                              52.0,  101.3,  10.0,    98.0,     // 15
                              46.0,   94.5,   7.4,    95.0,     // 16
                              46.0,   87.0,   3.6,    18.0,     // 17
                              46.0,   94.5,   4.3,    12.0,     // 18
                              48.0,   90.5,   9.0,    99.0,     // 19
                              56.0,   95.7,   7.0,    99.0)     // 20

    /** Blood Pressure (BP) response/output vector
     */
    val y = VectorD (105.0, 115.0, 116.0, 117.0, 112.0, 121.0, 121.0, 110.0, 110.0, 114.0,
                     114.0, 115.0, 114.0, 106.0, 125.0, 114.0, 106.0, 113.0, 110.0, 122.0)

    val x01 = x(?, 0 to 1)                                          // first two columns (0 and 1) of data matrix x
    val xy  = x :^+ y                                               // combined data matrix including the response column
    val _1  = VectorD.one (x.dim)                                   // column of all ones
    val ox  = _1 +^: x                                              // data matrix with _1 prepended for intercept models
    val oxy = ox :^+ y                                              // combined data matrix with _1 prepended for intercept models

end Example_BPressure

import Example_BPressure._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_BPressureTest` main function tests the `Example_BPressure` by printing
 *  out of the vectors and matrices.
 *  > runMain scalation.modeling.example_BPressureTest
 */
@main def example_BPressureTest (): Unit =

    println (s"x_fname  = ${stringOf (x_fname)}")
    println (s"x        = $x")
    println (s"y        = $y")
    println (s"xy       = $xy")
    println (s"ox_fname = ${stringOf (ox_fname)}")
    println (s"ox       = $ox")
    println (s"oxy      = $oxy")

    new Plot (null, y, null, "y vs. t")

end example_BPressureTest



//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_BPressureTest2` main function tests the multi-collinearity method in
 *  the `Regression` class using the following regression equation.
 *      y  =  b dot x  =  b_1*x_1 + b_2*x_2 + b_3*x_3 + b_4 * x_4
 *  @see online.stat.psu.edu/online/development/stat501/12multicollinearity/05multico_vif.html
 *  @see online.stat.psu.edu/online/development/stat501/data/bloodpress.txt
 *  > runMain scalation.modeling.example_BPressureTest2
 */
@main def example_BPressureTest2 (): Unit =

    import Example_BPressure._

    var mod = new Regression (x, y)                                 // regression model x0, x1, x2, x3 with no intercept
    mod.inSample_Test ()                                            // train and test the model
    println (mod.summary ())                                        // parameter/coefficient statistics

    mod = new Regression (x01, y)                                   // regression model x0, x1 with no intercept
    mod.inSample_Test ()                                            // train and test the model
    println (mod.summary ())                                        // parameter/coefficient statistics

end example_BPressureTest2

