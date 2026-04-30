
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Nirupom Bose Roy
 *  @version 2.0
 *  @date    Sun Mar 06 05:30:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Test Util for LBFGS_B
 */

package scalation.optimization.quasi_newton

import scalation.mathstat.{VectorD, FunctionV2S}
import scalation.banner

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Utility helpers for L-BFGS-B benchmark tests.
 */
object LBFGS_B_TestUtil:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Run one L-BFGS-B benchmark and print the result.
     *  @param name     benchmark/test name
     *  @param f        objective function
     *  @param x0       starting point
     *  @param bounds   optional box bounds; if null, uses default infinite bounds
     *  @param exactLS  whether to use exact 1D line search instead of More-Thuente
     */
    def runLBFGSB (name: String,
                   f: FunctionV2S,
                   x0: VectorD,
                   bounds: Bounds = null,
                   exactLS: Boolean = false): (Double, VectorD)=

        banner (s"L-BFGS-B Test: $name | exactLS = $exactLS")
        val opt = if bounds == null then new LBFGS_B (f, exactLS = exactLS)
                  else new LBFGS_B (f, exactLS = exactLS, l_u = bounds)
        val res = opt.solve (x0)
        println (s"result = $res")
        res
    end runLBFGSB

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Run both exact scalar line search and native More-Thuente for comparison.
     *  @param name    benchmark/test name
     *  @param f       objective function
     *  @param x0      starting point
     *  @param bounds  optional box bounds
     */
    def compareLS (name: String,
                   f: FunctionV2S,
                   x0: VectorD,
                   bounds: Bounds = null): Unit =

        banner (s"L-BFGS-B Line Search Comparison: $name")
        val resMT = if bounds == null then new LBFGS_B (f, exactLS = false).solve (x0)
                    else new LBFGS_B (f, exactLS = false, l_u = bounds).solve (x0)

        val res1D = if bounds == null then new LBFGS_B (f, exactLS = true).solve (x0)
                    else new LBFGS_B (f, exactLS = true, l_u = bounds).solve (x0)

        println (s"More-Thuente result = $resMT")
        println (s"Exact 1D result     = $res1D")
    end compareLS

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Run a bounded optimizer and the unconstrained native LBFGS from the same
     *  starting point under inactive bounds to compare solutions.
     *  @param name    benchmark/test name
     *  @param f       objective function
     *  @param x0      starting point
     *  @param bounds  wide/inactive bounds
     */
    def compareToLBFGS (name: String,
                        f: FunctionV2S,
                        x0: VectorD,
                        bounds: Bounds): Unit =

        banner (s"L-BFGS vs L-BFGS-B Comparison: $name")
        val resB = new LBFGS_B (f, l_u = bounds).solve (x0)
        val resU = LBFGS (f).solve (x0)
        println (s"L-BFGS-B result = $resB")
        println (s"L-BFGS   result = $resU")
    end compareToLBFGS

end LBFGS_B_TestUtil

