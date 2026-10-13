
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Tue May 26 18:21:28 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: ARMA Model Coefficient Stability Check
 */

package scalation
package modeling
package forecasting

import scala.math.{abs, pow}

import scalation.mathstat.VectorD

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARMA_Stability` object allows stability of ARMA (p, q) coefficient φ and θ
 *  values to be checked to make sure the roots of the characteristic polynomial are
 *  outside the unit circle.  It they are not, the fitted model may be unstable with
 *  poor accuracy in certain cases.
 */
object ARMA_Stability:

    private val debug = debugf ("ARMA_Stability", true)                    // debug function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check if the polynomial 1 - c1*z^-1 - c2*z^-2 ... is stable using the Jury Criterion.
     *  Check the stability/invertibility of ARMA parameters using the Jury Criterion.
     *
     *  Mathematical Mapping: Given the characteristic equation derived from time-lags:
     *      1 - c_1*z^(-1) - c_2*z^(-2) - ... - c_k*z^(-k) = 0
     *
     *  We multiply through by z^k to convert it to standard forward polynomial form:
     *      P(z) = a_0*z^k + a_1*z^(k-1) + a_2*z^(k-2) + ... + a_k = 0
     *
     *  This creates the mapping array 'a' where:
     *     a_0 = 1.0
     *     a_i = -c_i (for AR components) or +c_i (for additive MA components)
     *
     *  @note:  Bypasses asymmetric matrix eigenvalue solvers and complex numbers completely.
     *  @param c        the coefficients to check (pass φ for stationarity or θ for invertability)
     *                  a = [ 1.0, -φ_1, -φ_2, ... -φ_p ]  where φ_1 is the lag-1 AR coefficient
     *                  a = [ 1.0,  θ_1,  θ_2, ...  θ_q ]  where θ_1 is the lag-1 MA coefficient
     *  @param isPhi    whether the coefficient vector passed in is φ (for AR) or false θ (for MA)
     *  @param reverse  whether the vector is in historical order/oldest first (true) φ ∙ [ y_t-p, ... y_t-2, y_t-1 ]
     *                  or standard polynomial order (false)  φ ∙ [ y_t-1, y_t-2, ... y_t-p ]
     */
    def isStable (c: VectorD, isPhi: Boolean = true, reverse: Boolean = true): Int =
        val k = c.length
        if k == 0 then return 0                                           // nothing to check, return okay/true (0)
        
        // Convert to standard polynomial form: P(z) = a_0*z^k + a_1*z^(k-1) + ... + a_k = 0
        val a = new VectorD (k+1)
        a(0)  = 1.0
        if reverse then
            for i <- 1 to k do a(i) = if isPhi then -c(k-i) else c(k-i)   // reverse order
        else
            for i <- 1 to k do a(i) = if isPhi then -c(i-1) else c(i-1)   // same order

        debug ("isStable", s"coefficients in standard polynomial form a = $a")

        if a.sum <= 0.0 then return -1                                    // required condition 1: P(1) > 0

        var pMinus1 = 0.0
        for i <- 0 to k do pMinus1 += a(i) * pow (-1.0, k-i)              // required condition 2: (-1)^k * P(-1) > 0
        if pow (-1.0, k) * pMinus1 <= 0.0 then return -2

        if abs (a(k)) >= 1.0 then return -3                               // required condition 3: |a_k| < a_0

        // Sufficiency Check via Jury Table reduction
        if k >= 2 then
            var curRows = a
            cfor (0, k) { i =>
                val n = curRows.length
                val nextRows = new VectorD (n-1)
                val first    = curRows(0)
                val last     = curRows(n-1)
                
                debug ("isStable", s"step i = $i: abs ($first) <= abs ($last) = ${abs (first) <= abs (last)}")
                if abs (first) <= abs (last) then return -4              // required condition 4: strict contraction
                
                // Determinant contraction
                for i <- 0 until n-1 do nextRows(i) = first * curRows(i) - last * curRows(n-1-i)
                
                curRows = nextRows
            } // cfor
        end if

        0                                                                // passed all checks, return okay/true (0)
    end isStable

end ARMA_Stability


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRMA_StabilityTest` main function test the `ARMA_Stability` object.
 *  > runMain scalation.modeling.forecasting.aRMA_StabilityTest
 */
@main def aRMA_StabilityTest (): Unit =

    import ARMA_Stability.isStable

    banner ("aRMA_StabilityTest: Testing ARMA Jury Stability Criterion")

    // Test 1: Perfectly stable AR(1) parameter
    val phiStable1 = VectorD (0.5)
    val res1 = isStable (phiStable1, isPhi = true)
    println (s"Test 1 [Stable AR(1)]:     phi = [0.5]         -> Expected:  0, Actual: $res1")

    // Test 2: Fails Condition 1 (P(1) <= 0)
    val phiFail1 = VectorD (1.25)
    val res2 = isStable (phiFail1, isPhi = true)
    println (s"Test 2 [P(1) Failure]:      phi = [1.25]        -> Expected: -1, Actual: $res2")

    // Test 3: Fails Condition 2 (P(-1) Check)
    val phiFail2 = VectorD (-1.5)
    val res3 = isStable (phiFail2, isPhi = true)
    println (s"Test 3 [P(-1) Failure]:     phi = [-1.5]        -> Expected: -2, Actual: $res3")

    // Test 4: Fails Condition 3 (|a_k| >= 1.0 variance limit boundary)
    val phiFail3 = VectorD (-1.2, 0.2)                         // oldest first format maps to a = [1.0, -0.2, 1.2]
    val res4 = isStable (phiFail3, isPhi = true)
    println (s"Test 4 [|a_k| Boundary]:    phi = [-1.2, 0.2]   -> Expected: -3, Actual: $res4")

    // Test 5: Fails Condition 4 (Slips past basic checks, breaks deep inside table row reduction)
    val phiFail4 = VectorD (0.9, -0.3, -0.5)
    val res5 = isStable (phiFail4, isPhi = true)
    println (s"Test 5 [Table Contraction]: phi = [0.9, -0.3, -0.5] -> Expected: -4, Actual: $res5")

    // Test 6: Stable high order MA(2) setup
    val thetaStable2 = VectorD (0.1, 0.4)
    val res6 = isStable (thetaStable2, isPhi = false)
    println (s"Test 6 [Stable MA(2)]:     theta = [0.1, 0.4]  -> Expected:  0, Actual: $res6")

    println ("=" * 70)
    if res1 == 0 && res2 == -1 && res3 == -2 && res4 == -3 && res5 == -4 && res6 == 0 then
        println ("ALL TESTS PASSED: Numeric exit signals mapped exactly to theoretical errors.")
    else
        println ("TEST FAILURE: Check underlying loop index array bindings.")
    println ("=" * 70)

end aRMA_StabilityTest

