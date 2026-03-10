
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yulong Wanag
 *  @version 2.0
 *  @date    Sun April  23 15:13:42 EDT 2023
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Bounce Back at Constraint Boundary
 */

package scalation
package optimization

import scalation.mathstat.VectorD

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BoundsConstraint` trait provides a mechanism for bouncing back at
 *  constraint boundaries.
 *  @oaram lower  the lower bound constraint vector
 *  @oaram upper  the upper bound constraint vector
 */
trait BoundsConstraint (lower: VectorD = null, upper: VectorD = null):

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Constraint the current point x, so that lower <= x <= upper by bouncing
     *  back from a violated contraint.
     *  !@param x  the current point
     */
    def constrain (x: VectorD): Unit =
        if lower != null then
            for i <- lower.indices if x(i) < lower(i) do
                x(i) = if lower(i) < 0 then lower(i) * 0.95 else lower(i) * 1.05
        if upper != null then
            for i <- upper.indices if x(i) > upper(i) do
                x(i) = if upper(i) > 0 then upper(i) * 0.95 else upper(i) * 1.05
        end if
    end constrain

end BoundsConstraint

