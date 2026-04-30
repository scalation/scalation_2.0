
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Jan  9 14:06:47 EST 2010
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Base Trait for Curvilinear Shapes (Lines and Curves)
 */

package scalation
package scala2d

import scalation.mathstat.VectorD

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CurvilinearShape` trait provides a general type for lines and curves.
 *  It is analogous to `RectangularShape`.
 */
trait CurvilinearShape extends Shape:

    private val flaw = flawf ("CurvilinearShape")                 // flaw function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the x-coordinate of the center of a line/curve.  This method must
     *  be implemented by all classes mixing in this trait.
     */
    def getCenterX: Double

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the y-coordinate of the center of a line/curve.  This method must
     *  be implemented by all classes mixing in this trait.
     */
    def getCenterY: Double

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set (or reset) the location for a straight line.  This method must
     *  be implemented by all classes mixing in this trait.
     *  @param p1  the starting point
     *  @param p2  the ending point
     */
    def setLine (p1: R2, p2: R2): Unit

    def setLine (p1: VectorD, p2: VectorD): Unit =
        setLine (new R2 (p1(0), p1(1)), new R2 (p2(0), p2(1)))
    end setLine

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set (or reset) the location for a line/curve using the bend parameter
     *  to compute the control point.  This method must be implemented by all
     *  classes mixing in this trait.
     *  @param p1    the starting point
     *  @param p2    the ending point
     *  @param bend  the bend or curvature (1. => line-length)
     */
    def setLine (p1: R2, p2: R2, bend: Double): Unit

    def setLine (p1: VectorD, p2: VectorD, bend: Double): Unit =
        setLine (new R2 (p1(0), p1(1)), new R2 (p2(0), p2(1)), bend)
    end setLine

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set (or reset) the location for a line/curve using an explicitly
     *  given control point.  This is an optional method.
     *  @param p1  the starting point
     *  @param pc  the control point
     *  @param p2  the ending point
     */
    def setLine (p1: R2, pc: R2, p2: R2): Unit =
        flaw (s"setLine (p1 = $p1, pc = $pc, p2 = $p2)", "method is not overridden by mixin class")
    end setLine

    def setLine (p1: VectorD, pc: VectorD, p2: VectorD): Unit =
        setLine (new R2 (p1(0), p1(1)), new R2 (pc(0), pc(1)), new R2 (p2(0), p2(1)))
    end setLine

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the next point on the `CurvilinearShape` (one step beyond current point)
     *  and adjust from top-left to center coordinates for the object traversing
     *  the curve based on its width and height.  Return null if past end point.
     *  This is an optional method.
     *  @param width   the width of object traversing the curve
     *  @param height  the height of object traversing the curve
     */
    def next (width: Double, height: Double): R2 =
        flaw (s"next (width = $width, height = $height)", "method is not overridden by mixin class")
        null
    end next

end CurvilinearShape

