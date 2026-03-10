
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Sep 21 15:05:06 EDT 2009
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Quadratic (Quad) Curve
 */

package scalation
package scala2d

import java.awt.geom.Point2D.distance

import scala.math.{abs, pow, sqrt}

import scalation.mathstat.VectorD

import Colors._
import QCurve.calcControlPoint

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QCurve` class enhances the `QuadCurve.Double` class (from the `java.awt.geom`
 *  package) by allowing entities to move along such quadratic curves as well as
 *  lines.  Although the curve could be developed as a quadratic function where
 *  y = ax^2 + bx + c.  The following quadratic bezier formulation is used:
 *      p(t) = (x(t), y(t)) = [(1-t)^2 * p1] + [2 * (1-t) * t * pc] + [t^2 * p2].
 *  @param p1        the starting point for the quad curve
 *  @param pc        the control point for the quad curve
 *  @param p2        the ending point for the quad curve
 *  @param straight  whether the quad curve is straight (i.e., a line)
 */
case class QCurve (p1: R2  = new R2 (0.0, 0.0),
                   pc: R2  = new R2 (0.0, 0.0),
                   p2: R2  = new R2 (0.0, 0.0),
                   straight: Boolean = true)
     extends java.awt.geom.QuadCurve2D.Double (p1.x, p1.y, pc.x, pc.y, p2.x, p2.y)
     with CurvilinearShape:

    /** Trajectory parameter t ranges from 0. to 1. (indicates how far along the curve)
     */
    var _traj = 0.0

    /** Number of discrete steps to take along trajectory
     */
    private var steps = 200

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Construct a straight line (degenerate quad curve).
     *  @param _p1  the starting point
     *  @param _p2  the ending point
     */
    def this (_p1: R2, _p2: R2) =
        this (_p1, calcControlPoint (_p1, _p2), _p2, true)
    end this

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Construct a quad curve where bend indicates the distance to the control
     *  point.
     *  @param _p1   the starting point
     *  @param _p2   the ending point
     *  @param bend  the bend or curvature (1. => line length)
     */
    def this (_p1: R2, _p2: R2, bend: Double) =
        this (_p1, calcControlPoint (_p1, _p2, bend), _p2, false) 
    end this

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the current trajectory '_traj' of the curve.
     */
    def traj: Double = _traj

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the trajectory '_traj' to a new value.
     *  @param traj  the new trajectory for the curve
     */
    def traj_= (traj: Double): Unit = { _traj = traj }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the x-coordinate of the center of the line/curve.
     */
    def getCenterX: Double =
        if straight then (getX1 () + getX2 ()) / 2.0
        else             (getX1 () + 2.0 * getCtrlX () + getX2 ()) / 4.0
    end getCenterX

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the y-coordinate of the center of the line/curve.
     */
    def getCenterY: Double =
        if straight then (getY1 () + getY2 ()) / 2.0
        else             (getY1 () + 2.0 * getCtrlY () + getY2 ()) / 4.0
    end getCenterY

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set (or reset) the location for the `QCurve` as a line.
     *  @param _p1  the starting point
     *  @param _p2  the ending point
     */
    def setLine (_p1: R2, _p2: R2): Unit =
        val _pc = calcControlPoint (_p1, _p2)         // middle, on line => line
        setCurve (_p1, _pc, _p2)
    end setLine

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set (or reset) the location for the `QCurve` as a curve using bend to
     *  calculate the control point.
     *  @param _p1   the starting point
     *  @param _p2   the ending point
     *  @param bend  the bend or curvature (1. => line-length)
     */
    def setLine (_p1: R2, _p2: R2, bend: Double): Unit =
        val _pc = calcControlPoint (_p1, _p2, bend)   // off line => curve
        setCurve (_p1, _pc, _p2)
    end setLine

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set (or reset) the location for the `QCurve` as a curve using an explicitly
     *  given control point.
     *  @param _p1  the starting point
     *  @param _pc  the control point
     *  @param _p2  the ending point
     */
    override def setLine (_p1: R2, _pc: R2, _p2: R2): Unit =
        super.setCurve (_p1, _pc, _p2)
    end setLine

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the first/start point of the quad curve.
     */
    def getFirst: R2 = new R2 (getX1 (), getY1 ())

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the first/start point of the quad curve, adjusted from top-left to
     *  center coordinates.
     *  @param width   the width of object traversing the curve
     *  @param height  the height of object traversing the curve
     */
    def getFirst (width: Double, height: Double): R2 = 
        new R2 (getX1 () + width / 2.0, getY1 () + height / 2.0)
    end getFirst

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the control point of the quad curve.
     */
    def getControl: R2 = new R2 (getCtrlX (), getCtrlY ())
 
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the last/end-point of the quad curve.
     */  
    def getLast: R2 = new R2 (getX2 (), getY2 ())

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the last/end-point of the quad curve, adjusted from top-left to
     *  center coordinates.
     *  @param width   the width of object traversing the curve
     *  @param height  the height of object traversing the curve
     */
    def getLast (width: Double, height: Double): R2 = 
        new R2 (getX2 () + width / 2.0, getY2 () + height / 2.0)
    end getLast

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Whether ('x', 'y') and ('xe', 'ye') are essentially the same.
     */
    def isSame (x: Double, y: Double, xe: Double, ye: Double, step: Double): Boolean =
        (xe - x) * (xe - x) + (ye - y) * (ye -y) < step * step
    end isSame

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a value for the trajectory parameter 't' (in [0., 1.]) calculate
     *  the point on the curve using the Quadratic Bezier equation.
     *  @see en.wikipedia.org/wiki/Bézier_curve#Quadratic_curves
     */
    def eval (): R2 =
       new R2 (pow (1.0-_traj, 2) * getX1 () + 2.0 * (1.0-_traj) * _traj * getCtrlX () + pow (_traj, 2) * getX2 (),
           pow (1.0-_traj, 2) * getY1 () + 2.0 * (1.0-_traj) * _traj * getCtrlY () + pow (_traj, 2) * getY2 ())
    end eval

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the next point on the quad curve (one step beyond current point).
     *  Return null if 't > 1.0' (i.e., past end-point).
     */
    def next (): R2 =
        var q: R2 = null                          // the next point along the curve
        if _traj > 1.0 then _traj = 0.0           // reset trajectory
        else q = eval ()                          // calculate the new point
        _traj += 1.0 / steps.toDouble             // increment trajectory parameter
//      println ("QCurve.next: q = " + q)
        q
    end next

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the next point on the quad curve (one step beyond current point)
     *  and adjust from top-left to center coordinates for the object traversing
     *  the curve based on its width and height.
     *  Return null if 't > 1.0' (i.e., past end-point).
     *  @param width   the width of object traversing the curve
     *  @param height  the height of object traversing the curve
     */
    override def next (width: Double, height: Double): R2 =
        val q = next ()
        if q != null then new R2 (q.x - width / 2.0, q.y - height / 2.0) else null
    end next

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the number of steps for tokens to take as move along the quad curve.
     *  @param steps  the number of steps to take along the quad curve
     */
    def setSteps (_steps: Int): Unit = { steps = _steps }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the length of this `QCurve`.
     */
    def length: Double =
        val (x1, y1, xc, yc, x2, y2) = (getX1 (), getY1 (), getCtrlX (), getCtrlY (), getX2 (), getY2 ())
        (distance (x1, y1, x2, y2) +
        (distance (x1, y1, xc, yc) + distance (xc, yc, x2, y2))) / 2.0
    end length

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show the start, control and end-points of the the `QCurve`.
     */
    override def toString: String =
        s"QCurve (${getP1 ()}, $getCtrlPt ()}, $getP2 ()})"
    end toString

end QCurve


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QCurve` companion object provides formulas used by the `QCurve` class.
 */
object QCurve:

    /** Tolerance for comparing real numbers
     */
    private val EPSILON = 1E-7

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Calculate the slope of the line defined by points p1 and p2.
     *  Note:  if 'deltaX' is 0, the method returns infinity.
     *  @param p1  the starting point
     *  @param p2  the ending point
     */
    def slope (p1: R2, p2: R2): Double = (p2.y - p1.y) / (p2.x - p1.x)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Calculate the location (x, y) of the control point.  It is positioned
     *  orthogonal to the mid point of the line connecting p1 and p2 at a
     *  distance dist, where dist = bend * || p2 - p1 ||.  A bend of 0.0 gives
     *  a straight line, while 2.0/-2.0 gives a huge bend up-right/down-left.
     *  @param p1    the starting point
     *  @param p2    the ending point
     *  @param bend  the bend or curvature 
     */
    def calcControlPoint (p1: R2, p2: R2, bend: Double = 0.0): R2 =
        val mid = new R2 ((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0)
        if abs (bend) < EPSILON then
            mid
        else
            val m    = slope (p1, p2)
            val dist = bend * distance (p1.x, p1.y, p2.x, p2.y)
            if m.isInfinity then new R2 (mid.x + dist, mid.y)
            else new R2 (mid.x + dist * m / sqrt (1.0 + pow (m, 2)), mid.y - dist / sqrt (1.0 + pow (m, 2)))
        end if
    end calcControlPoint

    def calcControlPoint2 (p1: VectorD, p2: VectorD, bend: Double = 0.0): VectorD =
        val pc = calcControlPoint (new R2 (p1(0), p1(1)), new R2 (p2(0), p2(1)), bend)
        VectorD (pc.getX, pc.getY)
    end calcControlPoint2

end QCurve


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `qCurveTest` main function tests the `QCurve` classes' quad curves.
 *  > runMain scalation.scala2d.qCurveTest
 */
@main def qCurveTest (): Unit =

    banner ("Running qCurveTest")

    val line1 = new QCurve (new R2 (200, 200), new R2 (400, 200))
    val line2 = new QCurve (new R2 (200, 200), new R2 (200, 400))
    val line3 = new QCurve (new R2 (200, 200), new R2 (400, 400))

    val curve1 = new QCurve (new R2 (200, 200), new R2 (400, 200), 1.0)
    val curve2 = new QCurve (new R2 (200, 200), new R2 (200, 400), 1.0)
    val curve3 = new QCurve (new R2 (200, 200), new R2 (400, 400), 1.0)

    val curve4 = new QCurve (new R2 (200, 200), new R2 (400, 200), -2.0)
    val curve5 = new QCurve (new R2 (200, 200), new R2 (200, 400), -2.0)
    val curve6 = new QCurve (new R2 (200, 200), new R2 (400, 400), -2.0)

    class Canvas extends Panel:

        setBackground (white)

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Paint the components into the canvas (drawing panel).
         *  @param gr  low-resolution graphics environment
         */
        override def paintComponent (gr: Graphics): Unit =
            super.paintComponent (gr)
            val g2d = gr.asInstanceOf [Graphics2D]            // use hi-resolution
            g2d.setPaint (red)
            g2d.draw (line1)
            g2d.draw (curve1)
            g2d.draw (curve4)
            g2d.setPaint (blue)
            g2d.draw (line2)
            g2d.draw (curve2)
            g2d.draw (curve5)
            g2d.setPaint (purple)
            g2d.draw (line3)
            g2d.draw (curve3)
            g2d.draw (curve6)
        end paintComponent

    end Canvas

    // Put the drawing canvas in the visualization frame

    new VizFrame ("qCurveTest", new Canvas (), 600, 600)

end qCurveTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `qCurveTest2` main function tests traversal of `QCurve`'s (quad curves).
 *  > runMain scalation.scala2d.qCurveTest2
 */
@main def qCurveTest2 (): Unit =

    class QCurveAnimator extends VizFrame ("qCurveTest2", null, 600, 600) with Runnable:

        val curve = Array (new QCurve (new R2 (100, 200), new R2 (500, 200)),
                           new QCurve (new R2 (100, 200), new R2 (500, 200), .5),
                           new QCurve (new R2 (100, 200), new R2 (500, 200), -.5))
        val ball  = Ellipse ()

        def run (): Unit =
            val size    = 10.0
            var loc: R2 = null
             
            for i <- 0 until curve.length do
                println ("Move ball along RGB curve " + i)
                loc = curve(i).next (size, size)
                while loc != null do
                    Thread.sleep (50)
                    ball.setFrame (loc.x, loc.y, size, size)
                    repaint ()
                    loc = curve(i).next (size, size)
                end while
            end for
        end run

        class Canvas extends Panel:

            setBackground (white)

            //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
            /** Paint the components into the canvas (drawing panel).
             *  @param gr  low-resolution graphics environment
             */
            override def paintComponent (gr: Graphics): Unit =
                super.paintComponent (gr)
                val g2d = gr.asInstanceOf [Graphics2D]            // use hi-resolution
                g2d.setPaint (red)           // R in RGB order
                g2d.draw (curve(0))
                g2d.setPaint (green)         // G in RGB order
                g2d.draw (curve(1))
                g2d.setPaint (blue)          // B in RGB order
                g2d.draw (curve(2))
                g2d.setPaint (purple)
                g2d.fill (ball)
            end paintComponent

        end Canvas

        getContentPane ().add (new Canvas ())
        setVisible (true)

    end QCurveAnimator

    println ("Run qCurveTest2")
    new Thread (new QCurveAnimator ()).start ()

end qCurveTest2

