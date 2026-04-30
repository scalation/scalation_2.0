
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Oct 17 16:01:39 EDT 2011
 *  @see     LICENSE (MIT style license file). 
 *
 *  @note    Contour Plots for z = f(x, y) using color-coding for z
 *           Shows relative elevation by color and implicitly contour curves
 *
 *  see      www.math.uri.edu/~bkaskosz/fall21/sec-12-3.pdf
 */

package scalation
package mathstat

import scala.collection.mutable.ArrayBuffer
import scala.math.{ceil, floor, round}

import scalation.scala2d._
import scalation.scala2d.Colors._

import VectorDOps._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PlotC` class takes a function f and displays color-coded values for
 *  z = f(x, y) over a two dimensional grid defined the lower lb and upper ub bounds.
 *  An optional path is included that can be used to show, for example, the
 *  search path taken by an optimizer (e.g., a Conjugate Gradient NLP solver).
 *------------------------------------------------------------------------------
 *  Zoom functionality has two options:
 *  (1) mouse wheel controls the amount of zooming (in/out);
 *  (2) mouse dragging repositions the objects in the panel (drawing canvas).
 *  @see ZoomablePanel
 *------------------------------------------------------------------------------
 *  @param f       the function whose color-coded contour plot is sought
 *  @param lb      the lower bounds on the plotting domain
 *  @param ub      the upper bounds on the plotting domain
 *  @param path    the points on a path (e.g., a search path)
 *  @param opt     the optimal value if known (defaults to null)
 *  @param deltaF  estimate of the range of possible functional values (if < 0, will be computed)
 *  @param lbF     the lower bound on the functional value
 *  @param title   the title of the plot
 */
class PlotC (f: FunctionV2S, lb: VectorD, ub: VectorD, path: ArrayBuffer [VectorD] = null,
             opt: VectorD = VectorD.nullv, private var deltaF: Double = -1.0, private var lbF: Double = 0.0,
             title: String = "Contour Plot of f(x, y)")
      extends VizFrame (title, null):

    private val debug    = debugf ("PlotC", false)                    // debug function

    private val offset   = 50                                         // offset frame to axis
    private val frameW   = getW                                       // frame width
    private val frameH   = getH                                       // frame height
    private val baseX    = offset                                     // base for x-axis
    private val baseY    = frameH - offset                            // base for y-axis

    private val minX     = floor (lb(0))
    private val maxX     = ceil (ub(0))
    private val minY     = floor (lb(1))
    private val maxY     = ceil (ub(1))
    private val deltaX   = maxX - minX
    private val deltaY   = maxY - minY

    private val width_   = 9
    private val diameter = 6
    private val square   = Rectangle ()
    private val nsquares = 200.0                                      // number of squares per direction x, y
    private val dot      = Ellipse ()

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a canvas on which to draw the contour plot.
     */
    class Canvas
//        extends Panel:
          extends ZoomablePanel:

        setBackground (white)

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Paint the canvas by plotting color-coded squares representing the z-coordinate.
         *  @param gr  low-resolution graphics environment
         */
        override def paintComponent (gr: Graphics): Unit =
            super.paintComponent (gr)
            val g2d = gr.asInstanceOf [Graphics2D]                    // use hi-res graphics
            val mc  = 255                                             // max color value
            val c   = VectorD (1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.001)   // fractional value cutoffs
            val d   = 1.0 / c                                         // use reciprocal for multiplier

            g2d.setTransform (at)                                     // used for zooming (at @see `ZoomablePanel`)

            Plot.drawAxes (g2d, baseX, baseY, frameW, frameH, offset, minX, maxY, deltaX, deltaY)

            //:: Draw squares for the color-coded values of the points of the function being plotted

            var x_pos = 0
            var y_pos = 0

            var x = lb(0)
            while x <= ub(0) do
                var y = lb(1)
                while y <= ub(1) do
                    val vec  = VectorD (x, y)
                    val fv   = f(vec)
                    val frac = (fv - lbF) / deltaF                    // fractional way from lower to upper bound

                    // high to low: M, B, C, G, Y, R, W

                    val rgb = if frac > c(1) then
                        val z = (mc * (c(0) - frac)).toInt; (z, 0, z)          // magenta
                    else if frac > c(2) then
                        (0, 0, (mc * d(1) * (c(1) - frac)).toInt)              // blue
                    else if frac > c(3) then
                        val z = (mc * d(2) * (c(2) - frac)).toInt; (0, z, z)   // cyan
                    else if frac > c(4) then
                        (0, (mc * d(3) * (c(3) - frac)).toInt, 0)              // green
                    else if frac > c(5) then
                        val z = (mc * d(4) * (c(4) - frac)).toInt; (z, z, 0)   // yellow
                    else if frac > c(6) then
                        ((mc * d(5) * (c(5) - frac)).toInt, 0, 0)              // red
                    else
                        val z = (mc * d(6) * (c(6) - frac)).toInt; (z, z, z)   // gray
/*
                    val rgb = if frac > _2_3 then
                        ( ((frac-_2_3) * 765).toInt, ((1-frac) * 765).toInt, 0 )
                    else if frac > _1_3 then
                        ( 0, ((frac-_1_3) * 765).toInt, ((_2_3-frac) * 765).toInt )
                    else
                        ( ((_1_3-frac) * 400).toInt, 0, ((frac) * 765).toInt )
*/
                    debug ("paintComponent", s"(x, y) = $vec, fv = $fv, lbF = $lbF, frac = $frac, rgb = $rgb")
                    val color = new Color (rgb._1, rgb._2, rgb._3)
    
                    val xx    = round ((x - lb(0)) * (frameW - 2 * offset))
                    x_pos     = (xx / deltaX).toInt + offset
                    val yy    = round ((ub(1) - y) * (frameH - 2 * offset))
                    y_pos     = (yy / deltaY).toInt + offset - diameter
                    square.setFrame (x_pos, y_pos, width_, width_)    // x, y, w, h
                    g2d.setPaint (color)
                    g2d.fill (square)
                    y += deltaY / nsquares
                end while
                x += deltaX / nsquares
            end while

            //:: Draw the dots for the points on a search path, if given

            if path != null then
                val basicStroke   = g2d.getStroke.asInstanceOf [BasicStroke]
                val dashedLine    = Line (0, 0, 0, 0)
                val dashedStroke  = new BasicStroke (0.5, basicStroke.getEndCap, basicStroke.getLineJoin,
                                                     1.0, Array [Float](2), 0);
                var xPosPrev: Int = Int.MinValue
                var yPosPrev: Int = Int.MinValue

                for p <- path do
                    // Draw point in path.
                    val xx    = round ((p(0) - lb(0)) * (frameW - 2 * offset))
                    x_pos     = (xx / deltaX).toInt + offset
                    val yy    = round ((ub(1) - p(1)) * (frameH - 2 * offset))
                    y_pos     = (yy / deltaY).toInt + offset - diameter
                    dot.setFrame (x_pos, y_pos, diameter, diameter)   // x, y, w, h
                    g2d.setPaint (darkyellow)
                    g2d.fill (dot)

                    // Draw line connecting previous point to this point
                    if xPosPrev != Int.MinValue && yPosPrev != Int.MinValue then
                        dashedLine.setLine (xPosPrev + (diameter/2.0), yPosPrev + (diameter/2.0),
                                            x_pos  + (diameter/2.0), y_pos + (diameter/2.0))
                        g2d.setStroke (dashedStroke)
                        g2d.setPaint (black)
                        g2d.draw (dashedLine)
                        g2d.setStroke (basicStroke)

                    // Update previous positions.
                    xPosPrev = x_pos
                    yPosPrev = y_pos
            end if

            if opt != VectorD.nullv then
                val xx = round ((opt(0) - lb(0)) * (frameW - 2 * offset))
                x_pos = (xx / deltaX).toInt + offset
                val yy = round((ub(1) - opt(1)) * (frameH - 2 * offset))
                y_pos = (yy / deltaY).toInt + offset - diameter
                dot.setFrame (x_pos + (diameter/4.0), y_pos + (diameter/4.0),
                              diameter/2.0, diameter/2.0)                      // x, y, w, h
                g2d.setPaint (black)
                g2d.fill (dot)
            end if
        end paintComponent

    end Canvas

    if deltaF < 0.0 then resetBounds ()
    getContentPane ().add (new Canvas ())
    setVisible (true)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset the bounds on the functional values of f.  If the caller fails to
     *  provide an estimate for deltaF, this method should be called.
     */
    def resetBounds (): Unit =
        var minF = Double.PositiveInfinity
        var maxF = Double.NegativeInfinity

        var x = lb(0)
        while x <= ub(0) do
            var y = lb(1)
            while y <= ub(1) do
                val vec   = VectorD (x, y)
                val f_vec = f(vec)
                if f_vec < minF then minF = f_vec
                if f_vec > maxF then maxF = f_vec
                y += deltaY / nsquares
            end while
            x += deltaX / nsquares
        end while

        lbF    = minF                                                 // lower bounds on functional values for f
        deltaF = maxF - minF                                          // range of functional values for f
        debug ("resetBounds", s"lbF = $lbF, deltaF = $deltaF")
    end resetBounds

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert basic Contour information to a string.
     */
    override def toString: String = s"PlotC (lb = $lb, f(lb) = ${f(lb)}, ub = $ub, f(ub) = ${f(ub)})"

end PlotC


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `plotCTest` main function is used to test the `PlotC` class.  For additional examples,
 *  @see `RidgeRegression`, `LassoRegression`, `BridgeRegression` in `modeling`.
 *  @see scalation.scala2d.writeImage
 *  > runMain scalation.mathstat.plotCTest
 */
@main def plotCTest (): Unit =

    def f(x: VectorD): Double = (x(0)/2 - 3)~^2 + (x(1)/3 - 2)~^2

    val lb   = VectorD (0, 0)
    val ub   = VectorD (10, 10)
    val path = ArrayBuffer (VectorD (0, 0), VectorD (3, 2), VectorD (6, 6))
    val plot = new PlotC (f, lb, ub, path)
    println (s"plot = $plot")

    writeImage (DATA_DIR + "plotc.png", plot)

end plotCTest

