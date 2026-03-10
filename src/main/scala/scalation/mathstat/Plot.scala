
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Michael Cotterell
 *  @version 2.0
 *  @date    Sun Nov 15 15:05:06 EDT 2009
 *  @see     LICENSE (MIT style license file). 
 *
 *  @note    Plot Vectors y and z vs. x
 */

package scalation
package mathstat

import scala.math.{ceil, floor, min, pow, round}

import scalation.scala2d._
//import scalation.scala2d.BorderLayout._
import scalation.scala2d.Colors._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Plot` class takes x and y vectors of data values and plots the (x, y)
 *  data points.  Optionally, a z vector may be plotted with y.  Note, axes are
 *  determined by the x and y vectors only.  For more vertical vectors use `PlotM`.
 *------------------------------------------------------------------------------
 *  Zoom functionality has two options:
 *  (1) mouse wheel controls the amount of zooming (in/out);
 *  (2) mouse dragging repositions the objects in the panel (drawing canvas).
 *  @see ZoomablePanel
 *------------------------------------------------------------------------------
 *  @param x       the x vector of data values (horizontal), use null to use y's index
 *  @param y       the y vector of data values (primary vertical, black)
 *  @param z       the z vector of data values (secondary vertical, red) to compare with y
 *  @param title   the title of the plot
 *  @param lines   flag for generating a line plot
 */
class Plot (x: VectorD, y: VectorD, z: VectorD = null, title: String = "Plot y vs. x", lines: Boolean = false)
      extends VizFrame (title, null):

    private val xx: VectorD = if x == null then VectorD.range (0, y.dim) else x
    private val canvas      = new Canvas (xx, y, z, getW, getH, lines)

    getContentPane.add (canvas, BorderLayout.CENTER)
    setVisible (true)

end Plot


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Plot` companion object provides factory methods for creating plots.
 */
object Plot:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a plot object from integer vectors.
     *  @param x      the x vector of data values (horizontal)
     *  @param y      the y vector of data values (primary vertical)
     *  @param z      the z vector of data values (secondary vertical) to compare with y
     *  @param title  the title of the plot
     *  @param lines  flag for generating a line plot
     */
    def apply (x: VectorI, y: VectorI, z: VectorI = null, title: String, lines: Boolean = false): Plot =
        new Plot (x.toDouble, y.toDouble, if z == null then null else z.toDouble, title, lines)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Draw the x-axis and y-axis.
     *  @param g2d     the hi-res graphics
     *  @param baseX   the base for x
     *  @param baseY   the base for y
     *  @param frameW  the frame width
     *  @param frameH  the frame width
     *  @param offset  the offset in y coordinate
     */
    def drawAxes (g2d: Graphics2D, baseX: Int, baseY: Int, frameW: Int, frameH: Int,
                  offset: Int, minX: Double, maxY: Double, deltaX: Double, deltaY: Double): Unit =

        val stepsX = 10                                      // number of x-steps for axis
        val stepsY = 10                                      // number of y-steps for axis

        val axis = Line (0, 0, 0, 0)
        g2d.setPaint (black)
        g2d.setStroke (new BasicStroke (2.0f))

        // Draw the x-axis and y-axis

        axis.setLine (baseX - 1, baseY + 1, baseX + 10 + frameW - 2 * offset, baseY + 1)
        g2d.draw (axis)
        axis.setLine (baseX - 1, offset - 10, baseX - 1, baseY + 1)
        g2d.draw (axis)

        // Draw the labels on the x-axis

        var x_pos = 0
        var y_pos = baseY + 15
        var step  = deltaX / stepsX                          // for x-axis
        for j <- 0 to stepsX do
            val x_val = clip (minX + j * step)
            x_pos = offset - 8 + j * (frameW - 2 * offset) / stepsX
            g2d.drawString (x_val, x_pos, y_pos)
        end for

        // Draw the labels on the y-axis

        x_pos = baseX - 40 
        step  = deltaY / stepsY                              // for y-axis
        for j <- 0 to stepsY do
            val y_val = clip (maxY - j * step)
            y_pos = offset + 2 + j * (frameH - 2 * offset) / stepsY
            g2d.drawString (y_val, x_pos, y_pos)
        end for
    end drawAxes

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert value to string and cut out the first five characters.
     *  Caveat: only work for numbers < million, may rescale data otherwise
     *  @param x  the value to convert and clip/cut
     */
    def clip (x: Double): String =
        val s = x.toString 
        s.substring (0, min (s.length, 5))
    end clip

end Plot


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `FramelessPlot` class is used for embedded applications.
 *  @param x       the x vector of data values (horizontal)
 *  @param y       the y vector of data values (primary vertical)
 *  @param z       the z vector of data values (secondary vertical) to compare with y
 *  @param width   the width
 *  @param height  the height
 */
class FramelessPlot (x: VectorD, y: VectorD, z: VectorD = null, var width: Int = 840, var height: Int = 480):

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Dynamically create and return a drawing canvas.
     */
    def canvas: Canvas = new Canvas (x, y, z, width, height)

end FramelessPlot
 

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Canvas` class provides a canvas on which to draw the plot.
 *  @param x       the x vector of data values (horizontal)
 *  @param y       the y vector of data values (primary vertical)
 *  @param z       the z vector of data values (secondary vertical) to compare with y
 *  @param width   the frame width
 *  @param height  the frame height
 *  @param lines   flag for generating a line plot
 */
class Canvas (x: VectorD, y: VectorD, z: VectorD, width: Int, height: Int, lines: Boolean = false)
//    extends Panel:
      extends ZoomablePanel:

    private val EPSILON  = 1E-9                               // number close to zero
    private val SCALE    = 10                                 // FIX - pass as a parameter
    private val offset   = 80                                 // offset frame to axis
    private val frameW   = width                              // frame width
    private val frameH   = height                             // frame height
    private val baseX    = offset                             // base for x-axis
    private val baseY    = frameH - offset                    // base for y-axis

    private val minX     = floor (SCALE * x.min) / SCALE.toDouble
    private val maxX     = ceil (x.max + EPSILON)
    private val minY     = floor (SCALE * y.min) / SCALE.toDouble
    private val maxY     = ceil (y.max)
//  private val maxY     = ceil (y.max + EPSILON)
    private val deltaX   = maxX - minX
    private val deltaY   = maxY - minY

    private val diameter = 4
    private val dot      = Ellipse ()

    setBackground (white)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Paint the canvas by plotting the data points.
     *  @param gr  low-resolution graphics environment
     */
    override def paintComponent (gr: Graphics): Unit =
        super.paintComponent (gr)
        val g2d = gr.asInstanceOf [Graphics2D]                // use hi-res graphics

        g2d.setTransform (at)                                 // used for zooming (at @see `ZoomablePanel`)

        Plot.drawAxes (g2d, baseX, baseY, frameW, frameH, offset, minX, maxY, deltaX, deltaY)

        //:: Draw the dots for the data points being plotted

        var x_pos  = 0                                        // current x position
        var y_pos  = 0                                        // current y position
        var px_pos = 0                                        // previous x position
        var py_pos = 0                                        // previous y position

        for i <- 0 until y.dim do
            val xx = round ((x(i) - minX) * (frameW - 2 * offset))
            x_pos = (xx / deltaX).toInt + offset
            val yy = round ((maxY - y(i)) * (frameH - 2 * offset))
            y_pos = (yy / deltaY).toInt + offset
            dot.setFrame (x_pos, y_pos, diameter, diameter)   // x, y, w, h

            g2d.setPaint (black)
            g2d.fill (dot)

            // connect with lines
            if i != 0 && lines then
                g2d.setStroke (new BasicStroke (1.0f))
                g2d.drawLine (px_pos+1, py_pos+1, x_pos+1, y_pos+1)

            px_pos = x_pos                                    // update previous x
            py_pos = y_pos                                    // update previous y
        end for

        g2d.setStroke (new BasicStroke (2.0f))

        if z != null then
            for i <- 0 until min (y.dim, z.dim) do
                val xx = round ((x(i) - minX) * (frameW - 2 * offset))
                x_pos = (xx / deltaX).toInt + offset
                val yy = round ((maxY - z(i)) * (frameH - 2 * offset))
                y_pos = (yy / deltaY).toInt + offset
                dot.setFrame (x_pos, y_pos, diameter, diameter)   // x, z, w, h
                g2d.setPaint (red)
                g2d.fill (dot)

                // connect with lines
                if i != 0 && lines then
                    g2d.setStroke (new BasicStroke (1.0f))
                    g2d.drawLine (px_pos+1, py_pos+1, x_pos+1, y_pos+1)

                px_pos = x_pos                                // update previous x
                py_pos = y_pos                                // update previous y
            end for
        end if

    end paintComponent

end Canvas


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `plotTest` main function is used to test the `Plot` class.
 *  @see scalation.scala2d.writeImage
 *  > runMain scalation.mathstat.plotTest
 */
@main def plotTest (): Unit =

    val x = new VectorD (100)
    val y = new VectorD (100)
    for i <- 0 until 100 do { x(i) = i / 10.0; y(i) = pow (x(i) - 5, 2) }
    new Plot (x, y, null, "plot1", lines = true)
    new Plot (null, y, null, "plot2", lines = true)

//  val plot = new Plot (null, y, null, "plot2", lines = true)
//  writeImage (DATA_DIR + "plot.png", plot)

end plotTest

