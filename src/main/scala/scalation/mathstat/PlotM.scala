
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Michael Cotterell
 *  @version 2.0
 *  @date    Sun Nov 15 15:05:06 EDT 2009
 *  @see     LICENSE (MIT style license file). 
 *
 *  @note    Plot Rows in Matrix y versus x
 */

package scalation
package mathstat

import scala.math.{ceil, floor, pow, round}

import scalation.scala2d._
import scalation.scala2d.Colors._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PlotM` class takes an x vector and a y matrix of data values and plots
 *  the (x, y_i) data points for each row y_i of the matrix.
 *------------------------------------------------------------------------------
 *  Zoom functionality has two options:
 *  (1) mouse wheel controls the amount of zooming (in/out);
 *  (2) mouse dragging repositions the objects in the panel (drawing canvas).
 *  @see ZoomablePanel
 *------------------------------------------------------------------------------
 *  @param x_     the x vector of data values (horizontal)
 *  @param y_     the y matrix of data values where y(i) is the i-th vector (vertical)
 *  @param label  the label/legend/key for each curve in the plot
 *  @param title  the title of the plot
 *  @param lines  flag for generating a line plot
 */
class PlotM (x_ : VectorD, y_ : MatrixD, var label: Array [String] = null,
             title: String = "PlotM y_i vs. x for each i", lines: Boolean = false)
      extends VizFrame (title, null):

    private val xa: VectorD = if x_ == null then VectorD.range (0, y_.dim2) else x_

    private val EPSILON  = 1E-9                                      // number close to zero
    private val offset   = 70                                        // offset frame to axis
    private val frameW   = getW                                      // frame width
    private val frameH   = getH                                      // frame height
    private val baseX    = offset                                    // base for x-axis
    private val baseY    = frameH - offset                           // base for y-axis

    private val minX     = floor (xa.min)
    private val maxX     = ceil (xa.max + EPSILON)
    private val minY     = floor (y_.mmin)
    private val maxY     = ceil (y_.mmax)
//  private val maxY     = ceil (y_.mmax + EPSILON)
    private val deltaX   = maxX - minX
    private val deltaY   = maxY - minY

    private val diameter  = 4
    private val dot       = Ellipse ()

    if label == null then label = defaultLabels

    println (s"x-axis: minX = $minX, maxX = $maxX")
    println (s"y-axis: minY = $minY, maxY = $maxY")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return default labels for y-vector.
     */
    def defaultLabels: Array [String] =
        val l = new Array [String] (y_.dim)
        for i <- y_.indices do l(i) = "Vector" + i
        l
    end defaultLabels

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a canvas on which to draw the plot.
     */
    class CanvasP
//        extends Panel:
          extends ZoomablePanel:

        setBackground (white)
        private val colors = Array (red, green, blue, black, yellow, cyan, magenta)

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Paint the canvas by plotting the data points.
         *  @param gr  low-resolution graphics environment
         */
        override def paintComponent (gr: Graphics): Unit =
            super.paintComponent (gr)
            val g2d = gr.asInstanceOf [Graphics2D]                   // use hi-res graphics

            g2d.setTransform (at)                                    // used for zooming (at @see `ZoomablePanel`)

            Plot.drawAxes (g2d, baseX, baseY, frameW, frameH, offset, minX, maxY, deltaX, deltaY)

            //:: Draw the color keys below the x-axis

            g2d.drawString ("Key:", offset, frameH - 30)

            //:: Draw the dots for the data points being plotted

            var x_pos  = 0                                           // current x position
            var y_pos  = 0                                           // current y position

            for i <- 0 until y_.dim do
                val y_i = y_(i)
//              val color = randomColor (i)
                val color = colors (i % colors.size)
                g2d.setPaint (color)
                if i < label.length then g2d.drawString (label(i), offset * (i + 2), frameH - 30)

                var px_pos = 0                                       // previous x position
                var py_pos = 0                                       // previous y position

                for j <- xa.indices do
                    val xx = round ((xa(j) - minX) * (frameW - 2 * offset))
                    x_pos  = (xx / deltaX).toInt + offset
                    val yy = round ((maxY - y_i(j)) * (frameH - 2 * offset))
                    y_pos  = (yy / deltaY).toInt + offset
                    dot.setFrame (x_pos, y_pos, diameter, diameter)  // x, y, w, h
                    g2d.fill (dot)

                    // connect with lines
                    if j != 0 && lines then
                        g2d.setStroke (new BasicStroke (1.0f))
                        g2d.drawLine (px_pos+1, py_pos+1, x_pos+1, y_pos+1)

                    px_pos = x_pos                                   // update previous x
                    py_pos = y_pos                                   // update previous y

                end for
            end for
        end paintComponent

    end CanvasP

/*
    import java.awt.image.BufferedImage
    import java.io.File
    import javax.imageio.ImageIO
*/

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Save the plot in a file as an image.
     *  @see stackoverflow.com/questions/5655908/export-jpanel-graphics-to-png-or-gif-or-jpg/39490801#39490801
     *  @param fname  the name of the file to save the plot in
    def saveImage (fname: String): Unit =
        val bimg = new BufferedImage (getSize ().width, getSize ().height, BufferedImage.TYPE_INT_ARGB)
        val gr   = bimg.createGraphics ()
        paint (gr)
        gr.dispose ()
        ImageIO.write (bimg, "png", new File (fname))
    end saveImage
     */

    getContentPane ().add (new CanvasP ())
    setVisible (true)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this `PlotM` object to a string.
     */
    override def toString: String = s"PlotM (y = $y_ vs. x = $xa)"

end PlotM


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PlotM` companion object provides a builder method for plotting several
 *  y vectors versus an x vector.
 */
object PlotM:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a plot of several y vectors versus an x vector.
     *  @param x  the x vector of data values (horizontal)
     *  @param y  one or more vectors of values where y(i) is the i-th vector (vertical)
     */
    def apply (x: VectorD, y: VectorD*): PlotM =
        val yy = new MatrixD (y.length, x.dim)
        for i <- 0 until y.length do yy(i) = y(i)
        new PlotM (x, yy)
    end apply

end PlotM


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `plotMTest` main functionb is used to test the `PlotM` class.
 *  > runMain scalation.mathstat.plotMTest
 */
@main def plotMTest (): Unit =

    val x = new VectorD (200)
    val y = new MatrixD (5, 200)

    for i <- 0 until 200 do
        x(i)    = (i - 100) / 10.0
        y(0, i) = 10.0 * x(i)
        y(1, i) = pow (x(i), 2)
        y(2, i) = .1 * pow (x(i), 3)
        y(3, i) = .01 * pow (x(i), 4)
        y(4, i) = .001 * pow (x(i), 5)
    end for
    val plot = new PlotM (x, y, Array ("Linear", "Quadratic", "Cubic", "Quartic", "Quintic"))
    println (s"plot = $plot")

end plotMTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `plotMTest2` main function is used to test the `PlotM` class.  This
 *  version also plots lines connecting the points.
 ^  @see scalation.scala2d.writeImage
 *  > runMain scalation.mathstat.plotMTest2
 */
@main def plotMTest2 (): Unit =

    val x = new VectorD (200)
    val y = new MatrixD (6, 200)

    for i <- 0 until 200 do
        x(i)    = (i - 100) / 10.0
        y(0, i) = 10.0 * x(i)
        y(1, i) = pow (x(i), 2)
        y(2, i) = .1 * pow (x(i), 3)
        y(3, i) = .01 * pow (x(i), 4)
        y(4, i) = .001 * pow (x(i), 5)
        y(5, i) = .0001 * pow (x(i), 6)
    end for
    val plot = new PlotM (x, y, Array ("Linear", "Quadratic", "Cubic", "Quartic", "Quintic"), lines = true)
    println (s"plot = $plot")

    writeImage (DATA_DIR + "plotm.png", plot)

end plotMTest2

