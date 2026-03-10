
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Nov 2 15:05:06 EDT 2009
 *  @see     LICENSE (MIT style license file). 
 *
 *  @note    Histogram from Vector of Data
 */

package scalation
package mathstat

import scala.math.{ceil, floor, min}

import scalation.random.{Normal, Uniform}
import scalation.scala2d.{BasicStroke, Graphics, Graphics2D, Line, Rectangle, VizFrame, ZoomablePanel}
import scalation.scala2d.Colors._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Histogram` class takes a vector of values, counts the number of values
 *  in each of several intervals and displays the counts vertically in a histogram.
 *------------------------------------------------------------------------------
 *  Zoom functionality has two options:
 *  (1) mouse wheel controls the amount of zooming (in/out);
 *  (2) mouse dragging repositions the objects in the panel (drawing canvas).
 *  @see ZoomablePanel
 *------------------------------------------------------------------------------
 *  @param value         the vector of values (want several per interval)
 *  @param numIntervals  the number of intervals (typically 5 to 100)
 *  @param _title        title of the histogram
 *  @param counts        the counts per interval, if available
 */
class Histogram (value: VectorD, numIntervals: Int = 40, _title: String = "Histogram", counts: VectorD = null)
      extends VizFrame (_title, null):

    /** Create a drawing canvas
     */
    private val canvas = new HCanvas (getW, getH, value, numIntervals, counts)

    getContentPane ().add (canvas)
    setVisible (true)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert a Histogram to a string.
     */
    override def toString: String = canvas.toString

end Histogram


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `FramelessHistogram` class should be used in embedded applications.
 *  @param frameW        the width
 *  @param frameH        the height
 *  @param value         the vector of values (want several per interval)
 *  @param numIntervals  the number of intervals (typically 5 to 100)
 */
class FramelessHistogram (frameW: Int, frameH: Int, value: VectorD, numIntervals: Int):

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Dynamically create and return a drawing canvas.
     */
    def canvas: HCanvas = new HCanvas (frameW, frameH, value, numIntervals)

end FramelessHistogram


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Create a canvas on which to draw the histogram.
 *  @param frameW        the width
 *  @param frameH        the height
 *  @param value         the vector of values (want several per interval)
 *  @param numIntervals  the number of intervals (typically 5 to 100)
 *  @param counts        the counts per interval, if available
 */
class HCanvas (frameW: Int, frameH: Int, value: VectorD, numIntervals: Int, counts: VectorD = null)
//    extends Panel:
      extends ZoomablePanel:

    private val EPSILON       = 1E-9
    private val offset        = 50
    private val baseX         = offset
    private val baseY         = frameH - offset
    private val minValue      = floor (value.min)
    private val maxValue      = ceil (value.max + EPSILON)
    private val intervalWidth = (maxValue - minValue) / numIntervals.toDouble  // + EPSILON
    private val histogram     = computeHistogram ()
    private val maxHistogram  = histogram.max
    private val c             = computeCoordinates ()
    private val bar           = Rectangle ()
    private val axis          = Line (0, 0, 0, 0)

    setBackground (white)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Paint the canvas by drawing the rectangles (vertical bars) making up
     *  the histogram.
     *  @param gr  low-resolution graphics environment
     */
    override def paintComponent (gr: Graphics): Unit =
        super.paintComponent (gr)
        val g2d = gr.asInstanceOf [Graphics2D]                // use hi-res

        g2d.setTransform (at)                                 // used for zooming (at @see `ZoomablePanel`)

        var x_pos = 0
        var y_pos = 0
        var step  = 0.0

        //:: Draw the axes

        g2d.setPaint (black)
        g2d.setStroke (new BasicStroke (2.0f))
        axis.setLine (baseX - 1, baseY + 1, baseX + 10 + frameW - 2 * offset, baseY + 1)
        g2d.draw (axis)
        axis.setLine (baseX - 1, offset - 10, baseX - 1, baseY + 1)
        g2d.draw (axis)

        //:: Draw the labels on the axes

        y_pos = baseY + 15
        step  = (maxValue - minValue) / 10.0
        for j <- 0 to 10 do
            val x_val = clip (minValue + j * step)
            x_pos = offset - 8 + j * (frameW - 2 * offset) / 10 
            g2d.drawString (x_val, x_pos, y_pos)
        end for

        x_pos = baseX - 30
        for j <- 0 to 20 do
            val y_val = clip ((20 - j) * maxHistogram / 20)
            y_pos = offset + 2 + j * (frameH - 2 * offset) / 20 
            g2d.drawString (y_val, x_pos, y_pos)
        end for

        //:: Draw the bars making up the histogram

        for i <- c.indices do
            bar.setFrame (c(i, 0), c(i, 1), c(i, 2), c(i, 3))    // x, y, w, h
            g2d.setPaint (blue)
            g2d.fill (bar)
            g2d.setPaint (black)
            g2d.draw (bar)
        end for
    end paintComponent

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert value to string and cut out the first four characters.
     *  @param x  the value to convert and cut
     */
    def clip (x: Double): String =
        val s = x.toString 
        s.substring (0, min (s.length, 4))
    end clip

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the counts for each interval in the histogram.
     */
    def computeHistogram (): VectorD =
        if counts == null then
            val h = new VectorD (numIntervals)
            for x <- value do h((floor ((x - minValue) / intervalWidth)).toInt) += 1
            h
        else
            counts
    end computeHistogram

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute coordinates (x, y, w, h) for each of bars used to draw the histogram.
     */
    def computeCoordinates (): MatrixD =
        val c      = new MatrixD (numIntervals, 4)
        val w      = (frameW - 2 * offset) / numIntervals.toDouble
        val scale  = (baseY - offset) / maxHistogram

        for i <- 0 until numIntervals do
            val h = histogram(i) * scale
            c(i) = VectorD (baseX + i * w, baseY.toDouble - h, w, h)
        c
    end computeCoordinates

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert a Histogram to a string.
     */
    override def toString: String = histogram.toString

end HCanvas


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `histogramTest` main function is used to test the `Histogram` class.
 *  As 'k' increases, the sum of Uniform approaches Normal.
 *  > runMain scalation.mathstat.histogramTest
 */
@main def histogramTest (): Unit =

    val intervals  = 100
    val samples    = 40000
    val k          = 2               // number of Uniform RV to add up

    val uniformRV   = Uniform (0, 1)
    val uniformDist = new VectorD (samples)
    for i <- 0 until samples do
        var sum = 0.0
        cfor (0, k) { _ => sum += uniformRV.gen }
        uniformDist(i) = sum
    end for
   
    val h1 = new Histogram (uniformDist, intervals, "Histogram for Sum of Uniform")
    println (s"histogram = $h1")

    val normalRV   = Normal (0, 1)
    val normalDist = new VectorD (samples)
    for i <- 0 until samples do normalDist(i) = normalRV.gen

    val h2 = new Histogram (normalDist, intervals, "Histogram for Normal")
    println (s"histogram = $h2")

//  val h3 = new Histogram (VectorD (0.0, 2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0),
//                          5, "Simple Histogram")
//  println (s"histogram = $h3")

end histogramTest

