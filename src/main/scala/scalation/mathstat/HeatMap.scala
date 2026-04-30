
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Oct 27 17:08:17 EDT 2025
 *  @see     LICENSE (MIT style license file). 
 *
 *  @note    HeatMap to Display the Values in a Matrix with Color-Coding 
 *           e.g., Correlation Matrix, Lagged Cross Correlation Matrix
 */

package scalation
package mathstat

import scala.math.Pi

import scalation.scala2d.{BasicStroke, Graphics, Graphics2D, Line, Rectangle, VizFrame, ZoomablePanel}
import scalation.scala2d.Colors._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `HeatMap` class takes a matrix of values and displays them in color-coded
 *  cells where the color is based on the cell value (e.g., correlation).  The
 *  value for the cell 'heat(i, j)' is also displayed.
 *------------------------------------------------------------------------------
 *  Zoom functionality has two options:
 *  (1) mouse wheel controls the amount of zooming (in/out);
 *  (2) mouse dragging repositions the objects in the panel (drawing canvas).
 *  @see ZoomablePanel
 *------------------------------------------------------------------------------
 *  @param heat    the matrix of values (to be color-coded for display)
 *  @param name    the column names for the matrix (defaults to null)
 *  @param _title  the title of the heat-map (defaults to "HeatMap")
 */
class HeatMap (heat: MatrixD, name: Array [String] = null, _title: String = "HeatMap")
      extends VizFrame (_title, null):

    private val flaw = flawf ("HeapMap")                              // flaw function

    /** Create a drawing canvas
     */
    private val canvas = new HmCanvas (getW, getH, heat, name)

    if heat.dim2 != name.size then flaw ("init", s"requires heat.dim2 (${heat.dim2}) == name.size (${name.size}") 

    getContentPane ().add (canvas)
    setVisible (true)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert a HeatMap to a string.
     */
    override def toString: String = canvas.toString

end HeatMap


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `FramelessHeatMap` class should be used in embedded applications.
 *  @param frameW  the width
 *  @param frameH  the height
 *  @param heat    the matrix of values (to be color-coded for display)
 *  @param name    the column names for the matrix
 */
class FramelessHeatMap (frameW: Int, frameH: Int, heat: MatrixD, name: Array [String]):

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Dynamically create and return a drawing canvas.
     */
    def canvas: HmCanvas = new HmCanvas (frameW, frameH, heat, name)

end FramelessHeatMap


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Create a canvas on which to draw the heat-map.
 *  @param frameW  the frame width
 *  @param frameH  the frame height
 *  @param heat    the matrix of values (to be color-coded to display)
 *  @param name    the column names for the matrix
 */
class HmCanvas (frameW: Int, frameH: Int, heat: MatrixD, name: Array [String])
//    extends Panel:
      extends ZoomablePanel:

    private val offset  = 80                                          // offset heat-map with frame
    private val baseX   = offset
    private val baseY   = frameH - offset
    private val frameWO = frameW - 2 * offset                         // subtract left and right offsets
    private val frameHO = frameH - 2 * offset                         // subtract top and bottom offsets
    private val cell    = Rectangle ()                                // each matrix cell has color assigned
    private val axis    = Line (0, 0, 0, 0)                           // use lines for axes
    private val h_min   = heat.mmin                                   // minimum value in heat matrix
    private val h_max   = heat.mmax                                   // maximum value in heat matrix

    private val mc = 255 * 6                                          // max color value times 6
    private val c  = (VectorD.range (0 until 7) / 6.0).reverse        // fractional value cutoffs

    setBackground (white)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Paint the canvas by drawing the rectangles (cells) making up the heat-map.
     *  @param gr  low-resolution graphics environment
     */
    override def paintComponent (gr: Graphics): Unit =
        super.paintComponent (gr)
        val g2d = gr.asInstanceOf [Graphics2D]                        // use hi-res

        g2d.setTransform (at)                                         // used for zooming (at @see `ZoomablePanel`)

        val (nx, ny) = heat.dims
        val (cellW, cellH) = (frameWO / nx, frameHO / ny)
        var x_pos = 0
        var y_pos = 0

        //:: Draw the axes

        g2d.setPaint (black)
        g2d.setStroke (new BasicStroke (2.0f))
        axis.setLine (baseX - 1, baseY + 1, baseX + 10 + frameWO, baseY + 1)
        g2d.draw (axis)
        axis.setLine (baseX - 1, offset - 10, baseX - 1, baseY + 1)
        g2d.draw (axis)

        //:: Draw the labels on the axes

        y_pos = baseY + 15
        for i <- heat.indices do
            val x_val = i.toString
            x_pos = offset - 8 + i * (frameWO) / heat.dim 
            g2d.drawString (x_val, x_pos, y_pos)
            if name != null then
                val (nam, back) = clip (name(i), 20)
                drawRotatedString (g2d, nam, x_pos + cellW - 10 + back/2, y_pos + 10 + back/2)
//              g2d.drawString (nam, x_pos + cellW / 2 - back, y_pos)
        end for

        x_pos = baseX - 30
        for j <- heat.indices2 do
            val y_val = j.toString
            y_pos = offset + 20 + j * (frameHO) / heat.dim2
            g2d.drawString (y_val, x_pos, y_pos)
        end for

        //:: Draw the cells making up the heat-map

        x_pos = baseX - 30
        y_pos = baseY + 15
        val shiftW = cellW / 2 - 8
        val shiftH = cellH / 2
        for i <- heat.indices do
            x_pos = offset - 2 + i * (frameWO) / heat.dim 
            for j <- heat.indices2 do
                val h_ij = heat(i, j)
                y_pos = offset + 2 + j * (frameHO) / heat.dim2
                cell.setFrame (x_pos, y_pos, cellW, cellH)            // x, y, w, h
                val (col1, col2) = computeColor (h_ij)
                g2d.setPaint (col1)
                g2d.fill (cell)
                g2d.setPaint (col2)
                val (num, _) = clip (h_ij.toString, 4)
                g2d.drawString (num, x_pos + shiftW, y_pos + shiftH)
            end for
        end for
    end paintComponent

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Draw a text string rotated by the given angle.
     *  @param g2d    hi-res graphics
     *  @param text   the text string to be displayed
     *  @param cx     center for x-coordinate
     *  @param cy     center for y-coordinate
     *  @param angle  the rotation angle in radians (defaults to Pi/4 radians or 45 degrees)
     */
    def drawRotatedString (g2d: Graphics2D, text: String,
                           cx: Int, cy: Int, angle: Double = Pi/4): Unit =
        val oldTx = g2d.getTransform                                  // save current transform
        val fm    = g2d.getFontMetrics
        val textW = fm.stringWidth (text)

        // Rotate around center (cx, cy), then draw text centered on cx
        g2d.rotate (angle, cx.toDouble, cy.toDouble)
        g2d.drawString (text, (cx - textW / 2), cy)

        g2d.setTransform (oldTx)                                      // restore
    end drawRotatedString

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the color for cell (i, j) in the heat-map based on h_ij, returning
     *  the cell color and its complementary color (for writing a value).
     *  @param h_ij  the value of cell (i, j) in the heat matrix
     */
    def computeColor (h_ij: Double): (Color, Color) =
        val frac = (h_ij - h_min) / (h_max - h_min).toDouble

        // high to low: M, B, C, G, Y, R, W

        val rgb = if frac > c(1) then
            val z = (mc * (frac - c(1))).toInt; (z, 0, z)             // magenta
        else if frac > c(2) then
            (0, 0, (mc * (frac - c(2))).toInt)                        // blue
        else if frac > c(3) then
            val z = (mc * (frac - c(3))).toInt; (0, z, z)             // cyan
        else if frac > c(4) then
            (0, (mc * (frac - c(4))).toInt, 0)                        // green
        else if frac > c(5) then
            val z = (mc * (frac - c(5))).toInt; (z, z, 0)             // yellow
        else
            ((mc * (frac - c(6))).toInt, 0, 0)                        // red

        val comple = 255 - (rgb._1 + rgb._2 + rgb._3) / 2             // complementary color
        println (s"h_ij = $h_ij, frac = $frac, rgb = $rgb")
        (new Color (rgb._1, rgb._2, rgb._3),
         new Color (comple, comple, comple))
    end computeColor

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Clip/cut string s to mx characters and return half its pixel length.
     *  @param s   the value to clip/cut
     *  @param mx  the maximum number of characters
     */
    def clip (s: String, mx: Int = 12): (String, Int) =
        val len = math.min (s.length, mx)
        (s.substring (0, len), 4 * (len - 1)) 
    end clip

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert a heat matrix to a string.
     */
    override def toString: String = heat.toString

end HmCanvas


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `heatMapTest` main function is used to test the `HeatMap` class.
 *  It tests creating a heat-map for a matrix 
 *  > runMain scalation.mathstat.heatMapTest
 */
@main def heatMapTest (): Unit =

    import MatrixDOps.⊗

    val x    = VectorD (1, 2, 3, 4, 5, 6)
    val heat = x ⊗ x
    val name = Array ("v0", "v1", "v2", "v3", "v4", "v5")

    val hm = new HeatMap (heat, name, "HeatMap for matrix heat")
    println (s"heatMap = $hm")

end heatMapTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `heatMapTest2` main function is used to test the `HeatMap` class.
 *  It tests creating a heat-map for a CORRELATION matrix using a COVID-19 dataset.
 *  > runMain scalation.mathstat.heatMapTest2
 */
@main def heatMapTest2 (): Unit =

    val fileName = "covid_19_weekly.csv"

    val (xy, name) = MatrixD.loadH (fileName, 1)                      // trim first column
    val heat = xy.corr                                                // correlation x_i vs. x_j

    val hm = new HeatMap (heat, name, "HeatMap for correlation matrix")
    println (s"heatMap = $hm")

end heatMapTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `heatMapTest3` main function is used to test the `HeatMap` class.
 *  It tests creating a heat-map for a LAGGED CROSS CORRELATION matrix for name(0)
 *  using a COVID-19 dataset.
 *  > runMain scalation.mathstat.heatMapTest3
 */
@main def heatMapTest3 (): Unit =

    val fileName = "covid_19_weekly.csv"

    val (xy, name) = MatrixD.loadH (fileName, 1)                      // trim first column
    val (x, y) = (xy.not(?, 1), xy(?, 1))                             // "new_deaths" is column 1
    swap (name, 0, 1)                                                 // swap to make name(0) = "new_deaths"

    val lags = 20                                                     // check 20 lags 0, ..., 19
    val yx   = y +^: x                                                // prepend y "new_deaths"
    val heat = new MatrixD (lags, yx.dim2)
    for j <- yx.indices2; l <- 0 until lags do
        heat(l, j) = y.ccorr (yx(?, j), l)                            // cross correlation y vs. x_j at lag l
    
    val hm = new HeatMap (heat.transpose, name,
                          s"HeatMap for lagged cross correlation matrix for ${name(0)}")
    println (s"heatMap = $hm")

end heatMapTest3

