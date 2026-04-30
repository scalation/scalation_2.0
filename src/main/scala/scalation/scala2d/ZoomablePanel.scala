
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller and Casey Bowman
 *  @version 2.0
 *  @date    Mon Jun 19 01:30:58 EDT 2023
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    General Trait to Make Zoomable Panels
 */

package scalation
package scala2d

import java.awt.event.{MouseEvent, MouseListener, MouseMotionListener, MouseWheelEvent, MouseWheelListener}
import java.awt.geom.{AffineTransform, Point2D}

//import scala.runtime.ScalaRunTime.stringOf

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ZoomablePanel` class extends `Panel` with the ability to zoom in and out.
 *  The mouse wheel controls the amount of zooming, while mouse dragging repositions 
 *  the objects in the panel (drawing canvas).
 *  Must add the following statement into your `paintComponent` method.
 *       g2d.setTransform (at)                                   // used for zooming @author Casey Bowman
 *  @see `docs.oracle.com/en/java/javase/20/docs/api/java.desktop/java/awt/geom/AffineTransform.html
 *  @see `animation` package for examples
 */
trait ZoomablePanel
      extends Panel
         with MouseWheelListener
         with MouseListener
         with MouseMotionListener:
   
    protected val at      = new AffineTransform ()               // Affine Transform
    private   var scale   = 1.0
    private   var basex   = 0.0
    private   var basey   = 0.0
//  private   var originx = 0.0
//  private   var originy = 0.0

    addMouseWheelListener (this)
    addMouseMotionListener (this)
    addMouseListener (this)

    at.scale (2, 2)                                              // scale up for initial size
    val mat = Array.ofDim [Double] (6)
    at.getMatrix (mat)                                           // get the first two rows of transform matrix
//  println (s"zoomablePanel: mat = ${stringOf (mat)}")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Override the mouseWheelMoved method to set the scale of the
     *  panel using the scroll value (up is negative, down is positive).
     *  @param e  the mouse wheel event
     */
    override def mouseWheelMoved (e: MouseWheelEvent): Unit =
        var x = e.getX ().toDouble
        var y = e.getY ().toDouble
        val p = new Point2D.Double ()
        try
            at.inverseTransform (new Point2D.Double (x, y), p)
        catch
            case _ : Exception => {}
        end try
        x = p.getX ()
        y = p.getY ()
        var zoom = 1.0
        val r = e.getWheelRotation ()
        if r < 0 then zoom *= 1.05
        if r > 0 then zoom /= 1.05
        at.translate (x, y)
        at.scale (zoom, zoom)
        scale *= zoom
        at.translate (-x, -y)
        revalidate ()
        repaint ()
    end mouseWheelMoved

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Override the mouseDeagged method to reposition the objects in the
     *  panel (drawing canvas).
     *  @param e  the mouse dragged event
     */
    override def mouseDragged (e: MouseEvent): Unit =
        val dx   = (e.getX () - basex) / scale
        val dy   = (e.getY () - basey) / scale
//      originx += dx * scale
//      originy += dy * scale
        at.translate (dx, dy)
        basex = e.getX ()
        basey = e.getY ()
        revalidate ()
        repaint ()
    end mouseDragged

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Override the mousePressed method to reset the base coordinates.
     *  @param e  the mouse pressed event
     */
    override def mousePressed (e: MouseEvent): Unit =
        basex = e.getX ()
        basey = e.getY ()
    end mousePressed

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** No action is to be taken upon the following mouse events.
     */
    override def mouseMoved (e: MouseEvent): Unit = {}
    override def mouseClicked (e: MouseEvent): Unit = {}
    override def mouseEntered (e: MouseEvent): Unit = {}
    override def mouseExited (e: MouseEvent): Unit = {}
    override def mouseReleased (e: MouseEvent): Unit = {}

end ZoomablePanel

