
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Oct 24 15:42:07 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Animate the Motion of Objects - scala2d version
 */

package scalation
package animation

//import java.awt.{Dimension, Color, Graphics, Graphics2D}
//import java.awt.geom.{Ellipse2D, Point2D}
//import javax.swing.{JFrame, JPanel}
//import javax.swing.WindowConstants.EXIT_ON_CLOSE

import scalation.mathstat.VectorD
import scalation.scala2d.{Ellipse, Graphics, Graphics2D, VizFrame, ZoomablePanel}
import scalation.scala2d.Colors.{blue, red}

import scala.math.{cos, sin}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SimpleAnimator2` class demonstrates how the Java 2D API shapes can be
 *  put into motion to create a simple animation.
 *  This code uses the `scala2d` package to insulate from the particular graphics framework.
 *  @see `SimpleAnimator` to see the changes need to use Java awt/swing directly
 *  @param _title  the title of the display
 */
class SimpleAnimator2 (_title: String)
      extends VizFrame (_title, null, 1200, 800) with Runnable:

    private val tau     = 20                                          // operate at 50 Hz
    private val circle  = Ellipse ()                                  // the circle to traverse
    circle.setFrame (200, 200, 200, 200)                              // location and size of circle
    private val ballPos = VectorD (0, 300)                            // ball position
    private val ball    = Ellipse ()                                  // the moving ball

    getContentPane ().add (new Canvas ())
    setLocation (100, 100)
    setVisible (true)
    new Thread (this).start ()

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `Canvas` inner class is used to place shapes in the drawing region.
     */
    class Canvas
//        extends Panel:                                              // regular panel
          extends ZoomablePanel:                                      // adds zoom in and out

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Paint the canvas panel component.
         *  @param gr  the graphics context
         */
        override def paintComponent (gr: Graphics): Unit =
            super.paintComponent (gr)
            val g2d = gr.asInstanceOf [Graphics2D]                    // use hi-res

            g2d.setTransform (at)                                     // used for zooming @author Casey Bowman

            g2d.setPaint (blue)                                       // blue circle
            g2d.draw (circle)

            g2d.setPaint (red)                                        // read ball
            ball.setFrame (ballPos(0) - 10, ballPos(1) - 10, 20, 20)
            g2d.fill (ball)
        end paintComponent

    end Canvas

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Run method for the display thread that repeatedly updates coordinates,
     *  sleeps and repaints.
     */
    def run (): Unit =
        var theta = 0.0

        while true do
            theta     += 0.05
            ballPos(0) = 300 + 100 * cos (theta)
            ballPos(1) = 300 + 100 * sin (theta)
            println (s"ballPos = $ballPos")

            try
                Thread.sleep (tau)
            catch 
                case _ : InterruptedException => println ("SimpleAnimator2.run: sleep failed")
            end try

            repaint ()
        end while
    end run

end SimpleAnimator2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleAnimator2Test` is a main function for invoking the `SimpleAnimator2`.
 *  > runMain scalation.animation.simpleAnimator2Test
 */
@main def simpleAnimator2Test (): Unit =

     new SimpleAnimator2 ("SimpleAnimator2")

end simpleAnimator2Test

