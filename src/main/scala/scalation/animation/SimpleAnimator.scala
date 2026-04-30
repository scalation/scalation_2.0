
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Oct 24 15:42:07 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Animate the Motion of Objects - awt/swing version
 */

package scalation
package animation

import java.awt.{Dimension, Color, Graphics, Graphics2D}
import java.awt.geom.{Ellipse2D, Point2D}
import javax.swing.{JFrame, JPanel}
import javax.swing.WindowConstants.EXIT_ON_CLOSE

import scala.math.{cos, sin}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SimpleAnimator` class demonstrates how the Java 2D API shapes can be
 *  put into motion to create a simple animation.
 *  The code is for demonstration purposes: other code should use the `scala2d`
 *  package to insulate from the particular graphics framework.
 *  @see `SimpleAnimator2` to see the changes
 *  @param title  the title of the display
 */
class SimpleAnimator (title: String)
      extends JFrame (title) with Runnable:

    private val dim     = new Dimension (600, 500)                    // the size of the canvas
    private val tau     = 20                                          // operate at 50 Hz
    private val circle  = new Ellipse2D.Double (200, 200, 200, 200)   // the circle to traverse
    private val ballPos = new Point2D.Double (0, 300)                 // ball position
    private val ball    = new Ellipse2D.Double ()                     // the moving ball

    getContentPane ().add (new Canvas ())
    setLocation (100, 100)
    setSize (dim)
    setVisible (true)
    setDefaultCloseOperation (EXIT_ON_CLOSE)
    new Thread (this).start ()

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `Canvas` inner class is used to place shapes in the drawing region.
     */
    class Canvas extends JPanel:

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Paint the canvas panel component.
         *  @param gr  the graphics context
         */
        override def paintComponent (gr: Graphics): Unit =
            super.paintComponent (gr)
            val gr2 = gr.asInstanceOf [Graphics2D]                    // use hi-res

            gr2.setPaint (Color.blue)                                 // blue circle
            gr2.draw (circle)

            gr2.setPaint (Color.red)                                  // read ball
            ball.setFrame (ballPos.x - 10, ballPos.y - 10, 20, 20)
            gr2.fill (ball)
        end paintComponent

    end Canvas

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Run method for the display thread that repeatedly updates coordinates,
     *  sleeps and repaints.
     */
    def run (): Unit =
        var theta = 0.0

        while true do
            theta    += 0.05
            ballPos.x = 300 + 100 * cos (theta)
            ballPos.y = 300 + 100 * sin (theta)
            println (s"ballPos = $ballPos")

            try
                Thread.sleep (tau)
            catch 
                case _ : InterruptedException => println ("SimpleAnimator.run: sleep failed")
            end try

            repaint ()
        end while
    end run

end SimpleAnimator


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleAnimatorTest` is a main function for invoking the `SimpleAnimator`.
 *  Abstract Window Toolkit.
 *  > runMain scalation.animation.simpleAnimatorTest
 */
@main def simpleAnimatorTest (): Unit =

     new SimpleAnimator ("SimpleAnimator")

end simpleAnimatorTest

