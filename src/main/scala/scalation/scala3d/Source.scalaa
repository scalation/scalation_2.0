
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Jacobi Coleman
 *  @version 2.0
 *  @date    Wed May  1 01:19:46 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Class for Generating Vehicles
 */

package scalation
package scala3d

import scalafx.geometry.Point3D
import scalafx.scene.paint.{Color, PhongMaterial}
import scalafx.scene.shape.Box

import scalation.mathstat.Statistic

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Source` case class is used to create vehicles in the animation.
 *  @param size      the size of the source
 *  @param name      the name of the source
 *  @param location  the location of the source in the scene
 *  @param color     the color of the source
 */
case class Source (size: Int, name: String, location: Point3D = Point3D (0.0,0.0,0.0),
                   color: Color = Color.Burlywood):

    val entry        = new Box (size, size, size)
    entry.translateX = location.x;
    entry.translateY = location.y;
    entry.translateZ = location.z;
    val surface      = new PhongMaterial (color)
    entry.material   = surface
    val iArrivalTime = new Statistic ("Inter-Arrival Time")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this Source to a string.
     */
    override def toString: String = s"Source(color = $color, size = $size, id = $id)"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show the statistics collected by this Source.
     */
    def showStats (): Unit =
        println (Statistic.labels)
        println (iArrivalTime)
        println (iArrivalTime.show)
        println (s"ms = ${iArrivalTime.ms}, rms = ${iArrivalTime.rms}")
    end showStats

end Source

