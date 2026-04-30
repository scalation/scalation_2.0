
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Jacobi Coleman
 *  @version 2.0
 *  @date    Wed May  1 01:19:46 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Class for Maintaining the Animation Time
 */

package scalation
package scala3d

import scalafx.geometry.Point3D
import scalafx.scene.paint.{Color, PhongMaterial}
import scalafx.scene.shape.Box

import scalation.mathstat.Statistic

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Sink` case class is used to remove/terminate vehicles in the animation.
 *  @param size      the size of the sink
 *  @param name      the name of the sink
 *  @param location  the location of the sink in the scene
 *  @param color     the color of the sink
 */
case class Sink (size: Int, name: String, location: Point3D = Point3D (0.0,0.0,0.0),
                 color: Color = Color.Blue):

    val exit        = new Box (size, size, size)
    exit.translateX = location.x;
    exit.translateY = location.y;
    exit.translateZ = location.z;
    val surface     = new PhongMaterial (color)
    exit.material   = surface
    val travelTime  = new Statistic ("Travel Time")
    val trafficFlow = new Statistic ("Traffic Flow")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Terminate the given vehicle and collect travel-time statistics.
     *  @param vehicle  the vehicle to terminate
     *  @param clock    the animation clock
     */
    def terminate (vehicle: Vehicle, clock: Clock): Unit =
        val timediff = clock.now - vehicle.startTime
        travelTime.tally (timediff)
        println (s"vehicle $vehicle terminated at time ${clock.now} with travel-time = $timediff")
    end terminate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this Sink to a string.
     */
    override def toString: String = s"Sink(color = $color, size = $size, id = $id)"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show the statistics collected by this Sink.
     */
    def showStats (): Unit =
        println (Statistic.labels)
        println (travelTime)
        println (travelTime.show)
        println (s"ms = ${travelTime.ms}, rms = ${travelTime.rms}")
    end showStats

end Sink

