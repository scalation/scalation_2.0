
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Jacobi Coleman
 *  @version 2.0
 *  @date    Wed May  1 01:19:46 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Class for Representing Moveable Objects
 */

package scalation
package scala3d

import scalafx.geometry.Point3D
import scalafx.scene.paint.{Color, PhongMaterial}
import scalafx.scene.shape.Sphere

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Vehicle` companion object maintains a counter for making unique ids.
 */
object Vehicle:

    var id = 0
    def idNext (): Int = { id += 1; id }

end Vehicle


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Vehicle` case class is used to for representing moveable objects.
 *  @param clock     the animation clock
 *  @param color     the color of the vehicle
 *  @param size      the size of the vehicle
 *  @param location  the location of the vehicle in the scene
 *  @param id        the vehicle's unique identifier
 *  @param speed     the speed of the vehicle
 */
case class Vehicle (clock: Clock, color: Color = Color.Blue, size: Int = 5,
                    location: Point3D = Point3D (0, 0, 0),
                    id: Int = Vehicle.idNext (), speed: Double = 0):

    val body      = new Sphere (size)
    val surface   = new PhongMaterial (color)
    body.material = surface
    val startTime = clock.now

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this vehicle to a string.
     */
    override def toString: String = s"Vehicle(color = $color, size = $size, id = ${Vehicle.id})"

end Vehicle

