
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Dec  8 22:56:57 EST 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Monte Carlo Estimation of the Volume of a Unit Sphere
 */

package scalation
package simulation
package monte_carlo

import scalation.random.Random

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `sphereVolumeTest` main function provides a simple example of Monte Carlo
 *  estimation of the volume of a unit sphere.
 *  > runMain scalation.simulation.monte_carlo.sphereVolumeTest
 */
@main def sphereVolumeTest (): Unit =

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether the point (x, y, z) is inside the unit sphere.
     *  @param x  the x-coordinate
     *  @param y  the y-coordinate
     *  @param z  the z-coordinate
     */
    def inSphere (x: Double, y: Double, z: Double): Boolean =
        x~^2 + y~^2 + z~^2 <= 1.0
    end inSphere

    val n     = 100000000
    val r     = Random ()
    var count = 0

    cfor (0, n) { _ => if inSphere (r.gen, r.gen, r.gen) then count += 1 }
    println (s"Sphere Volume = ${(8.0 * count) / n.toDouble}")

end sphereVolumeTest

