
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Casey Bowman
 *  @version 2.0
 *  @date    Tue Feb  4 14:56:34 EST 2020
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Vehicle Is Enhanced SimActor Supporting Changing Velocities
 */

package scalation
package simulation
package process

import scala.collection.immutable.Map

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Vehicle` object contains driver/vehicle characteristics/properties.
 *  @see https://hypertextbook.com/facts/2001/MeredithBarricella.shtml
 */
object Vehicle:

    /** defaults values for driver/vehicle characteristics/properties (PUBLIC access required)
     *  units are meters and seconds
     *  @see https://en.wikipedia.org/wiki/Intelligent_driver_model
     */
    val def_prop = Map ("rt"   -> 1.0,                       // driver reaction time
                        "amax" -> 3.0,                       // max acceleration
                        "bmax" -> -3.5,                      // max deceleration (typically higher than acceleration)
//                      "v0"   -> 0.0,                       // starting velocity, from a stopped position
                        "v0"   -> 33.528,                    // starting velocity, sim segment of interstate
                        "vmax" -> 33.528,                    // max velocity
                        "T"    -> 3.0,                       // safe min time headway
                        "s"    -> 5.0,                       // safe min distance headway
                        "len"  -> 4.0,                       // length of the vehicles
                        "del"  -> 4.0)                       // acceleration exponent (delta)

    /** Current values for driver/vehicle characteristics/properties
     *  To change a property: Vehicle.prop("amax") = 4.0
     */
    private [process] var prop = def_prop

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the current value for 'prop("rt") the driver reaction time
     */
    inline def rt: Double   = prop("rt")                     // driver reaction time
    inline def amax: Double = prop("amax")                   // max acceleration
    inline def bmax: Double = prop("bmax")                   // max deceleration
    inline def v0: Double   = prop("v0")                     // starting velocity
    inline def vmax: Double = prop("vmax")                   // max velocity
    inline def T: Double    = prop("T")                      // min time headway
    inline def s: Double    = prop("s")                      // min distance headway
    inline def len: Double  = prop("len")                    // length of the vehicles
    inline def del: Double  = prop("del")                    // acceleration exponent (delta)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set all the driver/vehicle characteristics/properties to the new property values.
     *  @param new_prop  the new property values
     */
    def setProps (new_prop: Map [String, Double]): Unit = prop = new_prop

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Butcher's Method (fifth order) for numerically solving an ordinary differential equation.
     *  @param Ft     the "original" function value at time t
     *  @param ft     the "derivative" function value at time t
     *  @param ft_rt  the "derivative" function value at time t - rt
     *  @param rt     the time difference (reaction time)
     *
     *  FIX - integrate into Dynamics package
     */
    def butcher (Ft: Double, ft: Double, ft_rt: Double, rt: Double): Double =
        val _1_by_9 = 1.0 / 9.0
        val k1 = ft_rt
        val k3 = ft_rt + 0.25 * (ft - ft_rt)
        val k4 = ft_rt + 0.50 * (ft - ft_rt)
        val k5 = ft_rt + 0.75 * (ft - ft_rt)
        val k6 = ft
        Ft + _1_by_9 * (7 * k1 + 32 * k3 + 12 * k4 + 32 * k5 + 7 * k6) * rt
    end butcher

end Vehicle


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Vehicle` class extends the `SimActor` and represents a vehicle in motion.
 *  @param name_     the name of the vehicle
 *  @param director  the model to which this vehicle belongs
 */
abstract class Vehicle (name_ : String, director: Model)
         extends SimActor (name_, director)
            with Dynamics:

    // Each car needs to located itself in terms of what `Pathway` it is on and what node in that `Pathway`
//  protected var myPathway: Pathway = null
    private [process] var myPathNode: DoublyLinkedList [Vehicle]#Node = null

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The abstract method, 'act', is defined in each subclass to provide specific
     *  behavior.
     */
    override def act (): Unit = println ("Vehicle.act method should be overridden")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show the `SimActor`s full name and activation time.
     */
    override def toString: String = s"Vehicle ($me at $actTime with cor_id $cor_id and location $disp)"

end Vehicle

