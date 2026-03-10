

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Sep  7 15:05:06 EDT 2009
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Base Trait for Simulation Components
 */

package scalation
package simulation
package process

import scala.collection.mutable.{ArrayBuffer => VEC}
//import scala.collection.mutable.{ListBuffer => VEC}

import scalation.mathstat.{Statistic, TimeStatistic}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Component` trait provides basic common feature for simulation components.
 *  The list of subparts is empty for atomic components and nonempty for
 *  composite components.
 *  Identifiable has "name" the name of this component
 *  Locatable has "at" the location of this component
 */
trait Component
      extends Identifiable with Locatable:

    private val flaw = flawf ("Component")                      // flaw function

    private [process] val RAD     = 5.0                         // radius of a token (for animating entities)
    private [process] val DIAM    = 2.0 * RAD                   // diameter of a token (for animating entities)
    private [process] val subpart = VEC [Component] ()          // list of subparts of the Component
                                                                // (empty for atomic components, nonempty for composites)

    private var _durationStat: Statistic = null                 // collector of sample statistics (e.g., waiting time)
    private var _persistentStat: TimeStatistic = null           // collector of time persistent statistics (e.g., number in queue)
    private var _director: Model = null                         // director of the play/simulation model (to which this component belongs)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Initialize this component (all of its 'var's).
     *  @param label  the name of this component
     *  @param loc    the location of this component
     */
    def initComponent (label: String, loc: Array [Double]): Unit =
        name = label
        at   = loc
        initStats (label)
        //if at == null then flaw ("init", s"component '$name' has null location")
    end initComponent

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the director who controls the play/simulation this component is in.
     */
    def director: Model = _director

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set this component's director (the controller of the simulation model).
     *  @param director  the director of the play/simulation
     */
    def director_= (director: Model): Unit =
        if _director == null && director != null then _director = director
        else flaw ("setDirector", "director may only be set once")
    end director_=

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Indicate whether this component is composite, i.e., has subparts.
     */
    def composite: Boolean = subpart.size > 0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Initialize this component's statistical collectors.
     *  Sample statistics:  all `Component`s.
     *  Time-persistent statistics:  all except `Gate`, `Source` and `Sink`.
     *  @param label  the name of this component
     */
    protected def initStats (label: String): Unit =
        _durationStat   = new Statistic (label)
        if ! this.isInstanceOf [Source] && ! this.isInstanceOf [Sink] && ! this.isInstanceOf [Gate] then
            _persistentStat = new TimeStatistic ("p-" + label)
    end initStats

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Aggregate the statistics of this component's subparts.
     */
    def aggregate (): Unit =
        val n = subpart.size
        if n > 0 then
            val durationStatList   = VEC [Statistic] ()
            val persistentStatList = VEC [TimeStatistic] ()
            for p <- subpart do
                durationStatList += p.durationStat
                if director.full && p.persistentStat != null then persistentStatList += p.persistentStat
            end for
            _durationStat = Statistic.aggregate (durationStatList, name)
            if director.full then _persistentStat = TimeStatistic.aggregate (persistentStatList, "p-" + name)
        end if
    end aggregate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Abstract display method.
     */
    def display (): Unit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Tally the duration (e.g., waiting time) of an activity or delay.
     *  @param duration  the time duration
     */
    def tally (duration: Double): Unit = _durationStat.tally (duration)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Accumulate the value (e.g., number in  queue) weighted by its time duration.
     *  @param value  the value to accumulate
     */
    def accum (value: Double): Unit = _persistentStat.accum (value, _director.clock)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return sample statistics for durations for this component (e.g., Time in queue).
     */
    def durationStat: Statistic = _durationStat

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return time persistent statistics for value for this component (e.g. Number in queue).
     */
    def persistentStat: TimeStatistic = _persistentStat

end Component

