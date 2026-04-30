
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Jan  2 15:20:24 EST 2019
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model/Optimization Support: Hyper-Parameters
 */

package scalation

import scala.collection.mutable.LinkedHashMap                                // maintains specification order
import scala.runtime.ScalaRunTime.stringOf

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `HyperParameter` class provides a simple and flexible means for handling
 *  model/optimization hyper-parameters.  A model/optimizer may have one or more
 *  hyper-parameters that are organized into a map name -> (value, defaultV).
 *  Allows hyper-parameters to be changed without constant re-compilation or
 *  resorting to long arguments lists
 *  Usage: hp("eta") = 0.01
 */
class HyperParameter extends Cloneable:

    private val hparam = LinkedHashMap [String, (ValueType, ValueType)] ()   // hyper-parameter map

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given the name, return the hyper-parameter value.
     *  @param name  the name of the hyper-parameter
     */
    def apply (name: String): ValueType = hparam (name)._1

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given the name, return the hyper-parameter default value.
     *  @param name  the name of the hyper-parameter
     */
    def default (name: String): ValueType = hparam (name)._2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given the name, update the hyper-parameter value.  Note (v, d) for value, default
     *  @param name   the name of the hyper-parameter
     *  @param value  the value of the hyper-parameter
     */
    def update (name: String, value: ValueType): Unit =
        val (_, d) = hparam (name)
        hparam += name -> (value, d)
    end update

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create and return new hyper-parameters, updating the one with the given name.
     *  @param name   the name of the hyper-parameter
     *  @param value  the value of the hyper-parameter
     */
    def updateReturn (name: String, value: ValueType): HyperParameter =
        val hp2 = clone ().asInstanceOf [HyperParameter]
        val (_, d) = hp2.hparam (name)
        hp2.hparam += name -> (value, d)
        hp2
    end updateReturn

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create and return new hyper-parameters, updating the ones with the given nvs.
     *  @param nvs  the name-value pairs for the hyper-parameter
     */
    def updateReturn (nvs: (String, ValueType)*): HyperParameter =
        val hp2 = clone ().asInstanceOf [HyperParameter]
        for nv <- nvs do
            val (_, d) = hp2.hparam (nv._1)
            hp2.hparam += nv._1 -> (nv._2, d)
        end for
        hp2
    end updateReturn

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Concatenate this hyper-parameter map with a second hyper-parameter map.
     *  When the names are the same, the value for hp2 is taken.
     *  @param hp2  the second hyper-parameter map
     */
    def ++ (hp2: HyperParameter): HyperParameter =
        val hp3 = clone ().asInstanceOf [HyperParameter]
        for (n, v) <- hp2.hparam do hp3.hparam += n -> v           // (n, v) name, value pair
        hp3
    end ++
 
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add a new hyper-parameter to the map.
     *  @param name      the name of the hyper-parameter
     *  @param value     the value of the hyper-parameter
     *  @param defaultV  the defualt value of the hyper-parameter
     */
    def += (name: String, value: ValueType, defaultV: ValueType): Unit =
        hparam += name -> (value, defaultV)
    end +=
 
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Remove the hyper-parameter with the given name from the map.
     *  @param name  the name of the hyper-parameter
     */
    def -= (name: String): Unit =
        hparam -= name
    end -=

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Collect all values into an integer array.
     */
    def toInt: Array [Int] = (for v <- hparam.values yield v._1.toInt).toArray
 
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Collect all values into an integer array.
     */
    def toDouble: Array [Double] = (for v <- hparam.values yield v._1.toDouble).toArray
 
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert the hyper-parameter map to a string.
     */
    override def toString: String = hparam.toString.replace ("LinkedHashMap", "HyperParameter")
 
end HyperParameter


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `hyperParameterTest` main function is used to test the `HyperParameter` class.
 *  > runMain scalation.hyperParameterTest
 */
@main def hyperParameterTest (): Unit =

    val hp = new HyperParameter
    hp += ("eta", 0.1, 0.1)
    hp += ("bSize", 10, 10)
    hp += ("maxEpochs", 1000, 1000)

    println (s"hp = $hp")

//  hp("eta") = 0.2

    println (s"hp = $hp")
    println (s"""hp("eta") = ${hp("eta")}""")
    println (s"""hp.default ("eta") = ${hp.default ("eta")}""")

    val hp2 = new HyperParameter
    hp2 += ("cThresh", 0.5, 0.5)
    hp2 += ("eta", 0.2, 0.2)

    println (s"hp ++ hp2 = ${hp ++ hp2}")

end hyperParameterTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `hyperParameterTest2` main function is used to test the `HyperParameter` class.
 *  > runMain scalation.hyperParameterTest2
 */
@main def hyperParameterTest2 (): Unit =

    val hp = new HyperParameter
    hp += ("p", 3, 3)
    hp += ("d", 1, 1)
    hp += ("q", 2, 2)

    banner ("Values for hyper-parameters")
    println (s"hp          = $hp")
    println (s"hp.toInt    = ${stringOf (hp.toInt)}")
    println (s"hp.toDouble = ${stringOf (hp.toDouble)}")

end hyperParameterTest2

