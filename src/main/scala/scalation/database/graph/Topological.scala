
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Oct 14 22:41:46 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Topological Objects/Tokens Located in a Graph
 */

package scalation
package database
package graph

//import scala.collection.immutable.{Vector => VEC}
import scala.collection.mutable.{ArrayBuffer => VEC}
import scala.math.abs

type Element = Vertex | Edge

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
extension (el: Element)
    def tokens: VEC [Topological] = 
        el match
        case el: Vertex => tokens
        case el: Edge   => tokens
        end match
    end tokens


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Topological` trait provides topological coordinates that are topologically
 *  partially ordered.
 *  @param elem  the element in the graph (at a vertex or on an edge)
 *  @param dist  its distance along the segment
 */
trait Topological (var elem: Element, var dist: Double)
      extends PartiallyOrdered [Topological]:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compare two spatial objects based on their space coordinates.
     *  @param other  the other item to compare with this item
     */
    infix def tryCompareTo [B >: Topological: AsPartiallyOrdered] (other: B): Option [Int] =
        val oth = other.asInstanceOf [Topological]
        if elem == oth.elem then Option (dist compare oth.dist)
        else oth.elem `tryCompareTo` oth.elem
    end tryCompareTo

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the topological objects/tokens in the neighborhood of this token.
     */
    def neighbors: VEC [Topological] = elem.tokens

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the topological objects/tokens in the neighborhood of this token
     *  within distance d.
     *  @param d  the maximum allowed distance to be considered in the neighborhood
     */
    def neighbors (d: Double): VEC [Topological] =
        for t <- elem.tokens if abs (t.dist - dist) < d yield t
    end neighbors

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert the spatial object to a string.
     */
    override def toString: String = s"Topological ($elem, $dist)"

end Topological

