
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Oct  2 18:36:01 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Vertex in a Property Graph
 */

package scalation
package database
package graph

//import scala.collection.immutable.{Vector => VEC}
import scala.collection.mutable.{ArrayBuffer => VEC}
import scala.collection.mutable.Map
import scala.math.max

import scalation.mathstat.VectorD

import Vertex.prt

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Vertex` class maintains properties for a vertex, e.g., a person.
 *  A vertex is analogous to a tuple in an RDBMS.
 *  @param _name  the name of this vertex ('name' from `Identifiable`), serves as vertex label
 *  @param prop   maps vertex's property names into property values
 *  @param _pos   the position (Euclidean coordinates) of this vertex ('pos' from `Spatial)
 */
class Vertex (_name: String, val prop: Property, _pos: VectorD = null)
      extends Identifiable (_name)
         with Spatial (_pos)
         with PartiallyOrdered [Vertex]
         with Serializable:

    val tokens = Set [Topological] ()                     // topological objects/tokens at this vertex                     

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compare two spatial objects based on their space coordinates.
     *  @param other  the other item to compare with this item
     */
    override def tryCompareTo [B >: Vertex : AsPartiallyOrdered] (other: B): Option [Int] =
        val oth = other.asInstanceOf [Vertex]
        pos tryCompareTo oth.pos
    end tryCompareTo

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the center coordinates for this vertex.
     */
    def getCenter: VectorD = VectorD (_pos(0) + _pos(2)/2, _pos(1) + _pos(3)/2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this vertex object to a string.
     */
    override def toString: String =
        val propStr = if prop == null then "()" else prop.mkString (", ")
        s"Vertex($id, $name, $propStr, $pos)"
    end toString

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SHOW/print this table, one tuple per row.
     *  Formatting: regular column is 'width_' chars wide, 'X' is double that
     *  FIX - currently only works for width_, not width
     &  @param width_  the standard column width
     */
    def show (width_ : Int = 18): Unit =
        val wj  = width_
        val len = width_ * (prop.size)

        if len != 0 then
            println (s"\n>> Table $name with 1 row")
            println ("|-" + "-" * len + "-|")

            print ("| ")
            for (k, _) <- prop do prt (k, wj)
            println (" |")

            println ("|-" + "-" * len + "-|")

            print("| ")
            for (_, v) <- prop do prt (v, wj)
            println (" |")

            println ("|-" + "-" * len + "-|")
        end if
    end show

end Vertex


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Vertex` companion object provides a factory function for creating vertices.
 */
object Vertex:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an object of type `Vertex`.
     *  @param prop  maps vertex's property names into property values
     *  @param pos   the position (Euclidean coordinates) of this vertex
     */
    def apply (prop: Property, pos: VectorD = null): Vertex = 
        new Vertex (null, prop, pos)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show the vertices.
     *  @param vertices  the vertices to show
     */
    def show (vertices: VEC [Vertex]): Unit =
        for v <- vertices do println (v)
    end show

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** PRINT value-type v with a width of w.
     *  @param v the value to be printed
     *  @param w the width (# chars) for the column
     */
    def prt (v: ValueType, w: Int): Unit =
        val str = v.toString
        val w0  = str.size
        val rem = w - w0
        val lft = max (rem / 2, 0)
        val rht = max (rem - lft, 0)
        print (" " * lft + str + " " * rht)
    end prt

end Vertex


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `vertexTest` main function is used to test the `Vertex` class.
 *  > runMain scalation.database.graph.vertexTest
 */
@main def vertexTest (): Unit =

    val x0 = VectorD (200, 100, 40, 40)                           // vertex coordinates, size: x, y, w, h
    val x1 = VectorD (100, 400, 40, 40)
    val x2 = VectorD (500, 500, 40, 40)

    // use the class constructor to explicitly assign vertex label

    val vertices = VEC (
        new Vertex ("Bob", Map ("name" -> "Bob", "state" -> "GA", "salary" -> 85000.0), x0),
        new Vertex ("Sue", Map ("name" -> "Sue", "state" -> "FL", "salary" -> 95000.0), x1),
        new Vertex ("Joe", Map ("name" -> "Joe", "state" -> "GA", "salary" -> 99000.0), x2))

    banner ("Vertices with explicitly assigned vertex labels")
    Vertex.show (vertices)

    // use the companion object's apply method to use system generated vertex label

    val vertices2 = VEC (
        Vertex (Map ("name" -> "Bob", "state" -> "GA", "salary" -> 85000.0), x0),
        Vertex (Map ("name" -> "Sue", "state" -> "FL", "salary" -> 95000.0), x1),
        Vertex (Map ("name" -> "Joe", "state" -> "GA", "salary" -> 99000.0), x2))

    banner ("Vertices with system generated vertex labels")
    Vertex.show (vertices2)

    vertices2(0).show ()

end vertexTest

