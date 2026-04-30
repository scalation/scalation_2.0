
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Aug 29 14:14:32 EDT 2020
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    EdgeType - a collection of edges of the same type
 */

package scalation
package database
package graph

//import scala.collection.immutable.{Vector => VEC}
import scala.collection.mutable.{ArrayBuffer => VEC}

import scala.collection.mutable.Map
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat.VectorD
import scalation.scala2d.Colors.{Color, blue}
import scalation.scala2d.{CurvilinearShape, Arrow, QArrow}

import Vertex.prt

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `EdgeType` class collects edges of the same type, e.g., knows relationship type.
 *  An edge-type is analogous to a relation with foreign keys in an RDBMS.
 *  @param _name   the name of this edge-type ('name' from `Identifiable`)
 *  @param from    the source vertex
 *  @param schema  the property names for this edge-type
 *  @param to      the target vertex
 *  @param edges   the set of edges having this edge-type (extension)
 *  @param color   the display color for edges of this type
 *  @param shape   the display shape template for edges of this type
 */
class EdgeType (_name: String,
                val from:  VertexType, val schema: Schema, val to: VertexType,
                val edges: VEC [Edge] = VEC [Edge] (),
                val color: Color = blue,
                val shape: CurvilinearShape = Arrow ())
      extends Identifiable (_name)
         with Serializable:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a new shape object based on the shape template.
     */
    def newShapeObj: CurvilinearShape =
        shape match
        case _ : Arrow => Arrow ()
        case _        => QArrow ()
        end match
    end newShapeObj

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check that the properties names are in the schema for this edge-type, returning
     *  whether they match the schema.
     */
    def check: Boolean = true                                           // FIX - implement

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Rename the old property name with the new property name.
     *  @param pOld  existing property name
     *  @param pNew  new property name
     */
    def rename (pOld: String, pNew: String): EdgeType =
        new EdgeType (name + "_r", from, schema.updated (schema.indexOf(pOld), pNew), to,
                  for e <- edges yield Edge (e.from, e.prop.map (p => if p._1 == pOld then pNew -> p._2 else p), e.to) )
    end rename

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Expand this edge-type with its 'to' vertex-type, appending its properties.
     */
    def expandTo: EdgeType =
        val edgez = for e <- edges yield Edge (e.from, e.prop +++ e.to.prop, null)
        new EdgeType (name + "_et", from, schema ++ to.schema, null, edgez)
    end expandTo

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Expand this edge-type with its from vertex-type.
     */
    def expandFrom: EdgeType =
        val edgez = for e <- edges yield Edge (null, e.from.prop +++ e.prop, e.to)
        new EdgeType (name + "_ef", null, from.schema ++ schema, to, edgez)
    end expandFrom

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Expand this edge-type with both its from and to vertex-types.
     */
    def expand: VertexType =
        val vertices = for e <- edges yield Vertex (e.from.prop +++ e.prop +++ e.to.prop)
        new VertexType (name + "_e", from.schema ++ schema ++ to.schema, vertices)
    end expand

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Project each edge in this edge-type down to the given subschema x of properties.
     *  @param x  the subset of property names (subschema) to project onto
     */
    def project (x: Schema): EdgeType =
        new EdgeType (name + "_p", from, x, to,
                      for e <- edges yield Edge (e.from, e.prop.filter (x contains _._1), e.to))
    end project

    inline def project (x: String): EdgeType = project (strim (x))

    inline def π (x: String): EdgeType = project (strim (x))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Select the edges in this edge-type that satisfy the predicate.
     *  @param pred  the predicate to satisfy
     */
    def select (pred: Property => Boolean): EdgeType =
        new EdgeType (name + "_s", from, schema, to,
                      for e <- edges if pred (e.prop) yield e)
    end select

    inline def σ (pred: Property => Boolean): EdgeType = select (pred)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Union this edge-type with a second edge-type.
     *  @param et2  the second edge-type
     */
    infix def unionAll (et2: EdgeType): EdgeType =
        new EdgeType (name + "_ua_" + et2.name, from, schema, to, edges ++ et2.edges)
    end unionAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Union this edge-type with a second edge-type with no duplication.
     *  @param et2  the second edge-type
     */
    infix def union (et2: EdgeType): EdgeType =
        new EdgeType (name + "_u_" + et2.name, from, schema, to, (edges ++ et2.edges).distinct)
    end union

    inline def ∪ (et2: EdgeType): EdgeType = union (et2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Intersect this edge-type with a second edge-type.
     *  @param et2  the second edge-type
     */
    infix def intersect (et2: EdgeType): EdgeType =
        new EdgeType (name + "_u_" + et2.name, from, schema, to, edges intersect et2.edges)
    end intersect

    inline def ∩ (et2: EdgeType): EdgeType = intersect (et2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make all the edges in this edge-type distinct, i.e., no duplicate edges.
     */
    def distinct: EdgeType =
        new EdgeType (name + "_d", from, schema, to, edges.distinct)
    end distinct

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Subtract (minus) second edge-type from this edge-type.
     *  @param et2  the second edge-type
     */
    infix def minus (et2: EdgeType): EdgeType =
        new EdgeType (name + "_m_" + et2.name, from, schema, to, edges diff et2.edges)
    end minus

    inline def - (et2: EdgeType): EdgeType = minus (et2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Order the edges within this edge-type by the values of the given property name.
     *  @param pname  the property name on which to sort
     */
    def orderBy (pname: String): EdgeType =
        new EdgeType (name + "_o", from, schema, to, edges.sortWith (_.prop(pname) < _.prop(pname)))
    end orderBy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Limit the number of edges to n after skipping the first s.
     *  @param n  the number of edges to keep
     *  @param s  the number of edges to skip
     */
    def limit (n: Int, s: Int = 0): EdgeType =
        new EdgeType (name + "_l", from, schema, to, edges.slice (s, s + n))
    end limit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Insert a single edge into the existing list of edges in this edge-type.
     *  returning the updated edge type/set.
     *  @param e  the edge to insert
     */
    def insert (e: Edge): EdgeType =
        edges :+ e
        this
    end insert

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add edges into the existing list of edges in this edge-type.
     *  @param es  the edges to add
     */
    def add (es: Edge*): Unit = edges ++= es

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Delete a single edge from the existing list of edges in this edge-type.
     *  returning the updated edge type/set.
     *  @param e  the edge to delete
     */
    def delete (e: Edge): EdgeType =
        edges -= e
        this
    end delete

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print this edge-type.
     */
    def print (): Unit =
        banner ("Edge Type:")
        println (s"et = $this")
    end print

    private val width_ = 18

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SHOW/print this table, one edge per row.
     *  Formatting: regular column is 'width_' chars wide, 'X' is double that
     *  FIX - currently only works for width_, not width
     *  @param rng  the range of edges to show (e.g, 0 until 10), defaults to all
     */
    def show (rng: Range = edges.indices): Unit =
        if rng.size > 0 then
            val tuple0 = edges (0).prop
            val len = width_ * (tuple0.size + 2)                                  // + 2 for 'form' and 'to'
            val wj  = width_

            if len != 0 then

                println (s"\n>> Table $name with ${rng.size} rows")
                println ("|-" + "-" * len + "-|")
                System.out.print ("| ")
                prt ("from", wj)
                for (k, _) <- tuple0 do prt (k, wj)
                prt ("to", wj)
                println (" |")

                println ("|-" + "-" * len + "-|")
                for i <- rng do
                    System.out.print ("| ")
                    val tuple_i = edges (i).prop

                    if edges (i).from == null then prt ("null-vertex", wj)
                    else prt (edges (i).from.name, wj)

                    for (_, v) <- tuple_i do prt (v, wj)

                    if edges(i).to == null then prt ("null-vertex", wj)
                    else prt (edges(i).to.name, wj)
        
                    println (" |")
                end for

                println ("|-" + "-" * len + "-|")
            end if
        end if
    end show

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this edge-type to a string.
     */
    override def toString: String =
        val edgesStr = if edges == null then "()\n"
                       else edges.map ("\n\t\t" + _.toString)
        s"EdgeType (id = $id, name = $name, schema = ${stringOf (schema)},\n edges = $edgesStr \n)"
                   .replace ("ArrayBuffer", "VEC")
    end toString

end EdgeType
 

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `EdgeType` companion object provides factory functions for creating edge-types.
 */
object  EdgeType:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an edge-type for maintaing edges of same type.
     *  @param name    the name of this edge-type ('name' from `Identifiable`)
     *  @param from    the source vertex
     *  @param schema  the property names for this edge-type
     *  @param to      the target vertex
     *  @param edges   the set of edges having this edge-type (extension)
     *  @param color   the display color for edges of this type
     *  @param shape   the display shape template for edges of this type
     */
    def apply (name: String,
               from:  VertexType, schema: String, to: VertexType,
               edges: VEC [Edge] = VEC [Edge] (),
               color: Color = blue,
               shape: CurvilinearShape = Arrow ()): EdgeType =
        new EdgeType (name, from, strim (schema), to, edges, color, shape)
    end apply

end EdgeType


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `edgeTypeTest` main function is used to test the `EdgeType` class.
 *  > runMain scalation.database.graph.edgeTypeTest
 */
@main def edgeTypeTest (): Unit =

    val x0 = VectorD (200, 100, 40, 40)                           // vertex coordinates, size: x, y, w, h
    val x1 = VectorD (100, 400, 40, 40)
    val x2 = VectorD (500, 500, 40, 40)

    val v = VEC (
        new Vertex ("Bob", Map ("name" -> "Bob", "state" -> "GA", "salary" -> 85000.0), x0),
        new Vertex ("Sue", Map ("name" -> "Sue", "state" -> "FL", "salary" -> 95000.0), x1),
        new Vertex ("Joe", Map ("name" -> "Joe", "state" -> "GA", "salary" -> 99000.0), x2))

    val edges = VEC (
        new Edge ("knows", v(0), Map ("type" -> "knows", "since" -> 5), v(1)),
        new Edge ("knows", v(1), Map ("type" -> "knows", "since" -> 2), v(0), -1),
        new Edge ("knows", v(2), Map ("type" -> "knows", "since" -> 4), v(0)))

    val vt0 = VertexType ("person", "name, state, salary", v)
    val et0 = EdgeType ("knows", vt0, "type", vt0, edges)

    println (s"vertex-type vt0 = $vt0")
    println (s"edge-type   et0 = $et0")

    et0.show ()

end edgeTypeTest

