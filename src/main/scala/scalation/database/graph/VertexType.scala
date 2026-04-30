
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Abby Moore, Piyush Subedi
 *  @version 2.0
 *  @date    Sat Aug 29 14:14:32 EDT 2020
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    VertexType - a collection of vertices of the same type
 */

package scalation
package database
package graph

//import scala.collection.immutable.{Vector => VEC}
import scala.collection.mutable.{ArrayBuffer => VEC}

import scala.collection.mutable.Map
import scala.runtime.ScalaRunTime.stringOf
import scala.util.control.Breaks.{break, breakable}

import scalation.mathstat.VectorD
import scalation.scala2d.Colors.{Color, yellow}
import scalation.scala2d.{Ellipse, Octagon, Rectangle, RoundRectangle, Shape}

import Vertex.prt

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VertexType` class collects vertices of the same type, e.g., a person vertex-type.
 *  A vertex-type is analogous to a relation with no foreign keys in an RDBMS.
 *  @param _name   the name of this vertex-type ('name' form `Identifiable`)
 *  @param schema  the property names for this vertex-type
 *  @param verts   the set of vertices having this vertex-type (extension)
 *  @param color   the display color for vertices of this type
 *  @param shape   the display shape template for vertices of this type
 */
class VertexType (_name: String, val schema: Schema,
                  val verts: VEC [Vertex] = VEC [Vertex] (),
                  val color: Color = yellow,
                  val shape: Shape = Ellipse ())
      extends Identifiable (_name)
         with Serializable:

    private val debug     = debugf ("VertexType", true)                        // debug function
    private val flaw      = flawf ("VertexType")                               // flaw function
    private var primaryKey: String = null                                      // property used as primary key
    private val index     = Map [ValueType, Vertex] ()                         // primary key -> vertex

    def getPK: String = primaryKey

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a new shape object based on the shape template.
     */
    def newShapeObj: Shape =
        shape match
        case _ : Ellipse   => Ellipse ()
        case _ : Octagon   => Octagon ()
        case _ : Rectangle => Rectangle ()
        case _             => RoundRectangle ()
        end match
    end newShapeObj

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check that the properties names are in the schema for this vertex-type, returning
     *  whether they match the schema.
     */
    def check: Boolean =
        var matched = true
        breakable {
            for v <- verts; pname <- v.prop.keys if ! (schema contains pname) do
                println (s"check: error pname = $pname not found in schema")
                matched = false
                break ()
            end for
        } // breakable
        matched
    end check

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check whether the given property name is defined for all the vertices in this vertex-type.
     *  Useful check before, groupBy and project operators.
     *  @param pname  the given property name
     */
    def checkMissing (pname: String): Boolean =
        var defined = true
        breakable {
            for i <- verts.indices if ! verts(i).prop.isDefinedAt (pname) do
                debug ("checkMissing", s"vertex $i is missing the $pname property")
                defined = false
                break ()
            end for
        } // breakable
        defined
    end checkMissing

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build the index for the primary key.
     *  @param pkey  the property designated as the primary key
     */
    def buildIndex (pkey: String): Unit =
        if ! (schema contains pkey) then
            flaw ("buildIndex", s"primary key property = $pkey is not in the schema")
            return
        primaryKey = pkey
        for v <- verts do
            val key = v.prop(pkey)                                                      // the primary key
            val old = index.put (key, v)                                                // the old vertex
            if old != None then
                flaw ("buildIndex", s"duplicate primary key = $key inserted, old = $old")
        end for
    end buildIndex

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the unique vertex based on the primary key.
     *  @param key  the primary key
     */
    def find (key: ValueType): Vertex = index(key)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vertices where property pname has value pval.
     *  @param pname  the property name
     *  @param pval   the property value
     */
    def == (pname: String, pval: ValueType): VertexType =
        new VertexType (name + "_", schema,
                        for v <- verts if v.prop(pname) == pval yield v)
    end ==

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vertices where property pname is less than value pval.
     *  @param pname  the property name
     *  @param pval   the property value
     */
    def < (pname: String, pval: Double): VertexType =            // FIX - want Value not Double
        new VertexType (name + "_", schema,
                        for v <- verts if v.prop(pname) < pval yield v)
    end <

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Rename the old property name with the new property name.
     *  @param pOld  existing property name
     *  @param pNew  new property name
     */
    def rename (pOld: String, pNew: String): VertexType =
        new VertexType (name + "_r", schema.updated (schema.indexOf (pOld), pNew),
                    for v <- verts yield
                        Vertex (v.prop.map (p => if p._1 == pOld then pNew -> p._2 else p)) )
    end rename

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Project each vertex in this vertex-type onto to the given subschema x of properties.
     *  @param x  the subset of property names to project onto
     */
    def project (x: Schema): VertexType =
        if ! subset (x, schema) then flaw ("project", "subschema x does not follow schema")
        new VertexType (name + "_p", x,
                        for v <- verts yield Vertex (v.prop.filter (x contains _._1)))
    end project

    inline def project (x: String): VertexType = project (strim (x))

    inline def π (x: String): VertexType = project (strim (x))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Select the vertices in this vertex-type that satisfy the predicate.
     *  @param pred  the predicate to satisfy
     */
    def select (pred: Property => Boolean): VertexType =
        new VertexType (name + "_s", schema,
                        for v <- verts if pred (v.prop) yield v)
    end select

    inline def σ (pred: Property => Boolean): VertexType = select (pred)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Union this vertex-type with a second vertex-type.
     *  @param vt2  the second vertex-type
     */
    infix def unionAll (vt2: VertexType): VertexType =
        new VertexType (name + "_ua_" + vt2.name, schema, verts ++ vt2.verts)
    end unionAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Union this vertex-type with a second vertex-type with no duplication.
     *  @param vt2  the second vertex-type
     */
    infix def union (vt2: VertexType): VertexType =
        new VertexType (name + "_u_" + vt2.name, schema, (verts ++ vt2.verts).distinct)
    end union

    inline def ∪ (vt2: VertexType): VertexType = union (vt2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Intersect this vertex-type with a second vertex-type.
     *  @param vt2  the second vertex-type
     */
    infix def intersect (vt2: VertexType): VertexType =
        new VertexType (name + "_i_" + vt2.name, schema, (verts intersect vt2.verts))
    end intersect

    inline def ⋂ (vt2: VertexType): VertexType = intersect (vt2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make all the vectices in this vertex-type distinct, i.e., no duplicate vertices.
     */
    def distinct: VertexType =
        new VertexType (name + "_d", schema, verts.distinct)
    end distinct

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Subtract (minus) second vertex-type from this vertex-type.
     *  @param vt2  the second vertex-type
     */
    infix def minus (vt2: VertexType): VertexType =
        new VertexType (name + "_m_" + vt2.name, schema, verts diff vt2.verts)
    end minus

    inline def - (vt2: VertexType): VertexType = minus (vt2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Group the vertices within this vertex-type by the values of the given property name
     *  and apply the aggregate function agg_fn on the agg_name property.
     *  @param pname     the property name on which to group
     *  @param agg_name  the property name for the aggregate
     *  @param agg_fn    the aggrgate function
     */
    def groupBy (pname: String, agg_name: String, agg_fn: Double => Double): VertexType =
        debug ("groupBy", s"group $schema by $pname")
        if ! (schema contains pname) then flaw ("groupBy", s"property $pname missing from schema")
        if checkMissing (pname) then flaw ("groupBy", s"property $pname missing from a vertex")

        val groups   = verts.groupBy [ValueType] (_.prop(pname))                        // discriminator
        val vertices = VEC [Vertex] ()
        for g <- groups; v <- g._2 do
            vertices += Vertex (Map (pname -> v.prop(pname), (agg_name -> agg_fn (v.prop(agg_name).toDouble))))
        end for
        new VertexType (name + "_g", Array (pname, agg_name), vertices)
    end groupBy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Order the vertices within this vertex-type by the values of the given property name.
     *  @param pname  the property name on which to sort
     */
    def orderBy (pname: String): VertexType =
        new VertexType (name + "_o", schema, verts.sortWith (_.prop(pname) < _.prop(pname)))
    end orderBy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Limit the number of vertices to n after skipping the first s.
     *  @param n  the number of vertices to keep
     *  @param s  the number of vertices to skip
     */
    def limit (n: Int, s: Int = 0): VertexType =
        new VertexType (name + "_l", schema, verts.slice (s, s + n))
    end limit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Insert a single vertex into the existing list of vertices in this vertex-type,
     *  returning the updated vertex type/set.
     *  @param v  the vertex to insert
     */
    def insert (v: Vertex): VertexType =
        verts += v
        this
    end insert

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add vertices into the existing list of vertices in this vertex-type.
     *  @param vs  the vertex to insert
     */
    def add (vs: Vertex*): Unit = verts ++= vs

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Delete a single vertex from the existing list of vertices in this vertex-type,
     *  returning the updated vertex type/set.
     *  @param v  the vertex to delete
     */
    def delete (v: Vertex): VertexType =
        verts -= v
        this
    end delete

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print this vertex-type.
     */
    def print (): Unit =
        banner ("Vertex Type:")
        println (s"vt = $this")
    end print

    private val width_ = 18

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SHOW/print this vertex-type, one vertex per row.
     *  Formatting: regular column is 'width_' chars wide, 'X' is double that
     *  FIX - currently only works for width_, not width
     *  @param rng  the range of vertices to show (e.g, 0 until 10), defaults to all
     */
    def show (rng: Range = verts.indices): Unit =
        val len = width_ * (schema.size)
        val wj  = width_

        if len != 0 then
            println (s"\n>> Table $name with ${rng.size} rows")
            println ("|-" + "-" * len + "-|")
            System.out.print ("| ")
            for k <- schema do prt (k, wj)
            println (" |")

            println ("|-" + "-" * len + "-|")
            for i <- rng do
                System.out.print ("| ")
                val tuple_i = verts(i).prop
                for (_, v) <- tuple_i do prt (v, wj)
                println (" |")
            end for
            println ("|-" + "-" * len + "-|")
        end if
    end show

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this vertex-type to a string.
     */
    override def toString: String =
        val vertsStr = if verts == null then "()\n"
                       else verts.map ("\n\t\t" + _.toString)
        s"VertexType (id = $id, name = $name, schema = ${stringOf (schema)},\n verts = $vertsStr \n)"
                     .replace ("ArrayBuffer", "VEC")
    end toString

end VertexType


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VertexType` companion object provides factory functions for building vertex-types.
 */
object VertexType:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `VertexType` class collects vertices of the same type, e.g., a person vertex-type.
     *  A vertex-type is analogous to a relation with no foreign keys in an RDBMS.
     *  @param name   the name of this vertex-type ('name' form `Identifiable`)
     *  @param schema  the property names for this vertex-type
     *  @param verts   the set of vertices having this vertex-type (ext ension)
     *  @param color   the display color for vertices of this type
     *  @param shape   the display shape template for vertices of this type
     */
    def apply (name: String, schema: String,
               verts: VEC [Vertex] = VEC [Vertex] (),
               color: Color = yellow,
               shape: Shape = Ellipse ()): VertexType =
        new VertexType (name, strim (schema), verts, color, shape)
    end apply

end VertexType


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `vertexTypeTest` main function is used to test the `VertexType` class.
 *  > runMain scalation.database.graph.vertexTypeTest
 */
@main def vertexTypeTest (): Unit =

    val x0 = VectorD (200, 100, 40, 40)                           // vertex coordinates, size: x, y, w, h
    val x1 = VectorD (100, 400, 40, 40)
    val x2 = VectorD (500, 500, 40, 40)

    val vertices = VEC (
        new Vertex ("Bob", Map ("name" -> "Bob", "state" -> "GA", "salary" -> 85000.0), x0),
        new Vertex ("Sue", Map ("name" -> "Sue", "state" -> "FL", "salary" -> 95000.0), x1),
        new Vertex ("Joe", Map ("name" -> "Joe", "state" -> "GA", "salary" -> 99000.0), x2))
 
    val vt0 = VertexType ("person", "name, state, salary", vertices)

    println (s"vertex-type vt0 = $vt0")

    vt0.show()

end vertexTypeTest

