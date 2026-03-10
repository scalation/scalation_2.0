
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Aug 29 14:14:32 EDT 2020
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    VertexType - a collection of vertices of the same type
 */

package scalation
package database
package graph_relation

import scala.collection.immutable
import scala.collection.mutable.Map
import scala.util.control.Breaks.{break, breakable}

//import scala.collection.immutable.{Vector => VEC}
import scala.collection.mutable.{ArrayBuffer => VEC}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Vertex` class maintains properties for a vertex, e.g., a person.
 *  A vertex is analogous to a tuple in a traditional relational database, except that
 *  foreign keys are replaced with edges connecting to other vertices.
 *  @param prop  maps vertex's property names to property values
 *  @param edge  maps vertex's edge names to other vertices
 */
case class Vertex (prop: Property, edge: Reference = emptyRef)
     extends Serializable:

    override def toString: String =
        s"Vertex (prop = ${prop.mkString (", ")}, edge = ${edge.mkString (", ")})"
    end toString

end Vertex

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Reference` type is a map from reference names to vertices.
 */
type Reference = Map [String, Vertex]

val emptyRef = Map [String, Vertex] ()                                                  // empty references

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VertexType` class collects vertices of the same type, e.g., a person vertex type.
 *  A vertex type is analogous to a relation with no foreign keys in an RDBMS.
 *  @param name     the name of this vertex type
 *  @param schema   the property names for this vertex type
 *  @param eschema  the edge names for this vertex type
 *  @param verts    the set of vertices having this vertex type (extension)
 */
case class VertexType (name: String, schema: VEC [String], eschema: VEC [String], verts: VEC [Vertex])
      extends Serializable:

    private val flaw               = flawf ("VertexType")                               // flaw function
    private var primaryKey: String = null                                               // property used as primary key
    private val index              = Map [ValueType, Vertex] ()                         // primary key -> vertex  
    private var groups: immutable.Map [ValueType, VEC [Vertex]] = null                  // group vertices based on property
    private var count              = 0                                                  // internal counter

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Increment the counter to its next value and return it.
     */
    def next (): Int =
        count += 1
        count
    end next

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check that the properties names are in the schema and the edge names are in the eschema
     *  for this vertex type, returning whether they match.
     */
    def check: Boolean =
        var matched = true
        breakable {
            for v <- verts do
                for pname <- v.prop.keys do
                    if ! (schema contains pname) then
                        flaw ("check", s"error pname = $pname not found in schema")
                        matched = false
                end for
                for ename <- v.edge.keys do
                    if ! (eschema contains ename) then
                        flaw ("check", s"error ename = $ename not found in eschema")
                        matched = false
                end for
                if ! matched then break
            end for
        } // breakable
        matched
    end check

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
    /** Project each vertex in this vertex type down to the given subschema of properties.
     *  @param subschema  the subset of properies to project onto
     */
    def project (subschema: VEC [String]): VertexType =
        VertexType (name + "p" + next (), subschema, eschema,
                    for v <- verts yield Vertex (v.prop.filter (subschema contains _._1)))
    end project

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Project each vertex in this vertex type down to the given subschema of properties.
     *  @param subschema  the subset of properies to project onto
     */
    def project (subschema: String*): VertexType = project (VEC (subschema*))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Project each vertex in this vertex type down to the given esubschema of edges.
     *  @param esubschema  the subset of edges to project onto
     */
    def eproject (esubschema: VEC [String]): VertexType =
        VertexType (name + "e" + next (), schema, esubschema, null)
//                  for v <- verts yield Vertex (v.edge.filter (esubschema contains _._1)))   // FIX
    end eproject

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Project each vertex in this vertex type down to the given esubschema of edges.
     *  @param subschema  the subset of edges to project onto
     */
    def eproject (esubschema: String*): VertexType = eproject (VEC (esubschema*))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Select the vertices in this vertex type that satisfy the predicate.
     *  @param pred  the predicate to satisfy
     */
    def select (pred: Property => Boolean): VertexType =
        VertexType (name + "s" + next (), schema, eschema,
                    for v <- verts if pred (v.prop) yield v)
    end select

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Union this vertex type with a second vertex type.
     *  @param vtype2  the second vertex type
     */
    infix def unionAll (vtype2: VertexType): VertexType =
        VertexType (name + "a" + next (), schema, eschema, verts ++ vtype2.verts)
    end unionAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Union this vertex type with a second vertex type with no duplication.
     *  @param vtype2  the second vertex type
     */
    infix def union (vtype2: VertexType): VertexType =
        VertexType (name + "u" + next (), schema, eschema, (verts ++ vtype2.verts).distinct)
    end union

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make all the vertices in this vertex type distinct, i.e., no duplicate vertices.
     */
    def distinct: VertexType =
        VertexType (name + "d" + next (), schema, eschema, verts.distinct)
    end distinct

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Minus second vertex type from this vertex type.
     *  @param vtype2  the second vertex type
     */
    infix def minus (vtype2: VertexType): VertexType =
        VertexType (name + "m" + next (), schema, eschema, verts diff vtype2.verts)
    end minus

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Intersect this vertex type with a second vertex type with no duplication.
     *  @param vtype2  the second vertex type
     */
    infix def intersect (vtype2: VertexType): VertexType =
        VertexType (name + "i" + next (), schema, eschema, verts intersect vtype2.verts)
    end intersect

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Join this vertex type to a second vertex type (similar to a natural join).
     *  FIX - handle duplicate property/edge names
     *  @param ename   the edge name used to connect with vtype2 (analog of a foreign key)
     *  @param vtype2  the second vertex type
     */
    def join (ename: String, vtype2: VertexType): VertexType =
        val vertices =
            for v <- verts yield
                val link = v.edge(ename)
                Vertex (v.prop +++ link.prop, (v.edge ++ link.edge) -= ename)
            end for
        VertexType (name + "j" + next (), schema ++ vtype2.schema, eschema ++ vtype2.eschema, vertices)
    end join

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Group the vertices within this vertex type by the values of the given property name.
     *  @param pname  the propery name on which to group
     */
    def groupBy (pname: String): VertexType =
        groups  = verts.groupBy [ValueType] (_.prop(pname))                             // dicriminator
        this
    end groupBy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Order the vertices within this vertex type by the values of the given property name.
     *  @param pname  the propery name on which to sort
     */
    def orderBy (pname: String): VertexType =
        VertexType (name + "o" + next (), schema, eschema,
                    verts.sortWith (_.prop(pname) < _.prop(pname)))
    end orderBy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Limit the number of vertices to n after skipping the first s.
     *  @param n  the number of vertices to keep
     *  @param s  the number of vertices to skip
     */
    def limit (n: Int, s: Int = 0): VertexType =
        VertexType (name + "l" + next (), schema, eschema, verts.slice (s, s + n))
    end limit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert the meta-data for this vertex type to a string.
     */
    def meta: String =
        s"""
        name =        $name
        schema =      $schema
        primary key = $primaryKey
        eschema =     $eschema
        # vertices =  ${verts.size}
        """
    end meta

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show the groups that were created.
     */
    def showGroups: Iterable [String] = groups.map (_.toString + "\n")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert the data for this vertex type to a string.
     */
    override def toString: String =
        val vertices = (for v <- verts yield v.toString + "\n").toString.drop (11)
        s"VertexType (schema = $schema, verts = \n $vertices)"
    end toString

end VertexType


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VertexTypeTest` function is used to test the `VertexType` class.
 *  > runMain scalation.database.graph_relation.vertexTypeTest
 */
@main def vertexTypeTest (): Unit =

    val professor = VertexType ("professor",
                                VEC ("pid", "name", "phone"),
                                VEC (),
                                VEC (Vertex (Map ("pid" -> 1, "name" -> "Bob", "phone" -> 1234567)),
                                     Vertex (Map ("pid" -> 2, "name" -> "Sue", "phone" -> 2345678)),
                                     Vertex (Map ("pid" -> 3, "name" -> "Joe", "phone" -> 3456789))))
    println (s"check professor schema = ${professor.check}")
    professor.buildIndex ("pid")

    val course = VertexType ("course",
                             VEC ("cid", "cname", "dept"),
                             VEC ("taughtBy"),
                             VEC (Vertex (Map ("cid" -> 1, "cname" -> "database", "dept" -> "CSCI"),
                                          Map ("taughtBy" -> professor.verts(0))),
                                  Vertex (Map ("cid" -> 2, "cname" -> "networks", "dept" -> "CSCI"),
                                          Map ("taughtBy" -> professor.verts(1))),
                                  Vertex (Map ("cid" -> 3, "cname" -> "ai",       "dept" -> "ARTI"),
                                          Map ("taughtBy" -> professor.verts(2)))))
    println (s"check course schema = ${course.check}")
    course.buildIndex ("cid")

    println (s"professor.meta = ${professor.meta}")
    println (s"professor      = $professor")
    println (s"course.meta    = ${course.meta}")
    println (s"course         = $course")

    println ("query1: limit")
    val query1 = professor.limit (2)
    println (query1)

    println ("query2: find")
    val query2 = professor.find (2)
    println (query2)

    println ("query3: project")
    val query3 = professor.project (VEC ("name"))
    println (query3)

    println ("query3b: project")
    val query3b = professor.project ("name")
    println (query3b)

    println ("query4: select ==")
    val query4 = professor.select ((p: Property) => p("name") == "Sue")
    println (query4)

    println ("query4b: select ==")
    val query4b = professor.select (_("name") == "Sue")
    println (query4b)

    println ("query4c: select <")
    val query4c = professor.select (_("phone") < 2000000)
    println (query4c)

    println ("query5: unionAll")
    val query5 = professor unionAll professor
    println (query5)

    println ("query6: union")
    val query6 = professor union professor
    println (query6)

    println ("query7: minus")
    val query7 = professor minus professor
    println (query7)

    println ("query8: intersect")
    val query8 = professor intersect professor
    println (query8)

    println ("query9: join")
    val query9 = course.join ("taughtBy", professor)
    println (query9)

    println ("query10: groupBy")
    val query10 = course.groupBy ("dept")
    println (query10.showGroups)

    println ("query11: orderBy")
    val query11 = course.orderBy ("dept")
    println (query11)

end vertexTypeTest

