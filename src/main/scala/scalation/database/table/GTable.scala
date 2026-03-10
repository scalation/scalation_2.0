
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Jul  8 23:10:18 EDT 2022
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Graph Algebra (GA) for Graph DBMS
 */

package scalation
package database
package table 

//import com.google.gson.Gson

//import java.io.PrintWriter

import scala.collection.mutable.{ArrayBuffer => Bag, Map}
import scala.runtime.ScalaRunTime.stringOf

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Edge` class includes three parts:  The edge attributes in the form of
 *  a tuple of values, the source (from) vertex and target (to) vertex.
 *  @param from   the source vertex
 *  @param to     the target vertex
 *  @param tuple  the tuple part of the edge (for edge attributes)
 */
case class Edge (from: Vertex, to: Vertex, tuple: Tuple = null):

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an edge in the reverse direction.
     */
    def reverse: Edge = Edge (to, from, tuple)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert the tuple within this edge to a string.
     */
    override def toString: String = s"edge: ${stringOf (tuple)}"

end Edge


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Vertex` class extends the notion of `Tuple` into values stored in the tuple
 *  part, along with foreign keys links captured as outgoing edges.
 *  @param tuple  the tuple part of the vertex
 */
case class Vertex (tuple: Tuple):

    val edge = Map [String, Bag [Edge]] ()                      // map edge-label -> { edges }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the neighboring vertices reachable in one-hop by taking edges
     *  emanating from this vertex.
     */ 
    def neighbors: Bag [Vertex] =
        val vs = Bag [Vertex] ()
        val es: Iterable [Bag [Edge]] = edge.values
        for e <- es do vs ++ e.map (_.to)
        vs
    end neighbors

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the neighboring vertices reachable in one-hop by taking edge with
     *  edge-label elab emanating from this vertex.
     *  @param elab  the edge-label (wild-card "*" means take all edges)
     */ 
    def neighbors (elab: String): Bag [Vertex] =
        if elab == "*" then neighbors else edge (elab).map (_.to)
    end neighbors

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the neighboring vertices reachable in one-hop by taking edge with
     *  edge-label elab emanating from this vertex that are in the reference table.
     *  @param ref  the edge-label and reference table
     */ 
    def neighbors (ref: (String, GTable)): Bag [Vertex] =
        edge (ref._1).map (_.to) intersect ref._2.vertices
    end neighbors

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert the tuple within this vertex to a string.
     */
    override def toString: String = s"vertex: ${stringOf (tuple)}"

end Vertex


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `GTable` companion object provides factory methods for creating graph-tables.
 *  Supported domains/data-types are 'D'ouble, 'I'nt, 'L'ong, 'S'tring, and 'T'imeNum.
 */
object GTable:

    import Table.makeTuple

    private val debug = debugf ("GTable", true)                             // debug function
    private val flaw  = flawf ("GTable")                                    // flaw function
    private val cntr  = Counter ()                                          // counter for generating unique names

    private var useFullPath = false                                         // defaults to using relative file paths
    private var limit       = -1                                            // limit on number of lines to read

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the full-path flag to the value of parameter fullPath.
     *  @param fullPath  flag indicating whether full or relative paths should be used
     */
    def setFullPath (fullPath: Boolean = true): Unit = { useFullPath = fullPath }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the limit on the number of lines to read to lim.
     *  @param lim  the limit on the number of lines to read (<= 0 => unlimited)
     */
    def setLimit (lim: Int): Unit = { limit = lim }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a graph-table given convenient string specifications.
     *  @param name    the name of the graph-table
     *  @param schema  the attributes for the graph-table
     *  @param domain  the domains/data-types for attributes ('D', 'I', 'L', 'S', 'X', 'T')
     *  @param key     the attributes forming the primary key
     */
    def apply (name: String, schema: String, domain: String, key: String): GTable =
        new GTable (name, strim (schema), strim (domain).map (_.head), strim (key))
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Read the graph-table with the given name into memory loading its columns with data
     *  from the CSV file named fileName.  The attribute names are read from the FIRST LINE.
     *  @see scalation.readFileIntoArray
     *  @param fileName  the file name (or file-path) of the data file
     *  @param name      the name of the graph-table
     *  @param domain    the domains/data-types for attributes ('D', 'I', 'L', 'S', 'X', 'T')
     *  @param key       the attributes forming the primary key
     *  @param pos_      the sequence of column positions in the input file to be used (null => select all)
     *  @param sep       the element separation string/regex (e.g., "," ";" " +")
     */
    def load (fileName: String, name: String, domain: Domain, key: String,
              pos_ : Array [Int] = null, sep: String = ","): GTable =

        debug ("load", s"""fileName = $fileName, name = $name, domain = ${stringOf (domain)}, key = $key,
                       pos_ = $pos_, sep = '$sep'; useFullPath = $useFullPath, limit = $limit""")

        val pos    = if pos_ == null then Array.range (0, domain.size) else pos_
        val schema = Array.ofDim [String] (domain.size)

        if pos.size != domain.size then flaw ("apply", "pos size should be same as domain size")

        var s: GTable = null                    // new GTable (name, schema, domain, strim (key))

//      val lines = getFromURL_File (fileName)                              // read the CSV file
        val lines = readFileIntoArray (fileName, useFullPath, limit)        // read the CSV file
        var l_no  = 0                                                       // the line number

        println (s"lines(0) = ${lines(0)}")

        for ln <- lines do                                                  // iterate by lines in file

            if l_no == 0 then                                               // FIRST LINE - for schema
                val header = ln.split (sep, -1).map (_.trim)                // array of column names
                debug ("load", s"header = ${stringOf (header)}")
                for j <- pos.indices do schema(j) = header(pos(j))          // use those at positions in pos
                s = new GTable (name, schema, domain, strim (key))          // make graph-table after schema is formed

            else                                                            // REMAINING LINES
                val token = ln.split (sep, -1).map (_.trim)                 // array of token strings
                s.vertices += Vertex (makeTuple (token, domain, pos))

            l_no += 1
        end for
        s
    end load

end GTable

import GTable.{cntr, flaw, debug}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `GTable` class (graph-table) stores graph-table data and implements
 *  graph algebra operators.
 *  Supported domains/data-types are 'D'ouble, 'I'nt, 'L'ong, 'S'tring, and 'T'imeNum.
 *  @param name_    the name of the graph-table
 *  @param schema_  the attributes for the graph-table
 *  @param domain_  the domains/data-types for attributes ('D', 'I', 'L', 'S', 'X', 'T')
 *  @param key_     the attributes forming the primary key
 */
class GTable (name_ : String, schema_ : Schema, domain_ : Domain, key_ : Schema)
     extends Table (name_, schema_, domain_, key_)
        with Serializable:

    val vertices = Bag [Vertex] ()                                          // collection of related vertices
    val edgeType = Map [String, (GTable, Boolean)] ()                       // collection of outgoing edge types

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether this graph-table contains tuple u.
     *  @param t  the tuple to look for
     */
    override def contains (t: Tuple): Boolean = vertices.exists (_.tuple sameElements t)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the primary key for the given vertex.
     *  @param v  the given vertex
     */
    def getPkey (v: Vertex): Tuple = pull (v.tuple, key)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add an edge type to this graph-table (analog of a foreign key).
     *  @param elab  the edge-label for this edge type
     *  @param to      the graph-table/vertex type for the target vertices
     *  @param unique  whether the target vertex is unique, e.g., many-to-one relationship
     */
    def addEdgeType (elab: String, to: GTable, unique: Boolean = true): Unit =   
        edgeType    += elab -> (to, unique)
        to.edgeType += elab -> (this, false)
    end addEdgeType

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add a vertex to this graph-table (analog of a tuple).
     *  @param v  the vertex to add
     */
    def add (v: Vertex): Vertex =
         if typeCheck (v.tuple) then vertices += v
         v
    end add

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create and add a vertex to this graph-table.
     *  @param t  the tuple for creating a vertex to add
     */
    inline def addV (t: Tuple): Vertex = add (Vertex (t))

    inline def addV (x: ValueType*): Vertex = add (Vertex (x.toArray))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add an edge e from vertex u (to vertex v) with edge-label elab.
     *  @param u     the source vertex
     *  @param elab  the edge-label, e.g., taughtBy
     *  @param e     the edge (tuple, from (u), to (v))
     */
    def addE (u: Vertex, elab: String, e: Edge): GTable =
        val eType = edgeType
        if eType.keySet contains elab then
            val eset = u.edge.getOrElse (elab, null)

            if eset == null then u.edge += elab -> Bag (e)
            else if eType(elab)._2 then
                flaw ("addE", s"$name: attempt to link to multiple targets vertices when edge type is unique, elab = $elab")
            else
                eset += e
        else 
            flaw ("addE", s"elab = $elab not an edge type for $name")
        this
    end addE

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add an edge e from vertex e.to (to vertex e.from) with edge-label elab.
     *  @param elab  the edge-label, e.g., taughtBy
     *  @param e     the edge (from (u), to(v), tuple)
     */
    def addE (elab: String, e: Edge): GTable = addE (e.to, elab: String, e)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add two edges from vertex u to vertex v and from v to u both with edge-label elab.
     *  @param u      the source vertex, a student
     *  @param elab   the edge-label, e.g., "takes" as in student takes course
     *  @param e      the outgoing edge
     *  @param elab2  the reverse edge-label, e.g., "enrolls" as in course enrolls student
     *  @param v      the target vertex, e.g., a course
     *  @param tab    the target vertex's graph-table, e.g., course
     */
    def add2E (u: Vertex, elab: String, e: Edge, elab2: String, v: Vertex, tab: GTable): GTable =
        addE (u, elab, e)
        tab.addE (v, elab2, e.reverse)
        this
    end add2E

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add two edges from vertex u to vertex v and from v to u both with edge-label elab.
     *  @param elab   the edge-label, e.g., "takes" as in student takes course
     *  @param e      the outgoing edge
     *  @param elab2  the reverse edge-label, e.g., "enrolls" as in course enrolls student
     *  @param tab    the target vertex's graph-table, e.g., course
     */
    def add2E (elab: String, e: Edge, elab2: String, tab: GTable): GTable =
        add2E (e.from, elab, e, elab2, e.to, tab)
        this
    end add2E
 
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add edges from vertex u to vertices vs with edge-label elab.
     *  @param u     the source vertex, e.g., professor
     *  @param elab  the edge-label, e.g., taught
     *  @param vs    the target vertices, e.g., courses being taught
     */
    def addEs (u: Vertex, elab: String, es: Bag [Edge]): Unit =
        val eType = edgeType
        if eType.keySet contains elab then
            if eType(elab)._2 then
                flaw ("addEs", "attempt to link to multiple targets vertices when edge type is unique")
            else
                val eset = u.edge.getOrElse (elab, null)
                if eset == null then u.edge += elab -> es
                else eset ++= es
        else 
            flaw ("addEs", s"elab = $elab not an edge type for $name")
    end addEs

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** PROJECT the vertices in this graph-table onto the given attribute names.
     *  @param x  the schema/attribute names to project onto
     */
    override def project (x: Schema): GTable =
        val newKey = if subset (key, x) then key else x

        val s = new GTable (s"${name}_p_${cntr.inc ()}", x, pull (x), newKey)

        s.vertices ++= (for v <- vertices yield Vertex (pull (v.tuple, x)))
        s
    end project

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SELECT the vertices in this graph-table that satisfy the predicate p.
     *  @param p  the predicate (`Boolean` function) to be satisfied
     */
    override def select (predicate: Predicate): GTable =
        val s = new GTable (s"${name}_s_${cntr.inc ()}", schema, domain, key)

        s.vertices ++= (for v <- vertices if predicate (v.tuple) yield v)
        s
    end select

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SELECT the vertices in this graph-table that satisfy the given simple (3 token)
     *  condition.
     *  @param condition  the simple condition string "a1 op a2" to be satisfied, where
     *                    a1 is attribute, op is comparison operator, a2 is attribute or value
     */
    override def select (condition: String): GTable =
        val s = new GTable (s"${name}_s_${cntr.inc ()}", schema, domain, key)

        val (tok, twoAtrs) = parseCond (condition)
        val (a1, op, a2) = (tok(0), tok(1), tok(2))
        debug ("select", s"(a1, op, a2) = ($a1, $op, $a2)")
        s.vertices ++= selectVerts (a1, op, a2, twoAtrs)
        s
    end select

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vertices in this graph-table that satisfy the given simple (3 token)
     *  condition.
     *  @param a1       the left attribute
     *  @param op       the comparison operator (==, !=, <, <=. >, >=)
     *  @param a2       the right attribute or value
     *  @param twoAtrs  the whether a2 is an attribute or value
     *  @param verts    the initial collection of vertices
     */
    private def selectVerts (a1: String, op: String, a2: String, twoAtrs: Boolean,
                            verts: Bag [Vertex] = vertices): Bag [Vertex] =
        if twoAtrs then                                                     // a1 and a2 are attributes
            val a2_ = a2.toString
            op match
            case "==" => verts.filter (v => v.tuple(on(a1)) == v.tuple(on(a2_)))
            case "!=" => verts.filter (v => v.tuple(on(a1)) != v.tuple(on(a2_)))
            case "<"  => verts.filter (v => v.tuple(on(a1)) <  v.tuple(on(a2_)))
            case "<=" => verts.filter (v => v.tuple(on(a1)) <= v.tuple(on(a2_)))
            case ">"  => verts.filter (v => v.tuple(on(a1)) >  v.tuple(on(a2_)))
            case ">=" => verts.filter (v => v.tuple(on(a1)) >= v.tuple(on(a2_)))
            case _    => flaw ("select", s"$op is an unrecognized operator"); verts
        else                                                                // a1 is attribute, a2 is value
            val col = on(a1)
            val a2_ : ValueType = string2Dom (a2, domain (col))
            op match
            case "==" => verts.filter (v => v.tuple(col) == a2_)
            case "!=" => verts.filter (v => v.tuple(col) != a2_)
            case "<"  => verts.filter (v => v.tuple(col) <  a2_)
            case "<=" => verts.filter (v => v.tuple(col) <= a2_)
            case ">"  => verts.filter (v => v.tuple(col) >  a2_)
            case ">=" => verts.filter (v => v.tuple(col) >= a2_)
            case _    => flaw ("select", s"$op is an unrecognized operator"); verts
        end if
    end selectVerts

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** UNION this graph-table and r2.  Check that the two tables are compatible.
     *  If they are not, return the first table.
     *  Caveat:  Assumes the key from the first table still works (@see create_index)
     *  Acts like union-all, so to remove duplicates call create_index after union.
     *  @param r2  the second table (may be a Table or GTable)
     */
    override infix def union (r2: Table): GTable =
        if incompatible (r2) then return this

        val s = new GTable (s"${name}_u_${cntr.inc ()}", schema, domain, key)

        s.vertices ++= (
            if r2.isInstanceOf [GTable] then vertices ++ r2.asInstanceOf [GTable].vertices
            else vertices ++ r2.tuples.map (Vertex (_)))
        s
    end union

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute this graph-table MINUS (set difference) table r2 (this - r2).
     *  Check that the two tables are compatible.  If they are not, return the first table.
     *  @param r2  the second table (may be a Table or GTable)
     */
    override infix def minus (r2: Table): GTable =
        if incompatible (r2) then return this

        val s = new GTable (s"${name}_m_${cntr.inc ()}", schema, domain, key)

        for v <- vertices do
            if ! (r2 contains v.tuple) then s.vertices += v
        end for
        s
    end minus

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** INTERSECT this graph-table and r2.  Check that the two tables are compatible.
     *  If they are not, return the first table.
     *  @param r2  the second table (may be a Table or GTable)
     */
    override infix def intersect (r2: Table): GTable =
        if incompatible (r2) then return this

        val s = new GTable (s"${name}_i_${cntr.inc ()}", schema, domain, key)

        for v <- vertices do
            if r2 contains v.tuple then s.vertices += v
        end for
        s
    end intersect

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Expand and extract schema x from this graph-table and the referenced table.
     *  Acts as a lightweight join-project operator that only extracts attributes
     *  in schema x and does not create new edges.
     *  FIX:  for wildcard "*" extract all attributes
     *  @see https://arxiv.org/pdf/1806.07344.pdf
     *  @param x    the attributes to extract/collect from this and the referenced table.
     *  @param ref  the foreign key reference (edge-label, referenced table)
     */
    def expand (x: Schema, ref: (String, GTable)): GTable =
        val refTab = ref._2                                                 // edge-label, referenced table
//      val x1 = schema intersect x                                         // attributes from first table
        val x1 = meet (schema, x)                                           // attributes from first table
        val x2 = meet (refTab.schema, x)                                    // attributes from second table
        val newDom = pull (x1) ++ refTab.pull (x2)                          // corresponding domains
        debug ("expand", s"x1 = ${stringOf (x1)}, x2 = ${stringOf (x2)}, newDom = ${stringOf (newDom)}")

        val s = new GTable (s"${name}_x_${cntr.inc ()}", x, newDom, x)

        for u <- vertices do                                                // iterate over first table vertices
            val t1 = pull (u.tuple, x1)                                     // pull values from vertex u
            for v <- u.neighbors (ref) do                                   // iterate over second table vertices
                val t2 = refTab.pull (v.tuple, x2)                          // pull values from vertex v
                val w  = Vertex (t1 ++ t2)                                  // collect all attribute values
                s.vertices += w                                             // add vertex w to GTable s
        end for
        s
    end expand

    def expand (xs: String, ref: (String, GTable)): GTable = expand (strim (xs), ref)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the EDGE JOIN of this graph-table and the referenced table keeping
     *  concatenated vertices that are linked with edges having the given edge-label.
     *  Replaces fkey = pkey join and uses INDEX-FREE ADJACENCY.
     *  FIX - allow wild-card (e.g., *) for matching any edge-label
     *  @param ref  the foreign key reference (edge-label, referenced table, back edge-label)
     */
    def ejoin (ref: (String, GTable, String)): GTable =
        val (elab, refTab, elab2) = ref                                     // edge-label, referenced table, back edge-label
        val newKey = key ++ refTab.key                                      // FIX - many-to-?

        val s = new GTable (s"${name}_j_${cntr.inc ()}", schema ++ refTab.schema,
                            domain ++ refTab.domain, newKey)

        s.addEdgeType (elab, refTab)                                        // need to know if it is m-m or m-1
        s.addEdgeType (elab, refTab, false)

        for u <- vertices do                                                // iterate over first table vertices
            for v <- u.neighbors ((elab, refTab)) do                        // iterate over second table vertices
                debug ("ejoin", s"(u. v) = ($u, $v)")
                val w = Vertex (u.tuple ++ v.tuple)                         // collect all attribute values
                s.vertices += w                                             // add vertex w to GTable s
                updateEdges (s, w, u, v, elab, elab2)                       // add edges from u and v
                debug ("ejoin", s"add vertex w = $w)")
        end for
        s
    end ejoin

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an edge table containing all edges with the given edges label
     *  connecting vertices in this graph-table to vertices in the reference table.
     *  FIX - allow wild-card (e.g., *) for matching any edge-label
     *  @param ref  the foreign key reference (edge-label, referenced table)
     */
    def edgeTable (ref: (String, GTable)): GTable =
        val (elab, refTab) = ref                                            // edge-label, referenced table
        val key2   = refTab.key
        val newKey = key ++ refTab.key
        val newDom = pull (key) ++ refTab.pull (refTab.key)

        val s = new GTable (s"${name}_e_${cntr.inc ()}", newKey, newDom, newKey)

        for u <- vertices do
            for v <- u.neighbors ((elab, refTab)) do                        // iterate over second table vertices
                debug ("edgeTable", s"(u. v) = ($u, $v)")
                val w = Vertex (pull (u.tuple, key) ++ refTab.pull (v.tuple, key2))   // collect all attribute values
                s.vertices += w                                             // add vertex w to GTable s
                debug ("edgeTable", s"add vertex w = $w)")
        end for
        s
    end edgeTable

/*
        for u <- vertices; e <- u.edge do
            for v <- e._2 do                                                // add elab filter
                val w = Vertex (getPkey (u) ++ getPkey (v))
                s.vertices += w
//              debug ("edgeTable", s"add vertex w = $w)")
*/

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add edges to vertex w to include all outgoing edges of u and v, except
     *  those between u and v having edge-label elab.
     *  FIX - must also add edge types to GTable s
     *  @param s     the new graph-table 
     *  @param w     the new vertex 
     *  @param u     the original source vertex
     *  @param v     the original target vertex (may be null)
     *  @param elab  the edge-label
     */
    private def updateEdges (s: GTable, w: Vertex, u: Vertex, v: Vertex, elab: String, elab2: String): Unit =
        // add relevant edges from vertex u
        for uu <- u.edge do
            for u2 <- uu._2 if uu._1 != elab do                         // || u2 != v do
                debug ("updateEdges", s"u: $w -- ${uu._1} -> $u2")
                s.addE (w, uu._1, u2)
        end for

        if v != null then
            // add relevant edges from vertex v
            for vv <- v.edge do
                for v2 <- vv._2 if vv._1 != elab2 do                    // || v2 != u do
                    debug ("updateEdges", s"v: $w -- ${vv._1} -> $v2")
                    s.addE (w, vv._1, v2)
            end for
        end if
    end updateEdges

  // O U T P U T

    private val width_ = 18                                                 // default column width
    private val width  = Array.fill (domain.size) (width_)                  // width for each column

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset the width of column col to w.
     *  @param col  the column whose width is to be adjusted
     *  @param w    the new width (# chars) for column col
     */
    override def resetWidth (col: Int, w: Int): Unit = width(col) = w

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SHOW/print this graph-table, one vertex per row.
     *  @param rng  the range of vertices to show (e.g, 0 until 10), defaults to all
     */
    override def show (rng: Range = vertices.indices): Unit =
//      val edgeLabels = edgeType.keySet.toArray                            // all outgoing edges
//      val edgeLabels = (for (k, v) <- edgeType if v._2 yield k).toArray   // many-to-one cases
        val eschema    = schema // ++ edgeLabels
        val len        = width_ * (eschema.size + countX)

        println (s"\n>> GTable $name with ${rng.size} vertices, primary key = ${stringOf (key)}")
        println ("|-" + "-" * len + "-|")
        print ("| ")
        for j <- eschema.indices do
            val wj = if j < domain.size && domain(j) == 'X' then 2 * width_ else width_
            prt (eschema(j), wj)
        end for
        println (" |")

        println ("|-" + "-" * len + "-|")
        for i <- rng do
            print ("| ")
            val v_i = vertices(i)
            for j <- v_i.tuple.indices do
                val wj = if j < domain.size && domain(j) == 'X' then 2 * width_ else width_
                prt (v_i.tuple(j), wj)
            end for

/* Show edges separately
            val es = v_i.edge                                               // outgoing edges
            if es.nonEmpty then
                for elab <- edgeLabels do                                   // many-to-one cases
//                  debug ("show", s"es.head = ${es.head}, elab = $elab")
                    val x = es.getOrElse (elab, null)
                    if x != null then prt (x.head.tuple(0), width_)
                end for
            end if
*/
            println (" |")
        end for
        println ("|-" + "-" * len + "-|")
    end show

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** WRITE this graph-table into a JavaScript Object Notation (JSON) file.
     *  @param fileName  the file name of the data file
     *
    override def writeJSON (fileName: String = name + ".json"): Unit =
        val gson    = new Gson ()
        val jsonStr = gson.toJson (this)
        debug ("writeJSON", s"jsonStr = ${jsonStr.slice (0, min (jsonStr.size, 5000))}")
        val out = new PrintWriter (DATA_DIR + fileName)
        out.println (jsonStr)
        out.close ()
    end writeJSON
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the basic statistics for each column of this table.
     */
    override def stats: Table =
        val s = new Table (s"${name}_stats",
            Array ("column", "count", "countd", "min", "max", "sum", "avg"),
            Array ('S', 'I', 'I', 'S', 'S', 'D', 'D'), Array ("column"))

        val t = Bag [Tuple] ()
        for j <- colIndices do t += vertices(j).tuple
        for j <- colIndices do s.add (Table.stats (schema(j), col(j, t)))
        s
    end stats

end GTable


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `gTableTest` main function tests the `GTable` class with queries on the
 *  Bank database.
 *  > runMain scalation.database.table.gTableTest
 */
@main def gTableTest (): Unit =

    // Data Definition Language

    val customer = GTable ("customer", "cname, street, ccity", "S, S, S", "cname")
    val deposit  = GTable ("deposit", "accno, balance", "I, D", "accno")
    val branch   = GTable ("branch", "bname, assets, bcity", "S, D, S", "bname")
    val loan     = GTable ("loan", "loanno, amount", "I, D", "loanno")

    deposit.addEdgeType ("cname", customer)
    deposit.addEdgeType ("bname", branch)
    loan.addEdgeType ("cname", customer)
    loan.addEdgeType ("bname", branch)

    //--------------------------------------------------------------------------
    banner ("Populate Database")

    val v_Peter = customer.addV ("Peter", "Oak St",   "Bogart")
    val v_Paul  = customer.addV ("Paul",  "Elm St",   "Watkinsville")
    val v_Mary  = customer.addV ("Mary",  "Maple St", "Athens")
    customer.show ()

    val v_Alps     = branch.addV ("Alps",     20000000.0, "Athens")
    val v_Downtown = branch.addV ("Downtown", 30000000.0, "Athens")
    val v_Lake     = branch.addV ("Lake",     10000000.0, "Bogart")
    branch.show ()

    val v_11 = deposit.addV (11, 2000.0)
    val v_12 = deposit.addV (12, 1500.0)
    val v_13 = deposit.addV (13, 2500.0)
    val v_14 = deposit.addV (14, 2500.0)
    val v_15 = deposit.addV (15, 3000.0)
    val v_16 = deposit.addV (16, 1000.0)

    deposit.addE ("bname", Edge (v_11, v_Lake))
           .addE ("bname", Edge (v_12, v_Alps))
           .addE ("bname", Edge (v_13, v_Downtown))
           .addE ("bname", Edge (v_14, v_Lake))
           .addE ("bname", Edge (v_15, v_Alps))
           .addE ("bname", Edge (v_16, v_Downtown))

    deposit.addE ("cname", Edge (v_11, v_Peter))
           .addE ("cname", Edge (v_12, v_Paul))
           .addE ("cname", Edge (v_13, v_Paul))
           .addE ("cname", Edge (v_14, v_Paul))
           .addE ("cname", Edge (v_15, v_Mary))
           .addE ("cname", Edge (v_16, v_Mary))
    deposit.show ()

    val v_21 = loan.addV (21, 2200.0)
    val v_22 = loan.addV (22, 2100.0)
    val v_23 = loan.addV (23, 1500.0)
    val v_24 = loan.addV (24, 2500.0)
    val v_25 = loan.addV (25, 3000.0)
    val v_26 = loan.addV (26, 1000.0)

    loan.addE ("bname", Edge (v_21, v_Alps))
        .addE ("bname", Edge (v_22, v_Downtown))
        .addE ("bname", Edge (v_23, v_Alps))
        .addE ("bname", Edge (v_24, v_Downtown))
        .addE ("bname", Edge (v_25, v_Alps))
        .addE ("bname", Edge (v_26, v_Lake))

    loan.addE ("cname", Edge (v_21, v_Peter))
        .addE ("cname", Edge (v_22, v_Peter))
        .addE ("cname", Edge (v_23, v_Paul))
        .addE ("cname", Edge (v_24, v_Paul))
        .addE ("cname", Edge (v_25, v_Mary))
        .addE ("cname", Edge (v_26, v_Mary))
    loan.show ()

    //--------------------------------------------------------------------------
    banner ("Show Table Statistics")

/*
    customer.stats.show ()         // FIX - add support
    branch.stats.show ()
    deposit.stats.show ()
    loan.stats.show ()
*/

    //--------------------------------------------------------------------------
    banner ("Example Queries")

    banner ("live in Athens")
    val liveAthens = customer.σ ("ccity == 'Athens'").π ("cname") 
    liveAthens.show ()

    banner ("bank in Athens")
    val bankAthens = (deposit ⋈ (("bname", branch.σ ("bcity == 'Athens'")))) //.π ("cname")
    bankAthens.show ()

end gTableTest
 

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `gTableTest2` main function tests the `GTable` class with queries on the
 *  Student-Course-Professor database.
 *  > runMain scalation.database.table.gTableTest2
 */
@main def gTableTest2 (): Unit =

    // Data Definition Language

    val student   = GTable ("student",   "sid, sname, street, city, dept, level",
                                         "I, S, S, S, S, I", "sid")
    val professor = GTable ("professor", "pid, pname, street, city, dept",
                                         "I, S, S, S, S", "pid")
    val course    = GTable ("course",    "cid, cname, hours, dept",
                                         "I, X, I, S", "cid")

    student.addEdgeType ("cid", course, false)                // student has M courses
    course.addEdgeType ("sid", student, false)                // course has M students
    course.addEdgeType ("pid", professor)                     // course has 1 professor

    //--------------------------------------------------------------------------
    banner ("Populate Database")

    val v_Peter = student.addV (101, "Peter", "Oak St",   "Bogart",       "CS", 3)
    val v_Paul  = student.addV (102, "Paul",  "Elm St",   "Watkinsville", "CE", 4)
    val v_Mary  = student.addV (103, "Mary",  "Maple St", "Athens",       "CS", 4)

    val v_DrBill = professor.addV (104, "DrBill", "Plum St", "Athens",       "CS")
    val v_DrJohn = professor.addV (105, "DrJohn", "Pine St", "Watkinsville", "CE")

    val v_Database     = course.addV (4370, "Database Management", 4, "CS")
    val v_Architecture = course.addV (4720, "Comp. Architecture",  4, "CE")
    val v_Networks     = course.addV (4760, "Computer Networks",   4, "CS")

    student.add2E ("cid", Edge (v_Peter, v_Database),     "sid", course)
           .add2E ("cid", Edge (v_Peter, v_Architecture), "sid", course)
           .add2E ("cid", Edge (v_Paul,  v_Database),     "sid", course)
           .add2E ("cid", Edge (v_Paul,  v_Networks),     "sid", course)
           .add2E ("cid", Edge (v_Mary,  v_Networks),     "sid", course)
 
    course.addE ("pid", Edge (v_Database,     v_DrBill))
          .addE ("pid", Edge (v_Architecture, v_DrBill))
          .addE ("pid", Edge (v_Networks,     v_DrJohn))

    banner ("Show vertex-tables")

    student.show ()
    professor.show ()
    course.show ()

    banner ("Show edge-tables")

//  student.edgeTable (("*", course)).show ()
    student.edgeTable (("cid", course)).show ()
//  course.edgeTable (("pid", professor)).show ()      // FIX - crashes

    //--------------------------------------------------------------------------
    banner ("Example Queries")

    banner ("locations of students")
    val locs = student.project ("sname, city")
    locs.show ()

    banner ("living in Athens")
    val inAthens = student.select ("city == 'Athens'")
    inAthens.show ()

    banner ("not living in Athens")
    val notAthens = student minus inAthens
    notAthens.show ()

    banner ("student intersect inAthens")
    val inters = student intersect inAthens
    inters.show ()

    banner ("in-Athens union not-in-Athens")
    val unio = inAthens union notAthens
    unio.show ()    

    banner ("courses taken: course id")
    val taken_id = student.expand ("sname, cid", ("cid", course))
    taken_id.show ()

    banner ("courses taken: course name")
    val taken_nm = student.expand ("sname, cname", ("cid", course))
    taken_nm.show ()

    banner ("courses taken: course name via ejoin")
    val taken_ej = student.ejoin ("cid", course, "sid").project ("sname, cname")
    taken_ej.show ()

/*
    compare to equivalent for `Table` and `LTable`
    takes.join (("sid", student))
         .join (("cid", course))
         .project ("sname, cname")
*/

    banner ("student taught by")
    val taught_by = student.expand ("sname, cid", ("cid", course))
                           .expand ("sname, pname", ("pid", professor))
    taught_by.show ()

    banner ("student taught by via ejoin")
    val taught_by2 = student.ejoin ("cid", course, "sid")
                            .ejoin ("pid", professor, "cid")
                            .project ("sname, pname")
    taught_by2.show ()

/*
    compare to equivalent for `Table` and `LTable`
    takes.join (("sid", student))
         .join (("cid", course))
         .join (("pid", professor))
         .project ("sname, pname")
*/

end gTableTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `gTableTest3` main function tests the `GTable` object and class by loading
 *  a dataset from a file.  It loads the ScalaTion "covid_19" dataset/CSV file.
 *  - RELATIVE PATHS are from ScalaTion's DATA-DIR (@see Util.scala)
 *  - FULL PATHS are from the OS's root directory
 *  Defaults to RELATIVE PATHS; use `setFullPath` method to change.
 *  > runMain scalation.database.table.gTableTest3
 */
@main def gTableTest3 (): Unit =

    //--------------------------------------------------------------------------
    // Verify access to file contents, comment out readFile once verified.
    //--------------------------------------------------------------------------

    val fileName = "covid_19.csv"
    println (s"fileName = $fileName")
//  readFile (fileName)                                      // for RELATIVE PATHS
//  readFile (fileName, fullPath = true)                     // for FULL PATHS

    //--------------------------------------------------------------------------
    // Use sample row/tuple in the middle of the file that has full information.
    //--------------------------------------------------------------------------

    val data_str = """
12/29/2020,19658043,205972,184889.714,342639,3611,2372.857,1.04,27782,122664,106708,
253765556,1887683,1484784,0.134,7.5,4387280,4282967,31140,722024,333594,325788
"""

    //--------------------------------------------------------------------------
    // Use this to guess the data-types/domains.  See last step for making corrections.
    //--------------------------------------------------------------------------

    val domain = Table.tuple2type (strim (data_str))
    println (s"domain = ${stringOf (domain)}")

    //--------------------------------------------------------------------------
    // Data stored relative to the "scalation_2.0/data" directory, if not use full path.
    // Call the GTable.load method:
    //     def load (fileName: String, name: String, domain: Domain, key: String,
    //               pos_ : Array [Int] = null, sep: String = ","): GTable =
    //--------------------------------------------------------------------------

    val covid = GTable.load (fileName, "covid", domain, "date")
    covid.show (0 until 200)

    //--------------------------------------------------------------------------
    // If this fails due to incorrect domains, save the domain that was printed,
    // correct the domains that are incorrect, and try again.
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // for fullPath: GTable.setFullPath ()
    // for limit:    GTable.setLimit (200)
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // Serialize and output the data into a JSON file (covid.json) in DATA_DIR
    //--------------------------------------------------------------------------

//  covid.writeJSON ()

end gTableTest3

