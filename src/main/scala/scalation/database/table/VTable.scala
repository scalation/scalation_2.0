
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Jul  8 23:10:18 EDT 2022
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Graph-Relational Algebra (GRA) for Graph-Relational DBMS
 */

package scalation
package database
package table 

import scala.collection.mutable.{ArrayBuffer => Bag, Map}
import scala.runtime.ScalaRunTime.stringOf

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Vertex_` class extends the notion of `Tuple` into values stored in the tuple
 *  part, along with foreign keys links captured as outgoing edges.
 *  @param tuple  the tuple part of a vertex
 */
case class Vertex_ (tuple: Tuple):

    val edge = Map [String, Bag [Vertex_]] ()                      // map edge-label -> { vertices }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the neighboring vertices reachable in one-hop by taking edges
     *  emanating from this vertex.
     */ 
    def neighbors: Bag [Vertex_] =
        val vs = Bag [Vertex_] ()
        for e <- edge do vs ++= e._2
        vs
    end neighbors

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the neighboring vertices reachable in one-hop by taking edge with
     *  edge-label elab emanating from this vertex.
     *  @param elab  the edge-label (wild-card "*" means take all edges)
     */ 
    def neighbors (elab: String): Bag [Vertex_] =
        if elab == "*" then neighbors else edge (elab)
    end neighbors

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the neighboring vertices reachable in one-hop by taking edge with
     *  edge-label elab emanating from this vertex that are in the reference table.
     *  @param ref  the edge-label and reference table
     */ 
    def neighbors (ref: (String, VTable)): Bag [Vertex_] =
        edge (ref._1) intersect ref._2.vertices
    end neighbors

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert the tuple within this vertex to a string.
     */
    override def toString: String = stringOf (tuple)

end Vertex_


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VTable` companion object provides factory methods for creating vertex-tables.
 *  Supported domains/data-types are 'D'ouble, 'I'nt, 'L'ong, 'S'tring, and 'T'imeNum.
 */
object VTable:

    private val debug = debugf ("VTable", true)                             // debug function
    private val flaw  = flawf ("VTable")                                    // flaw function
    private val cntr  = Counter ()                                          // counter for generating unique names

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a vertex-table given convenient string specifications.
     *  @param name    the name of the vertex-table
     *  @param schema  the attributes for the vertex-table
     *  @param domain  the domains/data-types for attributes ('D', 'I', 'L', 'S', 'X', 'T')
     *  @param key     the attributes forming the primary key
     */
    def apply (name: String, schema: String, domain: String, key: String): VTable =
        new VTable (name, strim (schema), strim (domain).map (_.head), strim (key))
    end apply

end VTable

import VTable.{cntr, debug, flaw}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VTable` class (vertex-table) stores graph-relational data and implements
 *  graph-relational algebra operators.
 *  Supported domains/data-types are 'D'ouble, 'I'nt, 'L'ong, 'S'tring, and 'T'imeNum.
 *  @param name_    the name of the vertex-table
 *  @param schema_  the attributes for the vertex-table
 *  @param domain_  the domains/data-types for attributes ('D', 'I', 'L', 'S', 'X', 'T')
 *  @param key_     the attributes forming the primary key
 */
case class VTable (name_ : String, schema_ : Schema, domain_ : Domain, key_ : Schema)
     extends Table (name_, schema_, domain_, key_)
        with Serializable:

    val vertices = Bag [Vertex_] ()                                         // collection of related vertices
    val edgeType = Map [String, (VTable, Boolean)] ()                       // collection of outgoing edge types

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether this vertex-table contains tuple u.
     *  @param t  the tuple to look for
     */
    override infix def contains (t: Tuple): Boolean = vertices.exists (_.tuple sameElements t)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the primary key for the given vertex.
     *  @param v  the given vertex
     */
    def getPkey (v: Vertex_): Tuple = pull (v.tuple, key)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add an edge type to this vertex-table (analog of a foreign key).
     *  @param elab  the edge-label for this edge type
     *  @param to      the vertex-table/vertex type for the target vertices
     *  @param unique  whether the target vertex is unique, e.g., many-to-one relationship
     */
    def addEdgeType (elab: String, to: VTable, unique: Boolean = true): Unit =   
        edgeType    += elab -> (to, unique)
        to.edgeType += elab -> (this, false)
    end addEdgeType

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add a vertex to this vertex-table (vertex contains a tuple).
     *  @param v  the vertex to add
     */
    def add (v: Vertex_): Vertex_ =
        if typeCheck (v.tuple) then vertices += v
        v
    end add

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create and add a vertex to this vertex-table.
     *  @param t  the tuple for creating a vertex to add
     */
    inline def addV (t: Tuple): Vertex_ = add (Vertex_ (t))

    inline def addV (x: ValueType*): Vertex_ = add (Vertex_ (x.toArray))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add an edge from vertex u to vertex v with edge-label elab.
     *  @param eLab  the edge-label, e.g., taughtBy
     *  @param u     the source vertex, course
     *  @param v     the target vertex, e.g., professor teaching the course
     */
    def addE (elab: String, u: Vertex_, v: Vertex_): VTable =
        val eType = edgeType
        if eType.keySet contains elab then
            val vset = u.edge.getOrElse (elab, null)

            if vset == null then u.edge += elab -> Bag (v)
            else if eType(elab)._2 then
                flaw ("addE", "attempt to link to multiple targets vertices when edge type is unique")
            else
                vset += v
        else 
            flaw ("addE", s"elab = $elab not an edge type for $name")
        debug (s"addE", s"$name: \t u.edge = ${u.edge}")
        this
    end addE

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add two edges from vertex u to vertex v and from v to u both with edge-label elab.
     *  @param elab   the edge-label, e.g., taughtBy (pid)
     *  @param u      the source vertex, course
     *  @param v      the target vertex, e.g., professor teaching the course
     *  @param elab2  the edge-label in the reverse direction, e.g., teaches (cid)
     *  @param vtab   the target vertex's vertex-table
     */
    def add2E (elab: String, u: Vertex_, v: Vertex_, elab2: String, vtab: VTable): VTable =
        addE (elab, u, v)
        vtab.addE (elab2, v, u)
        this
    end add2E

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add edges from vertex u to vertices vs with edge-label elab.
     *  @param elab  the edge-label, e.g., teaches (cid)
     *  @param u     the source vertex, e.g., professor
     *  @param vs    the target vertices, e.g., courses being taught
     */
    def addEs (elab: String, u: Vertex_, vs: Bag [Vertex_]): Unit =
        val eType = edgeType
        if eType.keySet contains elab then
            if eType(elab)._2 then
                flaw ("addEs", "attempt to link to multiple targets vertices when edge type is unique")
            else
                val vset = u.edge.getOrElse (elab, null)
                if vset == null then u.edge += elab -> vs
                else vset ++= vs
        else 
            flaw ("addEs", s"elab = $elab not an edge type for $name")
//      debug ("addEs", s"edge = $edge")
    end addEs

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** PROJECT the vertices in this vertex-table onto the given attribute names.
     *  @param x  the schema/attribute names to project onto
     */
    override def project (x: Schema): VTable =
        val newKey = if subset (key, x) then key else x

        val s = new VTable (s"${name}_p_${cntr.inc ()}", x, pull (x), newKey)

        s.vertices ++= (for v <- vertices yield Vertex_ (pull (v.tuple, x)))
        s
    end project

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SELECT the vertices in this vertex-table that satisfy the predicate p.
     *  @param p  the predicate (`Boolean` function) to be satisfied
     */
    override def select (predicate: Predicate): VTable =
        val s = new VTable (s"${name}_s_${cntr.inc ()}", schema, domain, key)

        s.vertices ++= (for v <- vertices if predicate (v.tuple) yield v)
        s
    end select

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SELECT the vertices in this vertex-table that satisfy the given simple (3 token)
     *  condition.
     *  @param condition  the simple condition string "a1 op a2" to be satisfied, where
     *                    a1 is attribute, op is comparison operator, a2 is attribute or value
     */
    override def select (condition: String): Table =
        val s = new VTable (s"${name}_s_${cntr.inc ()}", schema, domain, key)

        val (tok, twoAtrs) = parseCond (condition)
        val (a1, op, a2) = (tok(0), tok(1), tok(2))
        debug ("select", s"(a1, op, a2) = ($a1, $op, $a2)")

        s.vertices ++= (for t <- selectTups (a1, op, a2, twoAtrs, vertices.map (_.tuple)) yield Vertex_ (t))
        s
    end select

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** UNION this vertex-table and r2.  Check that the two tables are compatible.
     *  If they are not, return the first table.
     *  Caveat:  Assumes the key from the first table still works (@see create_index)
     *  Acts like union-all, so to remove duplicates call create_index after union.
     *  @param r2  the second table (may be a Table or VTable)
     */
    override infix def union (r2: Table): VTable =
        if incompatible (r2) then return this

        val s = new VTable (s"${name}_u_${cntr.inc ()}", schema, domain, key)

        s.vertices ++= (
        if r2.isInstanceOf [VTable] then vertices ++ r2.asInstanceOf [VTable].vertices
        else vertices ++ r2.tuples.map (Vertex_ (_)))
        s
    end union

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute this vertex-table MINUS (set difference) table r2 (this - r2).
     *  Check that the two tables are compatible.  If they are not, return the first table.
     *  @param r2  the second table (may be a Table or VTable)
     */
    override infix def minus (r2: Table): VTable =
        if incompatible (r2) then return this

        val s = new VTable (s"${name}_m_${cntr.inc ()}", schema, domain, key)

        for v <- vertices do
            if ! (r2 contains v.tuple) then s.vertices += v
        end for
        s
    end minus

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** INTERSECT this vertex-table and r2.  Check that the two tables are compatible.
     *  If they are not, return the first table.
     *  @param r2  the second table (may be a Table or VTable)
     */
    override infix def intersect (r2: Table): VTable =
        if incompatible (r2) then return this

        val s = new VTable (s"${name}_i_${cntr.inc ()}", schema, domain, key)

        for v <- vertices do
            if r2 contains v.tuple then s.vertices += v
        end for
        s
    end intersect

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** EXPAND from this to the referenced vertex-table and extract schema x from
     *  this vertex-table and the referenced table.  Acts as a lightweight join-project
     *  operator that only extracts attributes in schema x and does not create new edges.
     *  @param x    the attributes to extract/collect from this and the referenced table.
     *  @param ref  the foreign key reference (edge-label, referenced table)
     */
    def expand (x: Schema, ref: (String, VTable)): VTable =
        val refTab = ref._2                                                 // edge-label, referenced table
//      val x1 = schema intersect x                                         // attributes from first table
        val x1 = meet (schema, x)                                           // attributes from first table
        val x2 = meet (refTab.schema, x)                                    // attributes from second table
        val newDom = pull (x1) ++ refTab.pull (x2)                          // corresponding domains
        debug ("expand", s"x1 = ${stringOf (x1)}, x2 = ${stringOf (x2)}, newDom = ${stringOf (newDom)}")

        val s = new VTable (s"${name}_x_${cntr.inc ()}", x, newDom, x)

        for u <- vertices do                                                // iterate over first table vertices
            val t1 = pull (u.tuple, x1)                                     // pull values from vertex u
            for v <- u.neighbors (ref) do                                   // iterate over second table vertices
                val t2 = refTab.pull (v.tuple, x2)                          // pull values from vertex v
                val w  = Vertex_ (t1 ++ t2)                                 // collect all attribute values
                s.vertices += w                                             // add vertex w to VTable s
        end for
        s
    end expand

    def expand (xs: String, ref: (String, VTable)): VTable = expand (strim (xs), ref)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the EDGE JOIN of this vertex-table and the referenced table keeping
     *  concatenated vertices that are linked with edges having the given edge-label.
     *  Replaces fkey = pkey join and uses INDEX-FREE ADJACENCY.
     *  FIX - allow wild-card (e.g., *) for matching any edge-label
     *  @param ref  the foreign key reference (edge-label, referenced table, back edge-label)
     */
    def ejoin (ref: (String, Table, String)): VTable =
        val (elab, refTab, elab2) = ref                                     // edge-label, referenced table, back edge-label
        val newKey = key ++ refTab.key                                      // FIX - many-to-?
        debug ("ejoin", s"$name ejoin ($elab, ${refTab.name}, $elab2)")

        val s = new VTable (s"${name}_j_${cntr.inc ()}", schema ++ refTab.schema,
                            domain ++ refTab.domain, newKey)

        updateEdgeTypes (s, this, refTab, elab, elab2)                      // update edge-types in new vertex-table

        if refTab.isInstanceOf [VTable] then                                // join to VTable
            val rVTable = refTab.asInstanceOf [VTable]
            for u <- vertices do                                            // iterate over first table vertices
                for v <- u.neighbors ((elab, rVTable)) do                   // iterate over second table vertices
                    val w = Vertex_ (u.tuple ++ v.tuple)                    // collect all attribute values
                    s.vertices += w                                         // add vertex w to VTable s
                    updateEdges (s, w, this, u, rVTable, v)                 // add edges from u and v
            end for
        else                                                                // join to Table
            for u <- vertices do
                for v <- u.edge(elab) if refTab contains v.tuple do
                    val w = Vertex_ (u.tuple ++ v.tuple)                    // collect all attribute values
                    s.vertices += w                                         // add vertex w to VTable s
                    updateEdges (s, w, this, u, null, null)                 // add edges from u only (v is not a vertex)
            end for
        end if
        s
    end ejoin

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update edge-types in the new vertex-table from those in tab1 and tab2
     *  that are not between tab1 and tab2 having edge-label elab.
     *  @param s      the new vertex-table 
     *  @param tab1   the left vertex-table 
     *  @param tab2   the right vertex-table (may be null)
     *  @param elab   the edge-label
     *  @param elab2  the back edge-label
     */
    private def updateEdgeTypes (s: VTable, tab1: VTable, tab2: Table, elab: String, elab2: String): Unit =
        for (k, vl) <- tab1.edgeType if k != elab && k != elab2 do s.addEdgeType (k, vl._1, vl._2)
        if tab2 != null && tab2.isInstanceOf [VTable] then
            val vtab2 = tab2.asInstanceOf [VTable]
            for (k, vl) <- vtab2.edgeType if k != elab && k != elab2 do s.addEdgeType (k, vl._1, vl._2)
        debug ("updateEdgeTypes", s"s.edgeType = ${s.edgeType}")
    end updateEdgeTypes

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update edges from vertex w to include relevant outgoing edges of u and v.
     *  @param s     the new vertex-table 
     *  @param w     the new vertex 
     *  @param tab1  the left vertex-table 
     *  @param u     the original source vertex (in tab1)
     *  @param tab2  the right vertex-table (may be null)
     *  @param v     the original target vertex (may be null) (in tab2)
     */
    private def updateEdges (s: VTable, w: Vertex_, tab1: VTable, u: Vertex_, tab2: Table, v: Vertex_): Unit =
        for (k, vl) <- s.edgeType do
            debug ("updateEdges", s"for ($k, $vl) add edge(s)")
            val tab = vl._1
            if tab.name == tab1.name then s.addE (k, w, u)
            else if tab2 != null && tab.name == tab2.name then s.addE (k, w, v)
        end for
    end updateEdges

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an edge table containing all edges with the given edges label
     *  connecting vertices in this vertex-table to vertices in the reference table.
     *  FIX - allow wild-card (e.g., *) for matching any edge-label
     *  @param ref  the foreign key reference (edge-label, referenced table)
     */
    def edgeTable (ref: (String, Table)): VTable =
        val refTab = ref._2                                                 // edge-label, referenced table
        val newKey = key ++ refTab.key
        val newDom = pull (key) ++ refTab.pull (refTab.key)

        val s = new VTable (s"${name}_e_${cntr.inc ()}", newKey, newDom, newKey)

        for u <- vertices; e <- u.edge do
            for v <- e._2 do                                                // add elab filter
                val w = Vertex_ (getPkey (u) ++ getPkey (v))
                s.vertices += w
//              debug ("edgeTable", s"add vertex w = $w)")
        end for
        s
    end edgeTable

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
    /** SHOW/print this vertex-table, one vertex per row.
     *  @param rng  the range of vertices to show (e.g, 0 until 10), defaults to all
     */
    override def show (rng: Range = vertices.indices): Unit =
//      val edgeLabels = edgeType.keySet.toArray                            // all outgoing edges
        val edgeLabels = (for (k, v) <- edgeType if v._2 yield k).toArray   // many-to-one cases
        val eschema    = schema ++ edgeLabels
        val len        = width_ * (eschema.size + countX)

        println (s"\n>> VTable $name with with ${rng.size} vertices, primary key = ${stringOf (key)}")
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

            val es = v_i.edge                                               // outgoing edges
            if es.nonEmpty then
                for elab <- edgeLabels do                                   // many-to-one cases
//                  debug ("show", s"es.head = ${es.head}, elab = $elab")
                    val x = es.getOrElse (elab, null)
                    if x != null then prt (x.head.tuple(0), width_)
                end for
            end if
            println (" |")
        end for
        println ("|-" + "-" * len + "-|")
    end show

end VTable


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `vTableTest` main function tests the `VTable` class with queries on the
 *  Bank database.
 *  > runMain scalation.database.table.vTableTest
 */
@main def vTableTest (): Unit =

    // Data Definition Language

    val customer = VTable ("customer", "cname, street, ccity", "S, S, S", "cname")
    val deposit  = VTable ("deposit", "accno, balance", "I, D", "accno")
    val branch   = VTable ("branch", "bname, assets, bcity", "S, D, S", "bname")
    val loan     = VTable ("loan", "loanno, amount", "I, D", "loanno")

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

    deposit.addE ("bname", v_11, v_Lake)
           .addE ("bname", v_12, v_Alps)
           .addE ("bname", v_13, v_Downtown)
           .addE ("bname", v_14, v_Lake)
           .addE ("bname", v_15, v_Alps)
           .addE ("bname", v_16, v_Downtown)

    deposit.addE ("cname", v_11, v_Peter)
           .addE ("cname", v_12, v_Paul)
           .addE ("cname", v_13, v_Paul)
           .addE ("cname", v_14, v_Paul)
           .addE ("cname", v_15, v_Mary)
           .addE ("cname", v_16, v_Mary)
    deposit.show ()

    val v_21 = loan.addV (21, 2200.0)
    val v_22 = loan.addV (22, 2100.0)
    val v_23 = loan.addV (23, 1500.0)
    val v_24 = loan.addV (24, 2500.0)
    val v_25 = loan.addV (25, 3000.0)
    val v_26 = loan.addV (26, 1000.0)

    loan.addE ("bname", v_21, v_Alps)
        .addE ("bname", v_22, v_Downtown)
        .addE ("bname", v_23, v_Alps)
        .addE ("bname", v_24, v_Downtown)
        .addE ("bname", v_25, v_Alps)
        .addE ("bname", v_26, v_Lake)

    loan.addE ("cname", v_21, v_Peter)
        .addE ("cname", v_22, v_Peter)
        .addE ("cname", v_23, v_Paul)
        .addE ("cname", v_24, v_Paul)
        .addE ("cname", v_25, v_Mary)
        .addE ("cname", v_26, v_Mary)
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

end vTableTest
 

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `vTableTest2` main function tests the `VTable` class with queries on the
 *  Student-Course-Professor database.  Note, since `VTable` can handle many-to-many
 *  relationships, the 'takes' association table is no longer needed.
 *  > runMain scalation.database.table.vTableTest2
 */
@main def vTableTest2 (): Unit =

    // Data Definition Language

    val student   = VTable ("student",   "sid, sname, street, city, dept, level",
                                         "I, S, S, S, S, I", "sid")
    val professor = VTable ("professor", "pid, pname, street, city, dept",
                                         "I, S, S, S, S", "pid")
    val course    = VTable ("course",    "cid, cname, hours, dept",
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

    student.add2E ("cid", v_Peter, v_Database,     "sid", course)
           .add2E ("cid", v_Peter, v_Architecture, "sid", course)
           .add2E ("cid", v_Paul,  v_Database,     "sid", course)
           .add2E ("cid", v_Paul,  v_Networks,     "sid", course)
           .add2E ("cid", v_Mary,  v_Networks,     "sid", course)

    course.addE ("pid", v_Database,     v_DrBill)
          .addE ("pid", v_Architecture, v_DrBill)
          .addE ("pid", v_Networks,     v_DrJohn)

    student.show ()
    professor.show ()
    course.show ()

    banner ("Edge Table")
    student.edgeTable (("*", course)).show ()

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

    banner ("student taught by")
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

end vTableTest2

