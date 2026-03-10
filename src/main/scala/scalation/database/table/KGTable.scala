
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Sep  8 20:34:26 EDT 2022
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Knowledge-Graph Algebra (KGA) for Knowledge-Graph DBMS
 */

package scalation
package database
package table

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `KGTable` companion object provides factory methods for creating
 *  knowledge-graph-tables.
 *  Supported domains/data-types are 'D'ouble, 'I'nt, 'L'ong, 'S'tring, and 'T'imeNum.
 */
object KGTable:

    private val cntr  = Counter ()                                          // counter for generating unique names

//  private var useFullPath = false                                         // defaults to using relative file paths
//  private var limit       = -1                                            // limit on number of lines to read

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a knowledge-graph-table given convenient string specifications.
     *  @param name    the name of the knowledge-graph-table
     *  @param schema  the attributes for the knowledge-graph-table
     *  @param domain  the domains/data-types for attributes ('D', 'I', 'L', 'S', 'X', 'T')
     *  @param key     the attributes forming the primary key
     *  @param parent  the parent (super-type) table
     */
    def apply (name: String, schema: String, domain: String, key: String,
               parent: KGTable = null): KGTable =
        new KGTable (name, strim (schema), strim (domain).map (_.head), strim (key), parent)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a knowledge-graph-table from an existing graph-table.
     *  @param gt  the source graph-table 
     */
    def apply (gt: GTable): KGTable = 
        val s = new KGTable (gt.name, gt.schema, gt.domain, gt.key)
        s.vertices ++= gt.vertices
        s
    end apply

end KGTable

import KGTable.cntr

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `KGTable` class (knowledge-graph-table) stores graph-relational data and
 *  implements knowledge graph algebra operators.
 *  Supported domains/data-types are 'D'ouble, 'I'nt, 'L'ong, 'S'tring, and 'T'imeNum.
 *  @param name_    the name of the graph-table
 *  @param schema_  the attributes for the graph-table
 *  @param domain_  the domains/data-types for attributes ('D', 'I', 'L', 'S', 'X', 'T')
 *  @param key_     the attributes forming the primary key
 *  @param parent   the parent (super-type) table
 */
class KGTable (name_ : String, schema_ : Schema, domain_ : Domain, key_ : Schema,
               val parent: KGTable = null)
     extends GTable (name_,
                     if parent == null then schema_ else parent.schema ++ schema_,
                     if parent == null then domain_ else parent.domain ++ domain_,
                     if parent == null then key_    else parent.key)
        with TNode
        with Serializable:

    if parent != null then parent.add (this)                           // subtypes from `TNode`

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show the immediate supertypes and subtypes of this kg-table.
     */
    def showHier (): Unit =
        println (name)
        val supertype_name = if parent == null then "none" else parent.name
        println (s" + supertype = $supertype_name")
        for n <- subtypes do println (s" - subtype   = ${n.asInstanceOf [KGTable].name}")
    end showHier

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return all vertices in this table and all subtables down to the given number
     *  of levels.
     *  @param levels  the number of levels to descend in the type hierarchy 
     */
    def all (levels: Int = 100): KGTable =
        val s = new KGTable (s"${name}_a_${cntr.inc ()}", schema_, domain_, key)
        addSubVertices (0, levels, s, this)
        s
    end all

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Recursively add the vertices from this kg-table and those of its subtypes. 
     *  @param s       the new kg-table/node
     *  @param n       the current kg-table/node
     *  @param lev     the current level
     *  @param levels  the number of levels to descend in the type hierarchy 
     */
    private def addSubVertices (lev: Int, levels: Int, s: KGTable, n: KGTable): Unit = 
        if lev < levels then
            val np = KGTable (n.project (schema_))                    // projected n
            s.vertices ++= np.vertices
            for nn <- n.subtypes do
                addSubVertices (lev + 1, levels, s, nn.asInstanceOf [KGTable])
        end if
    end addSubVertices

end KGTable


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `kGTableTest` main function tests the `KGTable` class with queries on the
 *  Bank database.
 *  > runMain scalation.database.table.kGTableTest
 */
@main def kGTableTest (): Unit =

    // Data Definition Language

    val customer = KGTable ("customer", "cname, street, ccity", "S, S, S", "cname")
    val deposit  = KGTable ("deposit", "accno, balance", "I, D", "accno")
    val branch   = KGTable ("branch", "bname, assets, bcity", "S, D, S", "bname")
    val loan     = KGTable ("loan", "loanno, amount", "I, D", "loanno")

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

end kGTableTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `kGTableTest2` main function tests the `KGTable` class with queries on the
 *  Student-Course-Professor database.
 *  > runMain scalation.database.table.kGTableTest2
 */
@main def kGTableTest2 (): Unit =

    // Data Definition Language

    val person    = KGTable ("person",    "id, name, street, city",
                                          "I, S, S, S", "id")
    val student   = KGTable ("student",   "dept, level",
                                          "S, I", null, person)
    val professor = KGTable ("professor", "dept",
                                          "S", null, person)
    val course    = KGTable ("course",    "cid, cname, hours, dept",
                                          "I, X, I, S", "cid")

    student.addEdgeType ("cid", course, false)                // student has M courses
    course.addEdgeType ("id", student, false)                 // course has M students
    course.addEdgeType ("pid", professor)                     // course has 1 professor

    //--------------------------------------------------------------------------
    banner ("Populate Database")

    person.addV (91, "Joe", "Birch St", "Athens")
    person.addV (92, "Sue", "Ceder St", "Athens")

    val v_Peter  = student.addV (101, "Peter", "Oak St",   "Bogart",       "CS", 3)
    val v_Paul   = student.addV (102, "Paul",  "Elm St",   "Watkinsville", "CE", 4)
    val v_Mary   = student.addV (103, "Mary",  "Maple St", "Athens",       "CS", 4)

    val v_DrBill = professor.addV (104, "DrBill", "Plum St", "Athens",       "CS")
    val v_DrJohn = professor.addV (105, "DrJohn", "Pine St", "Watkinsville", "CE")

    val v_Database     = course.addV (4370, "Database Management", 4, "CS")
    val v_Architecture = course.addV (4720, "Comp. Architecture",  4, "CE")
    val v_Networks     = course.addV (4760, "Computer Networks",   4, "CS")

    student.add2E ("cid", Edge (v_Peter, v_Database),     "id", course)
           .add2E ("cid", Edge (v_Peter, v_Architecture), "id", course)
           .add2E ("cid", Edge (v_Paul, v_Database),      "id", course)
           .add2E ("cid", Edge (v_Paul, v_Networks),      "id", course)
           .add2E ("cid", Edge (v_Mary, v_Networks),      "id", course)

    course.addE ("pid", Edge (v_Database,     v_DrBill))
          .addE ("pid", Edge (v_Architecture, v_DrBill))
          .addE ("pid", Edge (v_Networks,     v_DrJohn))

    student.show ()
    professor.show ()
    course.show ()
    person.show ()
    person.all().show ()

    banner ("Show Immediate Type Hierarchy")

    person.showHier ()
    student.showHier ()
    professor.showHier ()
    course.showHier ()

    banner ("Show edge-tables")

//  student.edgeTable (("*", course)).show ()
    student.edgeTable (("cid", course)).show ()
//  course.edgeTable (("pid", professor)).show ()      // FIX - crashes

    //--------------------------------------------------------------------------
    banner ("Example Queries")

    banner ("locations of students")
    val locs = student.project ("name, city")
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

    banner ("all people living in Athens")
    val people_Athens = person.all ().select ("city == 'Athens'")
    people_Athens.show ()

    banner ("courses taken: course id")
    val taken_id = student.expand ("name, cid", ("cid", course))
    taken_id.show ()

    banner ("courses taken: course name")
    val taken_nm = student.expand ("name, cname", ("cid", course))
    taken_nm.show ()

    banner ("courses taken: course name via ejoin")
    val taken_ej = student.ejoin ("cid", course, "sid").project ("name, cname")
    taken_ej.show ()

/*
    compare to equivalent for `Table` and `LTable`
    takes.join (("sid", student))
         .join (("cid", course))
         .project ("sname, cname")
*/

    banner ("student taught by")
    val taught_by = student.expand ("name, cid", ("cid", course))
                           .expand ("name, pname", ("pid", professor))
    taught_by.show ()

    banner ("student taught by via ejoin")
    val taught_by2 = student.ejoin ("cid", course, "sid")
                            .ejoin ("pid", professor, "cid")
                            .project ("name, pname")
    taught_by2.show ()

/*
    compare to equivalent for `Table` and `LTable`
    takes.join (("sid", student))
         .join (("cid", course))
         .join (("pid", professor))
         .project ("sname, pname")
*/

end kGTableTest2

