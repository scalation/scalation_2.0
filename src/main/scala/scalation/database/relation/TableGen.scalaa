
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Nov 17 20:55:43 EST 2016
 *  @see     LICENSE (MIT style license file).
 *
 *  @naote   Columnar Relational Table Generation
 */

package scalation
package database
package relation

import scala.collection.mutable.ArrayBuffer
import scala.util.control.Breaks.{break, breakable}

import scalation.mathstat._
import scalation.random._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TableGen` object generates data for `Table` classes (e.g., `Relation`).
 */
object TableGen:

    private val debug = debugf ("TableGen", true)              // debug flag
    private val flaw  = flawf ("TableGen")                     // debug flag

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Populate the columns in columnar table 'table'.
     *  @param table     the table/relation to populate
     *  @param refTable  the tables that are referenced by 'table' via its foreign keys
     */
    def popTable (table: Relation, nRows: Int, refTables: ArrayBuffer [Relation] = null): Unit =
        val ranD  = RandomVecD (dim = nRows, max = 2 * nRows)
        val ranI  = RandomVecI (dim = nRows, max = 2 * nRows, unique = false)
        val ranS  = RandomVecS (dim = nRows, unique = false)
        val uranI = RandomVecI (dim = nRows, max = 2 * nRows)
        val uranS = RandomVecS (dim = nRows)

//      val n     = table.cols                                 // number of columns
        val cn    = table.schema                               // column names
        val dn    = table.domain                               // columns  domains
        val pk    = table.key                                  // primary key
        val pk_   = table.on(pk(0))                            // position of primary key - FIX only first part of key
        val fkey  = table.fKeys                                // sequence of foreign key specifications
        var cols  = Vector.empty [Vectr]

        for j <- cn.indices do                                 // for jth column
            val fk_i = find (cn(j), fkey)                      // which foreign key, if any
            val colj = if j == pk_ then genUnique (dn(j))      // generate values for jth column
                       else if fk_i >= 0 then genFkey (fk_i)
                       else genVal (dn(j))
            cols = cols :+ colj                                // FIX - need more efficient approach
        end for

        debug ("popTable", s"cols = $cols")
        table.col = cols

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /*  Generate values for column 'j' by extracting random values from
         *  the column in the referenced table specified by 'fk_j'.
         *  @param fk_i  the relevant foreign key constraint specification
         */
        def genFkey (fk_i: Int): Vectr =
            val refTName = fkey(fk_i)._2
            val refCol   = fkey(fk_i)._3
            var iref     = -1
            debug ("genFkey", s"refTName = $refTName")

            breakable {
                for k <- refTables.indices if refTName == refTables(k).name do
                    iref = k
                    break ()
                end for
            } // breakable

            if iref >= 0 then
                val refTable = refTables(iref) 
                val rCol     = refTable.col(refCol)
                val ranK     = RandomVecI (dim = nRows, max = refTable.rows - 1, unique = false)
                rCol.asInstanceOf [VectorI].set (ranK.igen)         // FIX - generalize
                rCol
            else
                flaw ("genFkey", s"reference table $refTName not matched")
                null
            end if
        end genFkey

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /*  Randomly Generate unique values for column 'j'.
         *  @param dn  the domain (datatype)
         */
        def genUnique (dn: Char): Vectr =
            dn match
            case 'D' => { flaw ("genUnique", "type `Double` should not be a primary key"); null }
            case 'I' => uranI.igen
            case 'L' => uranI.igen
            case 'S' => uranS.sgen
            case _  => { flaw ("genUnique", "type not supported"); null }
            end match
        end genUnique

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /*  Randomly Generate values for column 'j'.
         *  @param dn  the domain (datatype)
         */
        def genVal (dn: Char): Vectr =
            dn match
            case 'D' => ranD.gen
            case 'I' => ranI.igen
            case 'L' => ranI.igen
            case 'S' => ranS.sgen
            case _  => { flaw ("genVal", "type not supported"); null }
            end match
        end genVal

    end popTable 

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the column with name 'cn' in a foreign key constraint, returning
     *  which constraint is matched, -1 otherwise.
     *  @param cn     the name of the column
     *  @param fkeyC  the foreign key contraint specifications
     */
    def find (cn: String, fkeyC: ArrayBuffer [(String, String, Int)]): Int =
        if fkeyC == null then return -1
        fkeyC.indexWhere (_._1 == cn)                                    // index of matched constraint
    end find
/*
        for i <- fkeyC.indices if cn == fkeyC(i)._1 do return i          // matched ith contraint
        -1
*/

end TableGen


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tableGenTest` main function is used to test the `TableGen` object.
 *  Create unpopulated tables and use the table generator to populate their columns.
 *  > runMain scalation.database.relation.tableGenTest
 */
@main def tableGenTest (): Unit =

    val student = Relation ("student",
                            Array ("sid", "name", "address", "status"),
                            Array ('I', 'S', 'S', 'S'), Array ("sid"))
    TableGen.popTable (student, 40)
    student.show ()

    val professor = Relation ("professor",
                              Array ("pid", "name", "deptid"),
                              Array ('I', 'S', 'S'), Array ("pid"))
    TableGen.popTable (professor, 10)
    professor.show ()

    val course = Relation ("course",
                           Array ("cid", "deptid", "crsname", "descr"),
                           Array ('S', 'S', 'S', 'S'), Array ("cid"))
    TableGen.popTable (course, 20)
    course.show ()

    val teaching = Relation ("teaching",
                             Array ("tid", "cid", "semester", "pid"),
                             Array ('I', 'S', 'S', 'I'), Array ("tid"))
    teaching.fKeys = ArrayBuffer (("cid", "course", 0), ("pid", "professor", 0))
    TableGen.popTable (teaching, 50, ArrayBuffer (course, professor))
    teaching.show ()
    teaching.show_foreign_keys ()

    val transript = Relation ("transript",
                              Array ("trid", "sid", "trid", "grade"),
                              Array ('I', 'I', 'S', 'S', 'S'), Array ("trid"))
    transript.fKeys = ArrayBuffer (("sid", "student", 0), ("trid", "teaching", 0))
    TableGen.popTable (transript, 70, ArrayBuffer (student, teaching))
    transript.show ()
    transript.show_foreign_keys ()

end tableGenTest

