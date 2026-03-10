
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Jun 25 20:41:18 EDT 2022
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Relational Table Generation
 */

package scalation
package database
package table

import scalation.mathstat._
import scalation.random._

import relation.Vectr

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TableGen` object generates data for instances of the `Table` class.
 */
object TableGen:

    private val flaw  = flawf ("TableGen")                            // flaw function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Randomly generate m tuples to populate the given table.
     *  @param table   the table to populate
     *  @param m       the number of tuples to generate
     *  @param stream  the random number stream
     */
    def popTable (table: Table, m: Int, stream: Int = 0): Unit =
        val n     = table.schema.size
        val ranD  = RandomVecD (dim = m, max = 2 * m, stream = stream)
        val ranI  = RandomVecI (dim = m, max = 2 * m, unique = false, stream = stream+1)
        val ranS  = RandomVecS (dim = m, unique = false, stream = stream+2)
        val uranI = RandomVecI (dim = m, max = 2 * m, stream = stream+3)  // unique random integers
        val uranS = RandomVecS (dim = m, stream = stream+4)               // unique random strings
        val pkey  = table.key
        val dmain = table.domain

        makeTuples ()

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Make tuples: (1) random unique values for primary key attributes,
         *  (2) copies of primary keys from fkt for foreign key attributes, or
         *  (3) random values based on domain for other attributes.
         *  Caveat: for a composite primary key (sid, cid) pulling primary keys from
         *          Student.sid and Course.cid may results in a duplicate primary key
         */
        def makeTuples (): Unit =
            val col = Array.ofDim [Vectr] (n)                         // generate column-by-column
            var j = 0
            while j < n do                                            // iterate through the attributes
                val atrj: String = table.schema(j)                    // j-th attribute name
                val fkt = table.linkTypes.getOrElse (atrj, null)      // which foreign key table or null

                if pkey contains atrj then                            // >> case PRIMARY KEY
                    col(j) = if fkt == null then genUnique (j)        // generate unique keys for attributes in pkey
                             else pullPkeys (fkt, j)                  // primary and foreign = copy pkey from fkt
                else if fkt != null then                              // >> case FOREIGN KEY
                    col(j) = pullPkeys (fkt, j)                       // foreign key = copy pkey from fkt
                else                                                  // >> case REGULAR ATTRIBUTE
                    col(j) = genValue (j)                             // generate value for a regular attribute
                j += 1                                                // increase attribute counter
            end while

            for i <- 0 until m do
                val t = Array.ofDim [ValueType] (n)                   // insert row-by-row
                for j <- 0 until n do t(j) = col(j)(i)                // build the next tuple/row
                table.add (t)                                         // insert into table
            end for
        end makeTuples

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Randomly pull m primary key values out of the foreign key table (fkt).
         *  Caveat:  Currently only works for non-composite foreign keys.
         *  @param fkt   the foreign key table (fkt) this_table references fkt
         *  @param strm  the random number stream to use (reduce redundancy)
         */
        def pullPkeys (fkt: Table, strm: Int): Vectr =
            val k     = fkt.rows                                      // number of rows in fkt
            val ranRw = RandomVecI (dim = m, max = k-1, min = 0, unique = false, stream = strm)
            val rows  = ranRw.igen                                    // randomly select m rows
            val jj    = fkt.on (fkt.key(0))                           // column for foreign key
            fkt.domain(jj) match
            case 'D' => VectorD (for i <- rows yield fkt.getPkey (i).key(0).asInstanceOf [Double])
            case 'I' => VectorI (for i <- rows yield fkt.getPkey (i).key(0).asInstanceOf [Int])
            case 'L' => VectorL (for i <- rows yield fkt.getPkey (i).key(0).asInstanceOf [Long])
            case 'S' => VectorS (for i <- rows yield fkt.getPkey (i).key(0).asInstanceOf [String])
            case _   => { flaw ("pullPkeys", "type not supported"); null }
        end pullPkeys

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /*  Randomly generate unique values for attribute/column j.
         *  @param j  which attribute/column
         */
        def genUnique (j: Int): Vectr =
            dmain(j) match
            case 'D' => { flaw ("genUnique", "type `Double` should not be a primary key"); null }
            case 'I' => uranI.igen
            case 'L' => uranI.igen
            case 'S' => uranS.sgen
            case _  => { flaw ("genUnique", "type not supported"); null }
        end genUnique

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /*  Randomly generate values for attribute/column 'j'.
         *  @param j  which attribute/column
         */
        def genValue (j: Int): Vectr =
            dmain(j) match
            case 'D' => ranD.gen
            case 'I' => ranI.igen
            case 'L' => ranI.igen
            case 'S' => ranS.sgen
            case _   => { flaw ("genValue", "type not supported"); null }
        end genValue

    end popTable 

end TableGen


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tableGenTest` main function is used to test the `TableGen` object.
 *  Create unpopulated tables and use the table generator to populate their columns.
 *  Tables:  student, professor, course, section, transcript
 *  > runMain scalation.database.table.tableGenTest
 */
@main def tableGenTest (): Unit =

    banner ("Create and populate the student table")
    val student = Table ("student", "sid, name, address, status", "I, S, S, S", "sid")
    TableGen.popTable (student, 40)
    student.show ()

    banner ("Create and populate the professor table")
    val professor = Table ("professor", "pid, name, deptid", "I, S, I", "pid")
    TableGen.popTable (professor, 10)
    professor.show ()

    banner ("Create and populate the course table")
    val course = Table ("course", "cid, deptid, crsname, descr", "I, I, S, S", "cid")
    TableGen.popTable (course, 20)
    course.show ()

    banner ("Create and populate the section table")
    val section = Table ("section", "crn, cid, semester, pid", "I, I, S, I", "crn")
    section.addLinkage ("cid", course)                             // teaching cid references course cid
    section.addLinkage ("pid", professor)                          // teaching pid references professor pid
    TableGen.popTable (section, 50)
    section.show ()
    section.show_foreign_keys ()

    banner ("Create and populate the transcript table")
    val transcript = Table ("transcript", "sid, crn, grade", "I, I, S", "sid, crn")
    transcript.addLinkage ("sid", student)                        // transcript sid references student sid
    transcript.addLinkage ("crn", section)                        // transcript crn references section crn
    TableGen.popTable (transcript, 70)
    transcript.show ()
    transcript.show_foreign_keys ()

    banner ("Check for duplicates: transcript has foreign keys inside primary keys")
    transcript.orderBy ("sid").show ()
    transcript.create_index ()
    println (s"After removing duplicates transcript has ${transcript.rows} rows")

end tableGenTest

