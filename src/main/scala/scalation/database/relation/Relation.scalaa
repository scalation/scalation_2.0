
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Yang Fan, Vinay Bingi, Santosh Uttam Bobade
 *  @version 2.0
 *  @date    Sun Aug 23 15:42:06 EDT 2015
 *  @see     LICENSE (MIT style license file).
 *
 *  @note   Implementation of (Columnar) Relational Algebra Operators
 *
 *  An implementation supporting columnar relational databases facilitating easy
 *  and rapid analytics.  The columns in a relation are vectors from the
 *  `scalation.mathstat` package.  Vectors and matrices may be readily extracted
 *  from a relation and feed into any of the numerous analytics techniques provided
 *  in `scalation.modeling`.  The implementation provides most of the columnar
 *  relational algebra operators given in the following paper:
 *  @see db.csail.mit.edu/projects/cstore/vldb.pdf
 *
 *  Some of the operators have unicode versions: @see `scalation.UnicodeTest`
 *
 *  Supports Time Series Databases (TSDB) via `TimeNum` domain/datatype and leftJoinApx
 *  rightJoinApx methods.
 */

package scalation
package database
package relation

import java.io._

import scala.collection.immutable.StringOps
import scala.collection.mutable.{ArrayBuffer => VEC, HashMap, IndexedSeq, Map}
//import scala.io.Source.fromInputStream
import scala.runtime.ScalaRunTime.stringOf
import scala.util.control.Breaks.{break, breakable}

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Combine two sequences of column names, keeping all names from cn1 and
 *  only those in cn2 that are not repeats (i.e., not already in cn1).
 *  @param cn1  the first sequence of column names
 *  @param cn2  the second sequence of column names
 */
def uniq_union (cn1: Schema, cn2: Schema): Schema =
    var cn3 = cn1
    for j <- cn2.indices if ! (cn3 contains cn2(j)) do cn3 = cn3 :+ cn2(j)
    cn3
end uniq_union

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Relation` companion object provides additional functions for the `Relation`
 *  class.
 *  FIX - apply methods - make compatible with RelationSQL
 */
object Relation:

    private val flaw   = flawf ("Relation")                                // flaw function
    private val ucount = Counter ()                                        // counter for making unique table names

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an unpopulated relation.
     *  @param name    the name of the relation
     *  @param schema  the names of columns
     *  @param domain  the string indicating domains for columns, e.g., Array ('S', 'D')
     *  @param key     the column number for the primary key
     */
    def apply (name: String, schema: Schema, domain: Domain, key: Schema): Relation =
        val n = schema.length
        new Relation (name, schema, domain, key, Vector.fill [Vectr] (n)(null))
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a relation from a sequence of row/tuples.  These rows must be converted
     *  to columns.
     *  @param name    the name of the relation
     *  @param schema  the names of columns
     *  @param domain  the string indicating domains for columns (e.g., 'SD' = 'String', 'Double')
     *  @param key     the column number for the primary key (< 0 => no primary key)
     *  @param row     the sequence of rows to be converted to columns for the columnar relation
     */
    def apply (name: String, schema: Schema, domain: Domain, key: Schema,
               row: VEC [Row]): Relation =
        val equivCol = Vector.fill [Vectr] (schema.length)(null)
        val r2 = new Relation (name, schema, domain, key, equivCol)
        for tuple <- row do r2.add (tuple)
        r2.materialize ()
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Read the relation with the given name into memory using serialization.
     *  @param name  the name of the relation to load
     */
    def apply (name: String): Relation =
        val ois = new ObjectInputStream (new FileInputStream (STORE_DIR + name + SER))
        val obj = ois.readObject ()
        ois.close ()
        val res = obj.asInstanceOf [Relation]
        res
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Read the relation with the given name into memory loading its columns
     *  with data from the CSV file named fileName.
     *  Note: "ln.split (eSep, -1)" will keep all values even if empty "one,,three" -> "one","",three"
     *  @param fileName  the file name of the data file
     *  @param name      the name of the relation
     *  @param schema   the names of columns
     *  @param domain    an optional string indicating domains for columns (e.g., 'SD' = 'String', 'Double')
     *  @param key       the column number for the primary key (< 0 => no primary key)
     *  @param skip      the number of lines in the CSV file to skip (e.g., header line(s))
     *  @param eSep      the element separation string/regex (e.g., "," ";" " +")
     */
    def load (fileName: String,
              name: String, schema: Schema, domain: Domain, key: Schema,
              skip: Int, eSep: String): Relation =
        var cnt    = skip
        val lines  = getFromURL_File (fileName)
        val newCol = Vector.fill [Vectr] (schema.length)(null)
        val r3     = new Relation (name, schema, domain, key, newCol)
        for ln <- lines do
            val buf = VEC.from (ln.split (eSep, -1))
            if cnt <= 0 then r3.add (r3.row (buf, domain)) else cnt -= 1
        end for
        r3.materialize ()
    end load

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Read the relation with the given name into memory loading its columns
     *  with data from the CSV file named fileName.  In this version, the column
     *  names are read from the first line of the file.
     *  @param fileName  the file name of the data file
     *  @param name      the name of the relation
     *  @param domain    an optional string indicating domains for columns (e.g., 'SD' = 'String', 'Double')
     *  @param key       the column number for the primary key (< 0 => no primary key)
     *  @param eSep      the element separation string/regex (e.g., "," ";" " +")
     *  @param cPos      the sequence of column positions in the input file to be used (null => select all)
     */
    def load (fileName: String,
              name: String, domain: Domain, key: Schema,
              eSep: String, cPos: VEC [Int]): Relation =
        val lines = getFromURL_File (fileName)
        var first = true
        var colBuffer: Array [VEC [String]] = null
        var colName:   VEC [String] = null

        if cPos == null then                                            // select all columns
            for ln <- lines do
                if first then
                    colName   = VEC.from (ln.split (eSep, -1).map (_.trim))
                    colBuffer = Array.fill (colName.length)(new VEC ())
                    first = false
                else
                    val values = ln.split (eSep, -1).map (_.trim)
                    for i <- colName.indices do colBuffer(i) += values(i)
                end if
            end for
        else                                                            // select cPos columns
            if domain.length != cPos.length then
                flaw ("apply", "cPos length should be same as domain length")
            end if
            for ln <- lines do
                if first then
                    val name  = ln.split (eSep, -1).map (_.trim)
                    colName = VEC [String] ()
                    colBuffer = Array.fill (cPos.length)(new VEC ())
                    for i <- colBuffer.indices do colName += name(cPos(i))
                    first = false
                else
                    val values = ln.split (eSep, -1).map (_.trim)
                    for i <- colName.indices do colBuffer(i) += values(cPos(i))
                end if
            end for
        end if
        new Relation (name, colName.toArray, domain, key, makeCol (colBuffer, domain))
    end load

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make the columns for the columnar table from data stored in colBuffer.
     *  @param colBuffer  the column buffer holding the data
     *  @param domain     the domains/datatypes for the columns
     */
    private def makeCol (colBuffer: Array [VEC [String]], domain: Domain): Vector [Vectr] =
        colBuffer.indices.map (i =>
            domain(i) match {
            case 'D' => VectorD.fromStrings (colBuffer(i))
            case 'I' => VectorI.fromStrings (colBuffer(i))
            case 'L' => VectorL.fromStrings (colBuffer(i))
            case 'S' => VectorS.fromStrings (colBuffer(i))
            case 'X' => VectorS.fromStrings (colBuffer(i))
            case 'T' => VectorT.fromStrings (colBuffer(i))
            case _   => flaw ("makeCol", s"domain type ${domain(i)} not supported")
                        null.asInstanceOf [Vectr]
        }).toVector.asInstanceOf [Vector [Vectr]]
    end makeCol

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Read the relation with the given name into memory loading its columns
     *  with data from the CSV file named fileName.  In this version, the column
     *  names are read from the first line of the file.  It uses col2 which is a
     *  temporary VEC, and maintains indices.
     *  @param fileName  the file name of the data file
     *  @param name      the name of the relation
     *  @param key       the column number for the primary key (< 0 => no primary key)
     *  @param domain    the string indicating domains for columns (e.g., 'SD' = 'String', 'Double')
     *  @param eSep      the element separation string/regex (e.g., "," ";" " +")
     */
    def load (fileName: String,
              name: String, domain: Domain, key: Schema,
              eSep: String = ","): Relation =
        var first         = true
        val lines         = getFromURL_File (fileName)
        var r3: Relation  = null
        var currentlineno = 0

        for ln <- lines do
            if first then
                val colName = Array.from (ln.split (eSep, -1))
                val newCol  = Vector.fill [Vectr] (colName.length)(null)
                r3    = new Relation (name, colName, domain, key, newCol)
                first = false
            else
                if currentlineno % 1000 == 0 then println (s"$currentlineno")
                val buf = VEC.from (ln.split (eSep, -1))
                r3.add (r3.row (buf, domain))
                currentlineno += 1
            end if
        end for
        r3.materialize ()
    end load

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Read the relation with the given name into memory loading its columns
     *  with data from the CSV file named fileName.  This version assumes
     *  defaults for eSep and skip of ("," and 0).
     *  @param fileName  the file name of the data file
     *  @param name      the name of the relation
     *  @param schema    the names of columns
     *  @param domain    the string indicating domains for columns, e.g., Array ('S', 'D')
     *  @param key       the column number for the primary key (< 0 => no primary key)
     */
    def load (fileName: String,
              name: String, schema: Schema, domain: Domain, key: Schema): Relation =
        val eSep   = ","
        val lines  = getFromURL_File (fileName)
        val newCol = Vector.fill [Vectr] (schema.length)(null)
        val r3     = new Relation (name, schema, domain, key, newCol)
        for ln <- lines do
            val buf = VEC.from (ln.split (eSep, -1))
            r3.add (r3.row (buf, domain))
        end for
        r3.materialize ()
    end load

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Read the relation with the given name into memory loading its columns
     *  with data from the .arff file named fileName.
     *  @param fileName  the file name of the data file
     *  @param domain    the string indicating domains for columns, e.g., Array ('S', 'D')
     *  @param key       the column number for the primary key (< 0 => no primary key)
     */
    def load (fileName: String, domain: Domain, key: Schema): Relation =
        val eSep  = "[, ]"
        val lines = getFromURL_File (fileName)
        var name: String = null
        var colBuffer: Array [VEC [String]] = null
        val colName   = VEC [String] ()
        var foundData = false

        for ln <- lines do
            if ln.indexOf ("%") != 0 then                          // skip comment
                if ln.indexOf ("@relation") == 0 then
                    name = ln.split (eSep, -1)(1)
                else if ln.indexOf ("@attribute") == 0 then
                    colName += ln.split (eSep, -1)(1)
                else if ln.indexOf ("@data") == 0 then
                    foundData = true
                    colBuffer = Array.ofDim (colName.length)
                    for i <- colBuffer.indices do colBuffer (i) = new VEC ()
                else if foundData then
                    val values = ln.split (eSep, -1)
                    values.indices.foreach (i => { colBuffer (i) += values (i) })
                end if
            end if
        end for
        new Relation (name, colName.toArray, domain, key,
                      colBuffer.indices.map (i => VectorS (colBuffer(i))).toVector)
    end load

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Read the relation with the given name into memory from a JSON file.
     *  @see https://github.com/FasterXML/jackson-databind
     *  @author Shubham Vasant Shingate
     *  FIX - does not work Scala 2.13, 3
     *  @param fileName  the file name of the JSON file
     *  @param name      the name of the relation to load
     *
    def load (fileName: String, name: String): Relation =
        import scala.jdk.CollectionConverters.asScalaIteratorConverter
//      import scala.collection.JavaConverters.asScalaIteratorConverter

        import com.fasterxml.jackson.databind.ObjectMapper
        type JSON_ELEM = java.util.LinkedHashMap [String, String]
        type JSON_TYPE = java.util.List [JSON_ELEM]

        var jsonList: java.util.List [JSON_ELEM] = null
        try
            val objMapper = new ObjectMapper ()
            val jsonStr   = fromInputStream (new FileInputStream (fileName)).mkString
            jsonList      = objMapper.readValue (jsonStr, classOf [JSON_TYPE])
        catch
            case e: FileNotFoundException => flaw ("apply", s"file $fileName not found")
            case e: IOException           => flaw ("apply", s"unable to read $fileName: $e")
        end try

        var splitStr = jsonList.get(0).toString
        var arrSize  = 0
        var flag     = true
        val colNames = Array [String] ()
        while arrSize != 1 do
            val subStr = splitStr.split ("=", 2)
            arrSize    = subStr.length
            if subStr(0).startsWith ("{") && arrSize != 1 then subStr(0) = subStr(0).substring(1)
            if ! flag && arrSize != 1 then subStr(0) = subStr(0).split (", ", 2)(1)
            if arrSize != 1 then
                colNames += subStr(0)
                splitStr  = subStr(1)
                flag      = false
            end if
        end while

        val newDomain = Array.fill ('S')(colNames.size)
        val rel = new Relation (name, colNames, newDomain, Array (colNames(0)),
                                Vector.fill [Vectr] (colNames.length)(null))
        for jsonData <- asScalaIteratorConverter (jsonList.iterator ()).asScala do
            val tuple  = VEC (asScalaIteratorConverter (jsonData.values().iterator()).asScala.toSeq :_*)
            rel.add (rel.row (tuple, null))
        end for
        rel.materialize ()
        rel
    end load
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a relation from the xy matrix of doubles.
     *  @param xy      the matrix containing the data
     *  @param name    the name of the relation
     *  @param schema  the names of columns
     *  @param key     the column number for the primary key (< 0 => no primary key)
     *  @param domain  the string indicating domains for columns, e.g., Array ('S', 'D')
     */
    def fromMatrix (xy: MatrixD,
                    name: String, schema: Schema, domain: Domain, key: Schema): Relation =
        val newCol = for j <- xy.indices2 yield xy(?, j)
        new Relation (name, schema, domain, key, newCol.toVector)
    end fromMatrix

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a relation from the x matrix of doubles and y vector of doubles
     *  or integers.
     *  @param x       the matrix containing the data
     *  @param y       the vector containing the data
     *  @param name    the name of the relation
     *  @param schema  the names of columns
     *  @param key     the column number for the primary key (< 0 => no primary key)
     *  @param domain  the string indicating domains for columns, e.g., Array ('S', 'D')
     */
    def fromMatrixV (x: MatrixD, y: VectorD,
                     name: String, schema: Schema, domain: Domain, key: Schema): Relation =
        val newCol = for j <- x.indices2 yield x(?, j)
        new Relation (name, schema, domain, key, newCol.toVector :+ y)
    end fromMatrixV

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the count (number of elements) of each of the columns of columnar
     *  relation r.
     *  @param r  the given relation
     */
    def count (r: Relation): IndexedSeq [Int] = VEC (r.col.map (_.size)*)

    def count1 (r: Relation, c: String): VectorI =
        val newcol_vals = VEC [Int] ()
        val old_col = r.column (c)
        old_col.foreach (x => { newcol_vals += r.groupMap (x).length })
        VectorI (newcol_vals)
    end count1

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the minimum of each of the columns of columnar relation r.
     *  @param r  the given relation
     */
    def min (r: Relation): Vector [Double] = for c <- r.col yield c.toDouble.min

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the maximum of each of the columns of columnar relation r.
     *  @param r  the given relation
     */
    def max (r: Relation): Vector [Double] = for c <- r.col yield c.toDouble.max

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the mean of each of the columns of columnar relation r.
     *  @param r  the given relation
     */
    def sum (r: Relation): Vector [Double] = for c <- r.col yield c.toDouble.sum

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the mean of each of the columns of columnar relation r.
     *  @param r  the given relation
     */
    def mean (r: Relation): Vector [Double] = for c <- r.col yield c.toDouble.mean

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the variance of each of the columns of columnar relation r.
     *  @param r  the given relation
     */
    def variance (r: Relation): Vector [Double] = for c <- r.col yield c.toDouble.variance

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get a Vectr of sum of the cName column for the r relation base on each group,
     *  the result will be the same size.
     *  @param r      the relation to operate on
     *  @param cName  sum on column "cName"
     *
    def sum (r: Relation, cName: String): Vectr =
        val cPos    = r.on.get(cName).get
        val domainc = r.domain(cPos)
        var columnlist: Vectr = null
        var count   = 0
        var pointer = 0
        var sumlist: Vectr = null
        for idx <- r.orderedIndex do
//          columnlist = Vectr.:+ (columnlist,r.index(idx)(cPos),r.domain,cPos)
            columnlist = Vectr.:+ (columnlist,r.index(idx)(cPos))
            if count +1 == r.grouplist(pointer) then
                val thisroundsum = Vec.sum(columnlist)
//              sumlist = Vectr.:+ (sumlist, thisroundsum, r.domain, cPos)
                sumlist = Vectr.:+ (sumlist, thisroundsum)
                columnlist = null
                pointer += 1
            end if
            count += 1
        end for
        sumlist
    end sum
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get a Vectr of max of the cName column for the r relation.
     *  @param r the relation you want to operate on
     *  @param cName  max on column "cName"
     *
    def max (r: Relation, cName: String): Vectr =
        val cPos    = r.on.get(cName).get
        val domainc = r.domain(cPos)
        var columnlist: Vectr = null
        var count   = 0
        var pointer = 0
        var maxlist: Vectr = null
        for idx <- r.orderedIndex do
//          columnlist = Vectr.:+ (columnlist,r.index(idx)(cPos),r.domain,cPos)
            columnlist = Vectr.:+ (columnlist,r.index(idx)(cPos))
            if count + 1 == r.grouplist(pointer) then
                val thisroundsum = Vec.max(columnlist)
//              maxlist = Vectr.:+ (maxlist, thisroundsum, r.domain, cPos)
                maxlist = Vectr.:+ (maxlist, thisroundsum)
                columnlist = null
                pointer += 1
            end if
            count += 1
        end for
        maxlist
    end max
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get a Vectr of min of the cName column for the r relation
     *  @param r      the relation you want to operate on
     *  @param cName  min on column "cName"
     *
    def min (r: Relation, cName: String): Vectr =
        val cPos    = r.on.get(cName).get
        val domainc = r.domain(cPos)
        var columnlist: Vectr = null
        var count   = 0
        var pointer = 0
        var minlist: Vectr = null
        for idx <- r.orderedIndex do
//          columnlist = Vectr.:+ (columnlist,r.index(idx)(cPos),r.domain,cPos)
            columnlist = Vectr.:+ (columnlist,r.index(idx)(cPos))
            if count + 1 == r.grouplist(pointer) then
                val thisroundsum = Vec.min (columnlist)
//              minlist = Vectr.:+ (minlist, thisroundsum, r.domain, cPos)
                minlist = Vectr.:+ (minlist, thisroundsum)
                columnlist = null
                pointer += 1
            end if
            count += 1
        end for
        minlist
    end min
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get a Vectr of average of the cName column for the r relation.
     *  @param r      the relation you want to operate on
     *  @param cName  average on column "cName"
     *
    def avg (r: Relation, cName: String): Vectr =
        val cPos    = r.on.get(cName).get
        val domainc = r.domain(cPos)
        var columnlist: Vectr = null
        var count   = 0
        var pointer = 0
        var avglist: Vectr = null
        for idx <- r.orderedIndex do
//          columnlist = Vectr.:+ (columnlist, r.index(idx)(cPos), r.domain, cPos)
            columnlist = Vectr.:+ (columnlist, r.index(idx)(cPos))
            if count + 1 == r.grouplist(pointer) then
                val thisroundsum = Vec.mean(columnlist)
//              avglist = Vectr.:+ (avglist, thisroundsum, r.domain, cPos)
                avglist = Vectr.:+ (avglist, thisroundsum)
                columnlist = null
                pointer += 1
            end if
            count += 1
        end for
        avglist
    end avg
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get a Vectr of count of the cName column for the r relation.
     *  @param r      the relation you want to operate on
     *  @param cName  the column name for the column to be counted
     *
    def count (r: Relation, cName: String): Vectr =
        val cPos = r.on.get (cName).get
        var countlist: Vectr = null
        var i = 0
        for p <- r.grouplist do
            val count = p - i
//          countlist = Vectr.:+ (countlist, count, r.domain, cPos)
            countlist = Vectr.:+ (countlist, count)
            i = p
        end for
        countlist
    end count
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get a Vectr of count of the cName column for the r relation.
     *  @param r      the relation you want to operate on
     *  @param cName  the column name for the column to be counted
     */
    def count (r: Relation, cName: String): Relation =
        val cPos = r.on(cName)
        r.groupBy (cName)
        
        val res = for (k, v) <- r.groupMap yield (k, v.size)
        println (s"count: res = $res")

        val newCName  = Array (cName, "count")
        val col0      = VectorS (res.keys.toVector.asInstanceOf [Vector [String]])  // FIX - generalize
        val col1      = VectorI (res.values.toVector)
        val newCol    = Vector (col0, col1).asInstanceOf [Vector [Vectr]]
        println (s"newCol = $newCol")
        val newKey    = Array (cName)
        val newDomain = Array (r.domain(cPos), 'I')
        new Relation (r.name + "_c_" + ucount.inc (), newCName, newDomain, newKey, newCol)
    end count

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** From function return cartesian product of all the relations.
     *  @param relations  the relations making up the from clause
     */
    def from (relations: Relation*): Relation =
        var result = relations(0)
        for i <- 1 until relations.size do result = result product relations(i)
        result
    end from

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test/execute the given query. 
     *  @param query   the query to execute
     *  @param result  the table resulting from the query
     */
    def test (query: String, result: Relation): Unit =
        banner (s"test: query = $query")
        result.show ()
    end test

end Relation

import Relation._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Relation` class stores and operates on vectors.  The vectors form the
 *  columns of the columnar relational datastore.  Columns may have any of the
 *  following types:
 *      D - `Double`   - `VectorD` -  64 bit double precision floating point number
 *      I - `Int`      - `VectorI` -  32 bit integer
 *      L - `Long`     - `VectorL` -  64 bit long integer
 *      S - `String`   - `VectorS` -  variable length numeric string
 *      X - `String`   - `VectorS` -  variable length numeric string (extended column width)
 *      T - `TimeNum`  - `VectorT` -  time numbers for date-time
 *  FIX - (1) don't allow (public) var
 *        (2) avoid unchecked or incomplete .asInstanceOf [T]
 *------------------------------------------------------------------------------
 *  @param name    the name of the relation
 *  @param schema  the names of columns
 *  @param domain  an optional string indicating domains for columns, e.g., Array ('S', 'D')
 *  @param key     the column number for the primary key (< 0 => no primary key)
 *  @param col     the Scala Vector of columns making up the columnar relation
 *  @param fKeys   an optional sequence of foreign keys - VEC (column name, ref table name, ref column position)
 */
class Relation (name: String, schema: Schema, domain: Domain, key: Schema,
                var col: Vector [Vectr] = null,
                var fKeys: VEC [(String, String, Int)] = null)
     extends Tabular [Relation] (name, schema, domain, key)
//   extends Table (name, schema, domain, key)
        with Serializable:

    private   val debug        = debugf ("Relation", true)                   // debug function
    private   val flaw         = flawf ("Relation")                          // flaw function
    private   val groupMap     = Map [ValueType, VEC [Int]] ()               // group rows by column value
//  private   var grouplist    = Vector [Int] ()                             // rows in group
    protected val index        = Map [ValueType, Row] ()                     // index that maps a key into row
    protected val indextoKey   = HashMap [Int, ValueType] ()                 // map index -> key
    private   var keytoIndex   = HashMap [ValueType, Int] ()                 // map key -> index
    protected var orderedIndex = Vector [ValueType] ()                       // re-ordering of the key column
    protected var hasIndex     = false                                       // whether this relation has an non-empty index

    if col == null then col = Vector.fill [Vectr] (schema.length)(null)
    if schema.length != col.length then flaw ("init", "incompatible sizes for 'schema' and 'col'")

    @transient
    private val col2 = Vector.fill (schema.size)(VEC [ValueType] ())   // efficient holding area for building columns

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the size in terms of number of rows in the relation.
     */
    def rows: Int = if col(0) == null then 0 else col(0).size

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a row by pulling values from an array of strings and converting
     *  elements to their appropriate types.
     *  @param sos   the sequence of strings holding the values
     *  @param _typ  the string of corresponding types, e.g., Array ('S', 'D', 'I')
     */
    @throws (classOf [Exception])
    def row (sos: VEC [String], dom: Domain): Row =
        var result: Vector [ValueType] = null
//      val typ = if _typ == null then "S" * sos.length else _typ    // missing => assume String
        try
            result = (for j <- sos.indices yield
                dom(j) match
                case 'D' => if sos(j).isEmpty then 0.0 else new StringOps (sos(j)).toDouble
                case 'I' => if sos(j).isEmpty then 0   else new StringOps (sos(j)).toInt
                case 'L' => if sos(j).isEmpty then 0L  else new StringOps (sos(j)).toLong
                case 'T' => if sos(j).isEmpty then TimeNum._0 else TimeNum (sos(j))
                case _   => sos(j)
                end match
            ).toVector.asInstanceOf [Vector [ValueType]]
        catch
            case ex: Exception =>
                println (s"row function throw exception, row is:\n $sos \ntuple length is: ${sos.size}")
                throw ex
        end try
        result
    end row

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether t and u are the same on column positions tp and up.
     *  @param t   the first row/tuple
     *  @param u   the second row/tuple
     *  @param tp  the column positions for row/tuple t
     *  @param up  the column positions for row/tuple u
     */
    def sameOn (t: Row, u: Row, tp: IndexedSeq [Int], up: IndexedSeq [Int]): Boolean =
        pull (t, tp) sameElements pull (u, up)
    end sameOn

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The create_index method helps, e.g., the popTable, methods to generate
     *  an index for the table.
     *  @param reset  if reset is true, use old index to build new index; otherwise, create new index
     */
    def create_index (reset: Boolean = false): Unit =
        if ! reset then
            for i <- 0 until rows do
                val jkey  = on(key(0))                                         // FIX - restricted to one column
                val mkey  = row(i)(jkey)                                       // key column is specified
                val tuple = row(i)
                index       += mkey -> tuple
                indextoKey  += i -> mkey
                keytoIndex  += mkey -> i
                orderedIndex = orderedIndex :+ mkey
            end for
        else                                                                    // use old index to build
            val newoderedIndex = new VEC [ValueType] ()
            val newkeytoIndex =  new HashMap [ValueType, Int] ()
            for i <- orderedIndex.indices do
                val mkey       = orderedIndex(i)
                val tuple      = row(keytoIndex(mkey))
                index         += mkey -> tuple
                newkeytoIndex += mkey -> i
                newoderedIndex.update (newoderedIndex.length, mkey)
            end for
            orderedIndex = newoderedIndex.toVector                              // map old keytoIndex to rowIndex to
            keytoIndex   = newkeytoIndex
        end if
        hasIndex = true
    end create_index

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** DROP the primary INDEX that maps the primary key to the tuple containing it.
     *  FIX - clear other indices
     */
    def drop_index (): Unit =
        index.clear ()
        hasIndex = false
    end drop_index

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return all of the columns in the relation.
     */
    def columns: Vector [Vectr] = col

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the column in the relation with column name cName.
     *  @param cName  column name used to retrieve the column vector
     */
    def column (cName: String): Vectr = col(on (cName))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a row by pulling values from all columns at position i.
     *  @param i  the i'th position
     */
    def row (i: Int): Row = for c <- col yield c(i)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the i-th primary key.
     *  @param i  the index in the tuples/row index
     */
    def getPkey (i: Int): KeyType = new KeyType (pull (row(i), key))
//  def getPkey (i: Int): KeyType = new KeyType (Table.project (row(i), key.map (on(_))))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add LINKAGE for this tabale to refTab by adding a FOREIGN KEY CONSTRAINT
     *  x to this table specifying the foreign key attribute fkey and the table
     *  it references refTab.
     *  Caveat:  a foreign key may not be composite.
     *  @param fkey    the foreign key attribute
     *  @param refTab  the table being referenced (to its primary key)
     */
    def addLinkage (fkey: String, refTab: Relation): Unit = ???

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check that all the foreign keys values in tuple t satisfy their
     *  REFERENTIAL INTEGRITY CONSTRAINTS.
     *  @param t  the tuple being checked for referential integrity
     */
    def referenceCheck (t: Tuple): Boolean = ???

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether this relation contains a row matching the given tuple.
     *  @param tuple  an aggregation of columns values (potential row)
     */
    infix def contains (tuple: Row): Boolean =
        var found = false
        breakable {
            for i <- 0 until rows if row(i) sameElements tuple do
                found = true
                break ()
            end for
        } // breakable
        found
    end contains

    def contains (u: Tuple): Boolean = ???                 // not to be implemented

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the table restricted to the given range of rows.
     *  @param rng  the given range of rows
     */
    def apply (rng: Range): Relation = ???

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the table restricted to the given collection of rows.
     *  @param pos  the given collection of rows
     */
    def apply (pos: collection.immutable.IndexedSeq [Int]): Relation =
        val newCol = (for j <- col.indices yield col(j).at(pos)).asInstanceOf [Vector [Vectr]]
        new Relation (name + "_s_" + ucount.inc (), schema, domain, key, newCol)
    end apply

    // R E L A T I O N   A L G E B R A   O P E R A T O R S

    // ================================================================== RENAME

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Rename this table, returning a shallow copy of this table.
     *  @param newName  the new name for the table.
     */
    def rename (newName: String): Relation =
        new Relation (newName, schema, domain, key, col, fKeys)
    end rename

    // ================================================================= PROJECT

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Project onto the columns with the given column names.
     *  @param cName  the names of the columns to project onto
     */
    def project (x: Schema): Relation = project (Array (x.map (on (_))*))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Project onto the columns with the given column positions.
     *  @param cPos  the positions of the columns to project onto
     */
    def project (cPos: IndexedSeq [Int]): Relation =
        val newCName  = for i <- cPos yield schema(i)
        val newCol    = cPos.map (col(_)).toVector
        val newKey    = newCName.asInstanceOf [Schema]
        val newDomain = pull (cPos)
        new Relation (name + "_p_" + ucount.inc (), newCName.asInstanceOf [Schema], newDomain, newKey, newCol)
    end project

    // ========================================================== PROJECT-SELECT

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Select elements from column cName in this relation that satisfy the
     *  predicate p and project onto that column.
     *  @param cName  the name of the column used for selection
     *  @param p      the predicate (`Boolean` function) to be satisfied
     */
    def selproject (cName: String, p: APredicate): Relation =
        val nu     = getMeta (cName)
        val newCol = Vector (col (nu._1).filter (p)).asInstanceOf [Vector [Vectr]]
        new Relation (name + "_s_" + ucount.inc (), nu._2, nu._3, nu._4, newCol)
    end selproject

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get meta-data about the column with name cName.
     *  @param cName  the name of the column
     */
    private def getMeta (cName: String): (Int, Schema, Domain, Schema) =
        val cn = on (cName)                                                 // column position
        (cn, Array (cName), pull (IndexedSeq (cn)), Array (cName))
    end getMeta

    // ================================================================== SELECT

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Select elements from column cName in this relation that satisfy the atomic
     *  predicate apred.
     *  @param cName  the name of the column used for selection
     *  @param apred  the predicate (`Boolean` function) to be satisfied
     */
    def select (cName: String, apred: APredicate): Relation =
        val colc = col (on(cName))
        apply (colc.filterPos (apred))
    end select

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SELECT the tuples in this table that satisfy the predicate.
     *  @param predicate  the predicate (`Boolean` function) to be satisfied
     */
    def select (predicate: Predicate): Relation = ???

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SELECT the tuples in this table that satisfy the given simple (3 token) condition.
     *  @param condition  the simple condition string "a1 op a2" to be satisfied, where
     *                    a1 is attribute, op is comparison operator, a2 is attribute or value
     */
    def select (condition: String): Relation = ???

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SELECT via the INDEX the tuple with the given primary key value pkey.
     *  Returns an empty table if the primary index has not been created.
     *  @param pkey  the primary key value
     */
    def select (pkey: KeyType): Relation = ???

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the relation whose rows are equal to value in the column with the given name. 
     *  @param cv  the (column-name, value) pair, e.g., ("time", 5.00)
     */
    def == (cv: (String, ValueType)): Relation = select (cv._1, (x: ValueType) => x == cv._2)
    def != (cv: (String, ValueType)): Relation = select (cv._1, (x: ValueType) => x != cv._2)
    def <  (cv: (String, ValueType)): Relation = select (cv._1, (x: ValueType) => x < cv._2)
    def <= (cv: (String, ValueType)): Relation = select (cv._1, (x: ValueType) => x <= cv._2)
    def >  (cv: (String, ValueType)): Relation = select (cv._1, (x: ValueType) => x > cv._2)
    def >= (cv: (String, ValueType)): Relation = select (cv._1, (x: ValueType) => x >= cv._2)

    // =========================================================== SET OPERATORS

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Union all (dups allowed) this relation and r2.  Check that the two relations
     *  are compatible.  If they are not, return the first this relation.
     *  @param r2  the second relation
     */
    infix def unionAll (r2: Relation): Relation =
        if incompatible (r2) then return this                                   // take only this relation

        val newCol = (for j <- col.indices yield col(j) ++ r2.columns(j)).toVector.asInstanceOf [Vector [Vectr]]
        new Relation (name + "_u_" + ucount.inc (), schema, domain, schema, newCol)
    end unionAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Union this relation and r2.  Check that the two relations are compatible.
     *  If they are not, return the first this relation.
     *  @param r2  the second relation
     */
    infix def union (r2: Relation): Relation =
        if incompatible (r2) then return this                                   // take only this relation
        val newCol  = (for j <- col.indices yield (col(j) ++ r2.columns(j)).asInstanceOf [Vectr])
        val newCols = newCol.transpose.distinct.transpose

        val ncl = VEC [Vectr] ()
        for cl <- newCols do
            val first = cl(0)
            first match
            case _: Double  => val rs = VectorD (for j <- cl yield j.asInstanceOf [Double]);  ncl += rs
            case _: Int     => val rs = VectorI (for j <- cl yield j.asInstanceOf [Int]);     ncl += rs
            case _: Long    => val rs = VectorL (for j <- cl yield j.asInstanceOf [Long]);    ncl += rs
            case _: String  => val rs = VectorS (for j <- cl yield j.asInstanceOf [String]);  ncl += rs
            case _: TimeNum => val rs = VectorT (for j <- cl yield j.asInstanceOf [TimeNum]); ncl += rs
            end match
        end for
        new Relation (name + "_u_" + ucount.inc (), schema, domain, schema, ncl.toVector)
    end union

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Intersect this relation and r2.  Check that the two relations are compatible.
     *  Slower and only to be used if there is no index.
     *  @param r2  the second relation
     */
    infix def intersect (r2: Relation): Relation =
        if hasIndex && r2.hasIndex then return intersect2 (r2)
        if incompatible (r2) then return null

        val newCol = Vector.fill [Vectr] (schema.length)(null)
        val r3 = new Relation (name + "_u_" + ucount.inc (), schema, domain, key, newCol.toVector)
        for i <- 0 until rows if r2 contains row(i) do r3.add (row(i))
        r3.materialize ()
    end intersect

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Intersect this relation and r2.  Check that the two relations are compatible.
     *  Use index to finish intersect operation.
     *  @param r2  the second relation
     */
    infix def intersect2 (r2: Relation): Relation =
        if incompatible (r2) then return null

        val newCol = Vector.fill [Vectr] (schema.length) (null)
        val r3     = new Relation (name + "_u_" + ucount.inc (), schema, domain, key, newCol)

        for i <- orderedIndex.indices do
            if r2.keytoIndex isDefinedAt orderedIndex(i) then
                if row(i) sameElements r2.row(r2.keytoIndex (orderedIndex(i))) then r3.add_ni (row(i))
            end if
        end for
        r3.materialize ()
    end intersect2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Take the difference of this relation and r2 (this - r2).  Check that
     *  the two relations are compatible.
     *  @param r2  the second relation
     */
    infix def minus (r2: Relation): Relation =
        if incompatible (r2) then return null

        val newCol = Vector.fill [Vectr] (schema.length)(null)
        val r3 = new Relation (name + "_m_" + ucount.inc (), schema, domain, key, newCol)
        for i <- 0 until rows if ! (r2 contains row(i)) do r3.add (row(i))
        r3.materialize ()
    end  minus

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Take the difference of this relation and r2 (this - r2).  Check that
     *  the two relations are compatible.  Indexed based minus.
     *  @param r2  the second relation
     */
    infix def minus2 (r2: Relation): Relation =
        if incompatible (r2) then return null

        val newCol = Vector.fill [Vectr] (schema.length)(null)
        val r3 = new Relation (name + "_m_" + ucount.inc (), schema, domain, key, newCol)
        for i <- orderedIndex.indices do
            if r2.keytoIndex isDefinedAt orderedIndex(i) then
                if ! (row(i) sameElements r2.row(r2.keytoIndex (orderedIndex(i)))) then r3.add_ni (row(i))
            else
                r3.add_ni (row(i))
            end if
        end for
        r3.materialize ()
    end minus2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether any rows/tuples exist in this relation.
     */
    def exists: Boolean = rows > 0

    // ================================================================= PRODUCT

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Cartesian product of this relation and r2 (this × r2).
     *  @param r2  the second relation
     */
    infix def product (r2: Relation): Relation =
        val ncols     = cols + r2.cols
        val newCName  = disambiguate (schema, r2.schema)
        val newCol    = Vector.fill [Vectr] (ncols) (null)
        val newKey    = key ++ r2.key
        val newDomain = domain ++ r2.domain
        val r3 = new Relation (name + "_j_" + ucount.inc (), newCName, newDomain, newKey, newCol)

        for i <- 0 until rows do
            val t = row(i)
            for j <- 0 until r2.rows do r3.add (t ++ r2.row(j))
        end for
        r3.materialize ()
    end product

    // ==================================================================== JOIN

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Join this relation and r2 by performing an "equi-join".  Rows from both
     *  relations are compared requiring cName1 values to equal cName2 values.
     *  Disambiguate column names by appending "2" to the end of any duplicate column name.
     *  @param cName1  the join column names of this relation (e.g., the Foreign Key)
     *  @param cName2  the join column names of relation r2 (e.g., the Primary Key)
     *  @param r2      the rhs relation in the join operation
     */
    def join (cName1: Schema, cName2: Schema, r2: Relation): Relation =
        val ncols = cols + r2.cols
        val cp1   = VEC.from (cName1.map (on (_)))                                      // get column positions in this
        val cp2   = VEC.from (cName2.map (r2.on (_)))                               // get column positions in r2
        if cp1.length != cp2.length then flaw ("join", "incompatible sizes on match columns")

        val newCName  = disambiguate (schema, r2.schema)
        val newCol    = Vector.fill [Vectr] (ncols) (null)
        val newKey    = key                                                      // FIX
        val newDomain = domain ++ r2.domain
        val r3 = new Relation (name + "_j_" + ucount.inc (), newCName, newDomain, newKey, newCol)

        for i <- 0 until rows do
            val t = row(i)
            for j <- 0 until r2.rows do
                val u = r2.row(j)
                if sameOn (t, u, cp1, cp2) then r3.add (t ++ u)
            end for
        end for
        r3.materialize ()
    end join

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Join this relation and r2 by performing an "equi-join", use index to join
     *  @param cName1  the join column names of this relation (e.g., the Foreign Key)
     *  @param cName2  the join column names of relation r2 (e.g., the Primary Key)
     *  @param r2      the rhs relation in the join operation
     */
    def joinindex (cName1: Schema, cName2: Schema, r2: Relation): Relation =
        val ncols = cols + r2.cols
        val cp1   = cName1.map (on (_))                                      // get column positions in this
        val cp2   = cName2.map (r2.on (_))                                   // get column positions in r2
        if cp1.length != cp2.length then flaw ("join", "incompatible sizes on match columns")

        val newCName = disambiguate (schema, r2.schema)
        val newCol   = Vector.fill [Vectr] (ncols)(null)
        val newKey   = if r2.key == cp2 then key                             // foreign key in this relation
                       else if key == cp1 then r2.key                        // foreign key in r2 table
                       else key ++ r2.key                                    // composite key

        val newDomain = domain ++ r2.domain
        val r3 = new Relation (name + "_j_" + ucount.inc (), newCName, newDomain, newKey, newCol)

        if cp1.size == 1 && cp2.size == 1 then
            if key == cp1 && r2.key == cp2 then
                for k <- orderedIndex do
                    val t = index(k)
                    val u = r2.index.getOrElse (k, null)
                    if u != null then r3.add_ni (t ++ u)
                end for
            else if key == cp1 then
                for idx <- r2.orderedIndex do
                    val u = r2.index(idx)
                    val t = index.getOrElse ((u(cp2(0))), null)
                    if t != null then r3.add_ni (t ++ u)
                    r3.add_ni(t ++ u)
                end for
            else if r2.key == cp2 then
                for idx <- orderedIndex do
                    val t = index(idx)
                    val u = r2.index.getOrElse ((t(cp1(0))), null)
                    if u != null then r3.add_ni (t ++ u)
                end for
            end if
        end if
        r3.materialize ()
    end joinindex

    // FIX - rewrite above method to fit this signature

    def join (ref: (String, Relation)): Relation = ???

    def _join (ref: (String, Relation)): Relation = ???

    def _join_ (ref: (String, Relation)): Relation = ???

    def _join (r2: Relation): Relation = ???

    def _join_ (r2: Relation): Relation = ???

    def join_ (r2: Relation): Relation = ???

    def limit (n: Int): Relation = ???

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Join this relation and r2 by performing a "natural-join".  Rows from both
     *  relations are compared requiring cName values to be equal.
     *  @param r2     the rhs relation in the join operation
     */
    infix def join (r2: Relation): Relation =
        val cName = schema intersect r2.schema
        val ncols = cols + r2.cols - cName.length
        val cp1   = VEC.from (cName.map (on (_)))                                // get column positions in this
        val cp2   = VEC.from (cName.map (r2.on (_)))                             // get column positions in r2
        var newDomain2 = r2.domain
        for i <- cp1.length - 1 to 0 by -1 do
            val (cp1_i, cp2_i) = (cp1(i), cp2(i))
            if domain(cp1_i) != r2.domain(cp2_i) then flaw ("join", s"column types do not match: $cp1, $cp2")
            newDomain2 = removeAt (newDomain2, cp2_i, 1)
        end for
        val cp3 = r2.schema.map (r2.on (_)) diff cp2                             // r2 specific columns

        val newCName  = uniq_union (schema, r2.schema)
        val newCol    = Vector.fill [Vectr] (ncols) (null)
        val newKey    = key                                                      // FIX
        val newDomain = domain ++ newDomain2
        val r3 = new Relation (name + "_j_" + ucount.inc (), newCName, newDomain, newKey, newCol)

        for i <- 0 until rows do
            val t = row(i)
            for j <- 0 until r2.rows do
                val u = r2.row(j)
                if sameOn (t, u, cp1, cp2) then { val u3 = pull (u, cp3); r3.add (t ++ u3) }
            end for
        end for
        r3.materialize ()
    end join

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The theta join, handle the predicates in where are connect by "and" (where a....and b....).
     *  @param r2  the second relation
     *  @param p0  the first theta join predicate (r1 cName, r2 cName, predicate to compare these two column)
     *  @param p   the rest of theta join predicates (r1 cName, r2 cName, predicates to compare these two column)
     */
    def join (r2: Relation, p0: APredicate2, p: APredicate2*): Relation =
        val ncols     = cols + r2.cols
        val newCName  = disambiguate (schema, r2.schema)
        val newCol    = Vector.fill [Vectr] (ncols) (null)
        val newKey    = key                                                      // FIX
        val newDomain = domain ++ r2.domain
        val r3        = new Relation (name + "_j_" + ucount.inc (), newCName, newDomain, newKey, newCol)

        var resultlist = IndexedSeq [(Int, Int)] ()
        for i <- 0 to p.size do
            var result = IndexedSeq [(Int, Int)] ()
            val p_i = if i == 0 then p0 else p(i-1)
            val cp1 = on (p_i._1)
            val cp2 = r2.on (p_i._2)
            if domain(cp1) != r2.domain(cp2) then flaw ("join", "differing domain strings")

            result = null                                                         // FIX the next line & remove this line
//          result = col(cp1).filterPos2 (r2.col (cp2), p_i._3)                   // single predicate

            debug ("join", s"after predicate $i: result = $result")
            resultlist = if i == 0 then result else resultlist intersect result
        end for

        val smallmapbig = resultlist.groupBy (_._1)
        for i <- smallmapbig.keySet.toVector.sorted do
            val t = index(indextoKey(i))
            val bigindexs = smallmapbig (i).map (x => x._2)
            for j <- bigindexs do
                val u = r2.index(r2.indextoKey(j))
                r3.add (t ++ u)
            end for
        end for
        r3.materialize ()
    end join

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** JOIN this table and r2 keeping concatenated tuples that satisfy the predicate.
     *  Caveat:  Assumes both keys are needed for the new key (depending on the
     *           predicate both may not be required).
     *  @param predicate  the join predicate to be satisfied
     *  @param r2         the second table
     */
    def join (predicate: Predicate2, r2: Relation): Relation = ???

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the THETA-JOIN of this table and r2 keeping concatenated tuples that
     *  satisfy the given simple (3 token) condition.
     *  @param condition  the simple condition "a1 op a2"
     *  @param r2         the second table
     */
    def join (condition: String, r2: Relation): Relation = ???

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Join this relation and r2 by performing an "left-join".  Rows from both
     *  relations are compared requiring cName1 values to equal cName2 values.
     *  Disambiguate column names by appending "2" to the end of any duplicate column name.
     *  All rows from the left table are maintained with missing values indicators used
     *  where needed.
     *  @param cName1  the join column names of this relation (e.g., the Foreign Key)
     *  @param cName2  the join column names of relation r2 (e.g., the Primary Key)
     *  @param r2      the rhs relation in the join operation
     */
    def leftJoin (cName1: String, cName2: String, r2: Relation): Relation =
        leftJoin (on (cName1), on (cName2), r2)
    end leftJoin

    def leftJoin (x: Schema, y: Schema, r2: Relation): Relation = ???

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Join this relation and r2 by performing a "left join".  Rows from both
     *  relations are compared requiring cp1 values to equal cp2 values.
     *  This method returns all the rows from this relation, and the matched rows
     *  from relation r2.  It adds a null tuples for the unmatched rows of relation r2
     *  FIX: It requires relations this and r2 to be sorted on column cp1 and cp2 resp., as it uses Sort-Merge join
     *  @param cp1  the position of the join column of this relation
     *  @param cp2  the position of the join column of r2 relation
     *  @param r2   the rhs relation in the join operation
     */
    def leftJoin (cp1: Int, cp2: Int, r2: Relation): Relation =
        val r3 = Relation (name + "_leftJoin_" + r2.name, schema ++ r2.schema, domain ++ r2.domain, key)
        val absentTuple = nullTuple (r2.domain)
        var j = 0
        for i <- 0 until rows do
            val t = row(i)
            val t_cp1 = t(cp1)
            while j < r2.rows-1 && r2.col(cp2)(j) < t_cp1 do j += 1
            val j_aux = j
            if t_cp1 == r2.row(j)(cp2) then
                while j < r2.rows && r2.col(cp2)(j) == t_cp1 do
                    val u = r2.row(j)
                    r3.add_ni (t ++ u)
                    j += 1
                end while
                j = j_aux
            else
                r3.add_ni (t ++ absentTuple)
            end if
        end for
        r3.materialize ()
    end leftJoin

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Join this relation and r2 by performing a "right join".  Rows from both
     *  relations are compared requiring cp1 values to equal cp2 values.
     *  This method returns all the rows from this relation, and the matched rows
     *  from relation r2.  It adds a null tuples for the unmatched rows of relation r2
     *  @param cp1  the position of the join column of this relation
     *  @param cp2  the position of the join column of r2 relation
     *  @param r2   the rhs relation in the join operation
     */
    def rightJoin (cp1: Int, cp2: Int, r2: Relation): Relation =
        r2.leftJoin (cp2, cp1, this)
    end rightJoin

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Join this relation and r2 by performing an "approximate left-join".  Rows from both
     *  relations are compared requiring cName1 values to apprximately equal cName2 values.
     *  Disambiguate column names by appending "2" to the end of any duplicate column name.
     *  All rows from the left table are maintained with missing values indicators used
     *  where needed.
     *  @param thres   the approximate equality threshold
     *  @param cName1  the join column names of this relation (e.g., the Foreign Key)
     *  @param cName2  the join column names of relation r2 (e.g., the Primary Key)
     *  @param r2      the rhs relation in the join operation
     */
    def leftJoinApx (thres: Double = 0.001) (cName1: String, cName2: String, r2: Relation): Relation =
//      setThreshold (thres)
        leftJoinApx (on (cName1), on (cName2), r2)
    end leftJoinApx

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Join this relation and r2 by performing a "left join".  Rows from both
     *  relations are compared requiring cp1 values to approximately equal cp2 values.
     *  This method returns all the rows from this relation, and the matched rows
     *  from relation r2.  It adds a null tuples for the unmatched rows of relation r2
     *  FIX: It requires relations this and r2 to be sorted on column cp1 and cp2 resp.,
     *  as it uses Sort-Merge join
     *  @param cp1  the position of the join column of this relation
     *  @param cp2  the position of the join column of r2 relation
     *  @param r2   the rhs relation in the join operation
     */
    def leftJoinApx (cp1: Int, cp2: Int, r2: Relation): Relation =
        val r3 = Relation (name + "_leftJoinApx_" + r2.name, schema ++ r2.schema, domain ++ r2.domain, key)
/*******  FIX
        val absentTuple = nullTuple (r2.domain)
        var j = 0
        for i <- 0 until rows do
            val t = row(i)
            val t_cp1 = t(cp1)
            while j < r2.rows-1 && !=~ (Vec (r2.col(cp2), j), t_cp1) && r2.col(cp2)(j) < t_cp1 do j += 1
            val j_aux = j
            if =~ (t_cp1, r2.row(j)(cp2)) then
                while j < r2.rows && =~ (Vec (r2.col(cp2), j), t_cp1) do
                    val u = r2.row(j)
                    r3.add_ni (t ++ u)
                    j += 1
                end while
                j = j_aux
            else
                r3.add_ni (t ++ absentTuple)
            end if
        end for
*******/
        r3.materialize ()
    end leftJoinApx

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Join this relation and r2 by performing a "right join".  Rows from both
     *  relations are compared requiring cp1 values to approximately equal cp2 values.
     *  This method returns all the rows from this relation, and the matched rows
     *  from relation r2.  It adds a null tuples for the unmatched rows of relation r2
     *  @param cp1  the position of the join column of this relation
     *  @param cp2  the position of the join column of r2 relation
     *  @param r2   the rhs relation in the join operation
     *
    def rightJoinApx (cp1: Int, cp2: Int, r2: Relation): Relation =
        r2.leftJoinApx (cp2, cp1, this)
    end rightJoinApx
     */

    // ================================================================== DIVIDE

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** DIVIDE this table by table r2.  Requires a tuple in the quotient part of
     *  this table to be paired with all tuples in table r2.
     *  @param r2  the second table
     */
    infix def divide (r2: Relation): Relation = ???

    // ================================================================ GROUP BY

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Group this relation by the specified column name, returning this relation.
     *  Each value in column cName will be mapped to a vector of row numbers containing
     *  the value, e.g., *  { (a, A), (b, C), (a, T) } makes map a -> (0, 2), b -> (1).
     *  @param cName  the group column name
     */
    def groupBy (cName: String): Relation =
        if ! (schema contains cName) then
            flaw ("groupBy", s"cName = $cName is not contained in schema")
        end if

        val _col = col(on (cName))                                           // the cName column
        for i <- indices do
            val key = _col(i).asInstanceOf [ValueType]
            val loc = groupMap.getOrElseUpdate (key, VEC [Int] ())
            loc += i                                                             // add index/row num i
        end for
        this
    end groupBy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Group this relation by the specified column names, returning this relation.
     *  @param cName  the group column names
     *
    def groupBy (cName: String*): Relation =
        if ! cName.map (c => schema contains(c)).reduceLeft (_ && _) then
            flaw ("groupBy", "groupbyName used to groupby doesn't exist in the cName")
        end if
        val equivCol = Vector.fill [Vectr] (schema.length)(null)
        if rows == 0 then return this

        val cPos = cName.map (on (_))

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /*  Sort on the given columns.
         *  @param sortColumn  the set of columns to sort on
         */
        def sortcol (sortColumn: Set [ValueType]): Vectr =
            println (s"sortCol: sortColumn = $sortColumn")
            var colcol: Vectr = null
            val domain = null
//          for x <- sortColumn do colcol = Vec.:+ (colcol, x, domain, 0)
            for x <- sortColumn do colcol = Vec.:+ (colcol, x)

            val sortcol = colcol; sortcol.sort (); sortcol
        end sortcol

        var groupIndexMap = Map [ValueType, Vector [ValueType]] ()
        val tempIndexMap  = Map [ValueType, Vector [ValueType]] ()
        var sortlst: Vectr = null

        for i <- cPos.indices do
            if i == 0 then
                index.foreach (indexmap => {
                    val key   = indexmap._2(cPos(i)).toString
                    val value = indexmap._1
                    if groupIndexMap contains key then groupIndexMap += key -> (groupIndexMap(key) :+ value)
                    else groupIndexMap += key -> Vector(value)
                }) // foreach
            else
                tempIndexMap.clear ()
                groupIndexMap.foreach (groupindexmap => {
                    val tempidxlist = groupindexmap._2
                    for idx <- tempidxlist do
                        val key   = groupindexmap._1.toString + "," + index(idx)(cPos(i))
                        val value = idx
                        if tempIndexMap contains(key) then tempIndexMap += key -> (tempIndexMap(key) :+ value)
                        else tempIndexMap += key -> Vector(value)
                    end for
                }) // for each
                groupIndexMap = tempIndexMap
            end if

            if i == cPos.size - 1 then
                orderedIndex = Vector ()
                grouplist    = Vector [Int] ()
                sortlst      = sortcol (groupIndexMap.keySet.toSet)
                for k <- 0 until sortlst.size do
                    val indexes  = groupIndexMap(Vec(sortlst, k))
                    orderedIndex = orderedIndex ++ indexes
                    grouplist    = grouplist :+ orderedIndex.length
                end for
            end if
        end for
        this
    end groupBy
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Assuming this table has been grouped by attribute g_atr, create a table
     *  where the first column is g_atr and the rest are AGGREGATE FUNCTIONs applied
     *  to their corresponding attributes.
     *  @param g_atr  the attribute the table has been grouped on
     *  @param f_as   the aggregate function and the attribute to apply it to (as varargs)
     */
    def aggregate (g_atr: String, f_as: (AggFunction, String)*): Relation = ???

    // ================================================================= ORDER BY

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Order (ascending) the rows in the relation by the selected columns _cName.
     *  A stable sorting is used to allow sorting on multiple columns.
     *  @param _cName  the column names that are to be sorted
     */
    def orderBy (_cName: String*): Relation =
        val cName = _cName.distinct
        if ! cName.map (c => schema contains (c)).reduceLeft (_ && _) then
            flaw ("orderBy", "cName used to orderBy does not exist in relation")
        end if

        val newCol = Vector.fill [Vectr] (cols)(null)
        val r2 = new Relation (name + "_o_" + ucount.inc (), schema, domain, key, newCol)

        val perm = orderByHelper (VEC (cName.map (on (_))*))
        for i <- perm do r2.add (row(i))
        r2.materialize ()
    end orderBy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Order (descending) the rows in the relation by the selected columns _cName.
     *  A stable sorting is used to allow sorting on multiple columns.
     *  @param _cName  the column names that are to be sorted
     */
    def orderByDesc (_cName: String*): Relation =
        val cName = _cName.distinct
        if ! cName.map (c => schema contains (c)).reduceLeft (_ && _) then
            flaw ("orderByDesc", "cName used to orderByDesc does not exist in relation")
        end if

        val newCol = Vector.fill [Vectr] (cols) (null)
        val r2 = new Relation (name + "_r_" + ucount.inc (), schema, domain, key, newCol)

        val perm = orderByHelper (VEC (cName.map (on (_))*))
        for i <- perm.reverse do r2.add (row(i))
        r2.materialize ()
    end orderByDesc

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Helper method for orderBy and orderByDesc.  Performs indirect merge-sort.
     *  @param cPos  sequence of column positions to sort
     */
    private def orderByHelper (cPos: VEC [Int]): Array [Int] =
        val perm: Array [Int] = null                            // FIX - change to var

        for i <- cPos.indices do
            val col_i = col (cPos(i)).toArray
            println (col_i)                                     // FIX - remove
/* FIX - add MergeSortIndirect to scalation package
            perm = if i == 0 then (new MergeSortIndirect (col_i)()).isort ()
                   else           (new MergeSortIndirect (col_i)(perm)).isort ()
*/
        end for
        perm
    end orderByHelper

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the basic statistics for each column of this table.
     */
    def stats: Relation = ???

    // ================================================================= UPDATES 

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add a tuple to this relation as a new row using col2 as a temp col to improve
     *  performance.  After a batch of adds, call materialize.
     *  @param tuple  an aggregation of columns values (new row)
     */
    @throws (classOf [Exception])
    def add (tuple: Row): Relation =
        try
            if tuple == null then throw new Exception ("Relation.add method: tuple is null")
            val rowIdx   = col2(0).length
            val newkey   = tuple(on(key(0)))                        // FIX - allow composite keys
            index       += newkey -> tuple
            keytoIndex  += newkey -> rowIdx
            orderedIndex = orderedIndex :+ newkey
            indextoKey  += rowIdx -> newkey
            for j <- tuple.indices do addElem (j, rowIdx, tuple(j))
        catch
            case ex: NullPointerException =>
                println ("tuple'size is: " + tuple.size)
                println ("col'size is:   " + col.size)
                throw ex
        end try
        this
    end add

    def add (t: Tuple): Relation = ???                               // not to be implemented

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add a tuple to this relation as a new row, materialize and return updated table.
     *  May call for last tuple in a batch of tuples to add
     *  @param tuple  an aggregation of columns values (new row)
     */
    def addm (tuple: Row): Relation =
        add (tuple)
        materialize ()
//      println ("addm: updated relation"); show ()
        this
    end addm

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add an element into col2, the holding area for input.  If the types
     *  of column domains are specified, the types are checked.
     *  @param j       the j-th column of col2
     *  @param rowIdx  the row index
     *  @param elem    the element to added
     */
    private def addElem (j: Int, rowIdx: Int, elem: ValueType): Unit =
        val typ = if domain == null then 'X' else domain(j)
        try
            //col2(j)(rowIdx) = elem
            col2(j) += elem
        catch
            case ex: ClassCastException =>
                if typ == 'S' then
//                  println (s"warning in addElem: colIdx j = $j, rowIdx = $rowIdx, elem = $elem, class = ${elem.getClass}, typ = $typ")
                    //col2(j)(rowIdx) = elem.toString                           // anything can be a string
                    col2(j) += elem.toString
                else if elem.isInstanceOf [String] || elem.isInstanceOf [Char] then
                    println (s"warning in addElem: colIdx j = $j, rowIdx = $rowIdx, elem = $elem, class = ${elem.getClass}, typ = $typ")
                else
                    println (s"exception in addElem: name = $name, colIdx j = $j, rowIdx = $rowIdx, elem = $elem, class = ${elem.getClass}, typ = $typ")
                    throw ex
                end if
        end try
    end addElem

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add a tuple into the col2, without maintaining the index (No Index (ni),
     *  orderedIndex, keytoIndex and indextoKey.
     *  @param tuple  the tuple to add
     */
    private def add_ni (tuple: Row): Unit =
        val rowIdx = col2(0).length
        for j <- tuple.indices do addElem (j, rowIdx, tuple(j))
    end add_ni

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Materialize the relation by copying the temporary col2 into col.
     *  It needs to be called by the end of the relation construction.
     */
    def materialize (): Relation =
        if domain == null then materialize1 () else materialize2 ()
    end materialize

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Materialize the relation by copying the temporary col2 into col.
     *  It needs to be called by the end of the relation construction.
     *  This version uses the type/domain of the first value to transform the col2 to col.
     */
    private [relation] def materialize1 (): Relation =

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /*  Transform the j-th column of col2 to the appropriate vector type.
         *  @param j  the j-th column index in the relation
         */
        def transform1 (j: Int): Vectr =
            val first       = col2(j)(0)
            var col_j: Vectr = null

            try
                col_j = first match
                case _: Double  => VectorD (col2(j).asInstanceOf [VEC [Double]])
                case _: Int     => VectorI (col2(j).asInstanceOf [VEC [Int]])
                case _: Long    => VectorL (col2(j).asInstanceOf [VEC [Long]])
                case _: String  => VectorS (col2(j).asInstanceOf [VEC [String]])
                case _: TimeNum => VectorT (col2(j).asInstanceOf [VEC [TimeNum]])
            catch
                case ex: java.lang.ClassCastException =>
                flaw ("transform1", s"ClassCastException - name = $name, j = $j, first = $first, col2(j) = ${col2(j)}")

            col_j
        end transform1

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /*  Transform the j-th column of col2 to the appropriate vector type and append.
         *  @param j  the j-th column index in the relation
         */
        def transform1b (j: Int): Vectr =
            val first       = col2(j)(0)
            var col_j: Vectr = null

            try
                col_j = first match
                case _: Double  => VectorD (col(j).asInstanceOf [VectorD] ++ col2(j).asInstanceOf [VEC [Double]])
                case _: Int     => VectorI (col(j).asInstanceOf [VectorI] ++ col2(j).asInstanceOf [VEC [Int]])
                case _: Long    => VectorL (col(j).asInstanceOf [VectorL] ++ col2(j).asInstanceOf [VEC [Long]])
                case _: String  => VectorS (col(j).asInstanceOf [VectorS] ++ col2(j).asInstanceOf [VEC [String]])
                case _: TimeNum => VectorT (col(j).asInstanceOf [VectorT] ++ col2(j).asInstanceOf [VEC [TimeNum]])
            catch
                case ex: java.lang.ClassCastException =>
                flaw ("transform1b", s"ClassCastException - name = $name, j = $j, first = $first, col2(j) = ${col2(j)}")

            col_j
        end transform1b

        debug ("materialize1", s"col2 = $col2")
        if col2(0).size == 0 then
            flaw ("materialize1", "no rows in col2 to materialize")
        else if colEmpty then
            col = (for j <- col2.indices yield transform1(j)).toVector
            for j <-col2.indices do col2(j).clear ()
        else
            col = (for j <- col.indices yield transform1b(j)).toVector
            for j <-col2.indices do col2(j).clear ()
        end if
        this
    end materialize1

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Materialize the relation by copying the temporary col2 into col.
     *  It needs to be called by the end of the relation construction.
     *  This version uses domain to transform the col2 to col according to the domain indicator.
     */
    private [relation] def materialize2 (): Relation =

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /*  Transform the j-th column of col2 to the appropriate vector type.
         *  @param j  the j-th column index in the relation
         */
        def transform2 (j: Int): Vectr =
            val domain_j    = domain(j)
            var col_j: Vectr = null

            try
                col_j = domain_j match
                case 'D' => VectorD (col2(j).asInstanceOf [VEC [Double]])
                case 'I' => VectorI (col2(j).asInstanceOf [VEC [Int]])
                case 'L' => VectorL (col2(j).asInstanceOf [VEC [Long]])
                case 'S' => VectorS (col2(j).asInstanceOf [VEC [String]])
                case 'X' => VectorS (col2(j).asInstanceOf [VEC [String]])
                case 'T' => VectorT (col2(j).asInstanceOf [VEC [TimeNum]])
                case  _  => flaw ("materialize2.transform2", s"($j) vector type not supported domain ($domain_j)"); null
            catch
                case ex: java.lang.ClassCastException =>
                flaw ("transform2", s"ClassCastException - name = $name, j = $j, domain_j = $domain_j, col2(j) = ${col2(j)}")

            col_j
        end transform2

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /*  Transform the j-th column of col2 to the appropriate vector type and append.
         *  @param j  the j-th column index in the relation
         */
        def transform2b (j: Int): Vectr =
            val domain_j    = domain(j)
            var col_j: Vectr = null

            try
            col_j = domain_j match
            case 'D' => col(j).asInstanceOf [VectorD] ++ col2(j).asInstanceOf [VEC [Double]]
            case 'I' => col(j).asInstanceOf [VectorI] ++ col2(j).asInstanceOf [VEC [Int]]
            case 'L' => col(j).asInstanceOf [VectorL] ++ col2(j).asInstanceOf [VEC [Long]]
            case 'S' => col(j).asInstanceOf [VectorS] ++ col2(j).asInstanceOf [VEC [String]]
            case 'X' => col(j).asInstanceOf [VectorS] ++ col2(j).asInstanceOf [VEC [String]]
            case 'T' => col(j).asInstanceOf [VectorT] ++ col2(j).asInstanceOf [VEC [TimeNum]]
            case  _  => flaw ("materialize2.transform2b", s"($j) vector type not supported domain ($domain_j)"); null
            catch
                case ex: java.lang.ClassCastException =>
                flaw ("transform2b", s"ClassCastException - name = $name, j = $j, domain_j = $domain_j, col2(j) = ${col2(j)}")

            col_j
        end transform2b

        debug ("materialize2", s"col2 = $col2")
        if col2(0).size == 0 then
            flaw ("materialize2", "no rows in col2 to materialize")
        else if colEmpty then
            col = (for j <- col2.indices yield transform2(j)).toVector
            for j <-col2.indices do col2(j).clear ()
        else
            col = (for j <- col.indices yield transform2b(j)).toVector
            for j <-col2.indices do col2(j).clear ()
        end if
        this
    end materialize2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether all of the columns in the relation are empty.
     */
    def colEmpty: Boolean = col.forall (_ == null)
/*
        for column <- col if column != null do return false
        true
    end colEmpty
*/

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the column named cName using newVal for elements with value matchStr.
     *  @param cName     the name of the column to be updated
     *  @param newVal    the value used to assign updated values
     *  @param matchVal  the value to be matched to elements
     */
    def update (cName: String, newVal: ValueType, matchVal: ValueType): Boolean =
        var change = false
        val col_j = col(on(cName))
        for i <- col_j.indices if col_j(i) == matchVal do { change = true; assign (col_j, i, newVal) }
        change
    end update

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Assign the value newVal to column/vector c at index position i.
     *  @param c       the name of the column/vector to be assigned
     *  @param i       the index position to be assigned
     *  @param newVal  the value used for assignment
     */
    def assign (col_j: Vectr, i: Int, newVal: ValueType): Unit =
        val first = col_j(0)
        first match
        case _: Double   => (col_j.asInstanceOf [VectorD])(i) = newVal.asInstanceOf [Double]
        case _: Int      => (col_j.asInstanceOf [VectorI])(i) = newVal.asInstanceOf [Int]
        case _: Long     => (col_j.asInstanceOf [VectorL])(i) = newVal.asInstanceOf [Long]
        case _: String   => (col_j.asInstanceOf [VectorS])(i) = newVal.asInstanceOf [String]
        case _: TimeNum  => (col_j.asInstanceOf [VectorT])(i) = newVal.asInstanceOf [TimeNum]
        case null        => flaw ("assign", s"vector type ($first) not supported")
        end match
    end assign

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the column named cName using function func for elements with
     *  value matchVal.
     *  @param cName     the name of the column to be updated
     *  @param func      the function used to assign updated values
     *  @param matchVal  the value to be matched to elements
     */
    def update (cName: String, func: ValueType => ValueType, matchVal: ValueType): Boolean =
        var change = false
        val col_j = col (on(cName))
        for i <- col_j.indices if col_j(i) == matchVal do { change = true; assign (col_j, i, func (col_j(i))) }
        change
    end update

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the column named cName using function func for elements where
     *  the predicate pred evaluates to true.
     *  @param cName  the name of the column to be updated
     *  @param func   the function used to assign updated values
     *  @param pred   the predicated used to select elements for update
     *
    def update (cName: String, func: (ValueType) => ValueType, pred: APredicate): Boolean =
        val col_j = col (on(cName))
        for i <- col_j.indices if pred (col_j(i)) do assign (col_j, i, func (col_j(i)))
    end update
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Delete the rows from this relation that satisfy the predicate.
     *  @param  p  the predicate
     */
    def delete (p: Predicate): Boolean =
        false
/*
        var pos = VEC [Int] ()
        for i <- p.indices do
            domain (on(p(i)._1)) match
            case 'D' => val pos1 = col (on(p(i)._1)).asInstanceOf [VectorD].filterPos (p(i)._2.asInstanceOf [Double => Boolean])
                        if i > 0 then pos = pos intersect pos1 else pos ++= pos1
            case 'I' => val pos1 = col (on(p(i)._1)).asInstanceOf [VectorI].filterPos (p(i)._2.asInstanceOf [Int => Boolean])
                        if i > 0 then pos = pos intersect pos1 else pos ++= pos1
            case 'L' => val pos1 = col (on(p(i)._1)).asInstanceOf [VectorL].filterPos (p(i)._2.asInstanceOf [Long => Boolean])
                        if i > 0 then pos = pos intersect pos1 else pos ++= pos1
            case 'S' => val pos1 = col (on(p(i)._1)).asInstanceOf [VectorS].filterPos (p(i)._2.asInstanceOf [String => Boolean])
                        if i > 0 then pos = pos intersect pos1 else pos ++= pos1
            case 'X' => val pos1 = col (on(p(i)._1)).asInstanceOf [VectorS].filterPos (p(i)._2.asInstanceOf [String => Boolean])
                        if i > 0 then pos = pos intersect pos1 else pos ++= pos1
            case 'T' => val pos1 = col (on(p(i)._1)).asInstanceOf [VectorT].filterPos (p(i)._2.asInstanceOf [TimeNum => Boolean])
                        if i > 0 then pos = pos intersect pos1 else pos ++= pos1
            case _   => flaw ("delete", "predicate type not supported")
                        null
            end match
        end for
        val indices = Set (0 to rows-1 :_*) diff pos.toSet
        for i <- 0 until cols do Vec.delete (col(i), pos.asInstanceOf [VEC [Int]])
        apply (indices.toArrayBuffer.sorted)
*/
    end delete

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this relation into a string column by column.
     */
    override def toString: String =
        val sb = new StringBuilder (s"Relation(name = $name, key = $key, domain = $domain,\nschema = $schema,\n")
        for i <- col.indices do sb.append (s"${col(i)} \n")
        sb.replace (sb.length-1, sb.length, ")").mkString
    end toString

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show this relation row by row with Fixed witdh columns.
     *  @param rng  the renage of rows to show/display
     */
    def show0 (rng: Range = 0 until rows): Unit =
        val wid   = 18                                                         // column width
        val rep   = math.max (wid, wid * schema.length)                        // repetition = width * # columns
        val title = s"| Relation name = $name, key-column = $key "

        println (s"|-${"-"*rep}-|")
        println (title + " "*math.max (0, rep-title.length) + "   |")
        println (s"|-${"-"*rep}-|")
        print ("| "); for cn <- schema do print (s"%${wid}s".format (cn)); println (" |")
        println (s"|-${"-"*rep}-|")
        for i <- rng do
            print ("| ")
            for cv <- row(i) do
                if cv.isInstanceOf [Double] then print (s"%${wid}g".format (cv))
                else print (s"%${wid}s".format (cv))
            end for
            println (" |")
        end for
        println (s"|-${"-"*rep}-|")
    end show0

    private def lineLen (domain: Domain, clen: Int): Int =
        math.max (clen, clen * schema.length) + (clen - 1) * domain.count (_ == 'X')
    end lineLen

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show this relation row by row.  Handles extended width strings.
     *  @param rng  the range of rows to show/display
     */
    def show (rng: Range = 0 until rows): Unit =
        val wid   = 18                                                         // column gap
        val wid1  = wid + 1                                                    // column width
        val rep   = lineLen (domain, wid1)                                     // repetition = line length
        val title = s"| Relation name = $name, key-column = $key "

        println (s"|-${"-"*rep}-|")
        println (title + " "*math.max (0, rep-title.length) + "   |")
        println (s"|-${"-"*rep}-|")

        print ("| ")
        for j <- 0 until schema.length do                                      // for each column
            val cn = schema(j)                                                 // column name
            if domain(j) == 'X' then
                print (s"%${2*wid+1}s".format (cn))
            else
                print (s"%${wid1}s".format (cn))
        end for
        println (" |")
        println (s"|-${"-"*rep}-|")

        for i <- rng do                                                        // for each row
            print ("| ")
            for j <- 0 until schema.length do                                  // for each column
                val cv = col(j)(i)                                             // column value
                if domain(j) == 'X' then
                    print (s" %${2*wid}s".format (cv))
                else if domain(j) == 'D' then
                    print (s" %${wid}g".format (cv))
                else
                    print (s" %${wid}s".format (cv))
            end for
            println (" |")
        end for
        println (s"|-${"-"*rep}-|")
    end show

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SHOW/print this relation's primary index.
     */
    def show_index (): Unit =
        println (s"\n>> Table $name has indexed primary key = ${stringOf (key)}")
        for (k, v) <- index do println (s"index: ${stringOf (k)} -> ${stringOf (v)}")
    end show_index

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show this relation's foreign keys.
     */
    def show_foreign_keys (): Unit =
        val wid    = 18                                                        // column width
        val rep    = wid * schema.length                                       // repetition = width * # columns
        val title  = s"| Relation name = $name, foreign keys = "
        val fkline = s"| $fKeys "

        println (s"|-${"-"*rep}-|")
        println (title + " "*(rep-title.length) + "   |")
        println (s"|-${"-"*rep}-|")
        println (fkline + " "*(rep-fkline.length) + "   |")
        println (s"|-${"-"*rep}-|")
    end show_foreign_keys

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this relation into a matrix of doubles, e.g., in the regression
     *  equation: xb = y create matrix xy.
     *  @param colPos  the column positions to use for the matrix
     *
    def toMatrix (colPos: VEC [Int]): MatrixD =
        val colVec = for x <- project (colPos).col yield x.toDouble
        MatrixD (colVec).transpose
    end toMatrix
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this relation into a matrix of doubles, e.g., in the regression
     *  equation: xb = y create matrix xy.  It will convert strings to doubles.
     *  @param cols  the column positions to use for forming the matrix
     */
    def toMatrix (cols: Array [Int] = Array.range (0, schema.size)): MatrixD =
        val colVec =
        for x <- project (cols).col yield {
            try x.toDouble
            catch case num: NumberFormatException => x.asInstanceOf [VectorS].map2Int._1.toDouble
        } // for
        MatrixD (colVec).transpose
    end toMatrix

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this relation into a matrix of doubles and a vector of doubles,
     *  e.g., in the regression equation: xb = y create matrix x and vector y.
     *  @param colPos   the column positions to use for the matrix
     *  @param colPosV  the column position to use for the vector
     */
    def toMatrixV (colPos: Array [Int], colPosV: Int): (MatrixD, VectorD) =
        val colVec = for x <- project (colPos).col yield x.toDouble
        (MatrixD (colVec).transpose, col(colPosV).toDouble)
    end toMatrixV

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert the colj column of this relation into a vector of doubles, etc.
     *  @param colj  the column position to use for the vector
     */
    def toVectorD (colj: Int = 0): VectorD = col(colj).toDouble
    def toVectorI (colj: Int = 0): VectorI = col(colj).toInt
    def toVectorL (colj: Int = 0): VectorL = col(colj).toLong
    def toVectorS (colj: Int = 0): VectorS = col(colj).toString2
    def toVectorT (colj: Int = 0): VectorT = col(colj).toTimeNum

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert the given columns within this relation to a map: keyColPos -> valColPos.
     *  @param keyColPos  the key column positions
     *  @param valColPos  the value column positions
     */
    def toMap (keyColPos: VEC [Int], valColPos: Int): Map [VEC [ValueType], ValueType] =
        val map = Map [VEC [ValueType], ValueType] ()
        for i <- indices do
            val tuple = row(i)
            map += keyColPos.map (tuple(_)) -> tuple(valColPos)
        end for
        map
    end toMap

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Save this relation in a file using serialization.
     */
    def save (): Unit =
        val oos = new ObjectOutputStream (new FileOutputStream (STORE_DIR + name + SER))
        oos.writeObject (this)
        oos.close ()
    end save

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Write this relation into a CSV file with each row written to a line.
     *  @param fileName  the file name of the data file
     */
    def writeCSV (fileName: String): Unit =
        val out = new PrintWriter (BASE_DIR + fileName)
        out.println (schema.toString.drop (5).dropRight (1))
        for i <- 0 until rows do out.println (row(i).toString.drop (7).dropRight (1))
        out.close
    end writeCSV

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Write this relation into a JSON file.
     *  @param fileName  the file name of the data file
     */
    def writeJSON (fileName: String): Unit =
        val out = new PrintWriter (BASE_DIR + fileName)

        out.println ("[")
        for i <- 0 until rows do
            out.println ("{")
            val row_i = row(i)
            for j <- 0 until cols do
                out.print ("\"" + schema(j) + "\"" + ":")
                out.print (row_i(j))
                if j != cols-1 then out.print (",")
                out.println ()
            end for
            out.println ("},")
        end for
        out.println ("]")
        out.close ()
    end writeJSON

    // ============================================ BUILT-IN AGGREGATE FUNCTIONS

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the mean of the values in column cName.
     *  @param cName  the column name
     */
    def avg (cName: String): Double  = col(on(cName)).toDouble.mean
    def mean (cName: String): Double = col(on(cName)).toDouble.mean

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of values in column cName.
     *  @param cName  the column name
     */
    def count (cName: String): Int = rows

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the maximum value in column cName.
     *  @param cName  the column name
     */
    def max (cName: String): ValueType =
        val cl = col(on(cName))
        cl(0) match
        case _: Double  => cl.asInstanceOf [VectorD].max
        case _: Int     => cl.asInstanceOf [VectorI].max
        case _: Long    => cl.asInstanceOf [VectorL].max
        case _: String  => cl.asInstanceOf [VectorS].max
        case _: TimeNum => cl.asInstanceOf [VectorT].max
        end match
    end max

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the minimum value in column cName.
     *  @param cName  the column name
     */
    def min (cName: String): ValueType =
        val cl = col(on(cName))
        cl(0) match
        case _: Double  => cl.asInstanceOf [VectorD].min
        case _: Int     => cl.asInstanceOf [VectorI].min
        case _: Long    => cl.asInstanceOf [VectorL].min
        case _: String  => cl.asInstanceOf [VectorS].min
        case _: TimeNum => cl.asInstanceOf [VectorT].min
        end match
    end min

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the sum of the values in column cName.
     *  @param cName  the column name
     */
    def sum (cName: String): ValueType =
        val cl = col(on(cName))
        cl(0) match
        case _: Double  => cl.asInstanceOf [VectorD].sum
        case _: Int     => cl.asInstanceOf [VectorI].sum
        case _: Long    => cl.asInstanceOf [VectorL].sum
        case _: String  => cl.asInstanceOf [VectorS].sum
        case _: TimeNum => cl.asInstanceOf [VectorT].sum
        end match
    end sum

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the variance of the values in column cName.
     *  @param cName  the column name
     */
    def variance (cName: String): Double = col(on(cName)).toDouble.variance

/** As seen from class Relation, the missing signatures are as follows.
 *  For convenience, these are usable as stub implementations.
 */
//    def leftJoin (thres: Double) (cName1: String, cName2: String, r2: Relation): Relation = ???

end Relation


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Example database for days of the week.
 */
object Ex_Days:

    val weekdays = Relation ("weekdays",
                             Array ("day", "time"),
                             Array ('S', 'D'), Array ("day"),
                             VEC [Row] (Vector ("Mon", 5.00),
                                        Vector ("Tue", 8.15),
                                        Vector ("Wed", 6.30),
                                        Vector ("Thu", 9.45),
                                        Vector ("Fri", 7.00)))

    val weekend  = Relation ("weekend",
                             Array ("day", "time"),
                             Array ('S', 'D'), Array ("day"),
                             VEC [Row] (Vector ("Sat", 3.00),
                                        Vector ("Sun", 4.30)))

end Ex_Days


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `relationTest` main function tests the operations provided by `Relation`.
 *  on the `Ex_Days` schema.
 *  > runMain scalation.database.relation.relationTest
 */
@main def relationTest (): Unit =

    import Ex_Days._

    weekdays.create_index ()
    weekend.create_index ()

    banner ("weekdays"); weekdays.show ()
    banner ("weekend");  weekend.show ()

    println (">>>>> project")
    test ("weekdays.project (\"day\")",  weekdays.project ("day"))
    test ("weekdays.project (\"time\")", weekdays.project ("time"))

    println (">>>>> selproject")
    test ("weekdays.selproject (\"day\", _ == \"Mon\")", weekdays.selproject ("day", _ == "Mon"))

    println (">>>>> select")
    test ("weekdays.select (\"day\",  _ == \"Mon\")", weekdays.select ("day",  _ == "Mon"))
    test ("weekdays.select (\"day\",  _ > \"Mon\")",  weekdays.select ("day",  _ > "Mon"))
    test ("weekdays.select (\"day\",  _ < \"Wed\")",  weekdays.select ("day",  _ < "Wed"))
    test ("weekdays.select (\"time\", _ == 5.00)",    weekdays.select ("time", _ == 5.00))
    test ("weekdays.select (\"time\", _ > 5.00)",     weekdays.select ("time", _ > 5.00))

    test ("weekdays.select (\"day\", _ > \"Mon\").select (\"time\", _ > 7.00)",
           weekdays.select ("day", _ > "Mon").select ("time", _ > 7.00))

    println (">>>>> union")
    val week = weekdays union weekend
    test ("weekdays union weekend", week)

    println (">>>>> intersect")
    test ("week intersect weekend", week intersect weekend)

    println (">>>>> addm")
    test ("weekend.addm (\"Zday\", 1.00)", weekend.addm (Vector ("Zday", 1.00)))

    println (">>>>> minus")
    test ("week minus weekend", week minus weekend)

    val cross = week product weekend
    println (">>>>> product")
    test ("week product weekend", cross)
    count (cross, "day").show ()

    println (">>>>> join")
    test ("week.join (\"day\", \"day\" weekend)", week.join ("day", "day", weekend))
    test ("week join weekend", week join weekend)

//  week.writeCSV ("relation" + ⁄ + "week.csv")

end relationTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `relationTest2` main function tests the operations provided by `Relation`.
 *  The relational algebra operators are given using Unicode.
 *  @see en.wikipedia.org/wiki/List_of_Unicode_characters
 *  > runMain scalation.database.relation.relationTest2
 */
@main def relationTest2 (): Unit =

    import Ex_Days._

    weekdays.create_index ()
    weekend.create_index ()

    banner ("weekdays"); weekdays.show ()
    banner ("weekend");  weekend.show ()

    println (">>>>> project")
    test ("weekdays.π (\"day\")",  weekdays.π ("day"))
    test ("weekdays.π (\"time\")", weekdays.π ("time"))

    println (">>>>> selproject")
    test ("weekdays.σπ (\"day\", _ == \"Mon\")", weekdays.selproject ("day", _ == "Mon"))

    println (">>>>> select")
    test ("weekdays.σ (\"day\",  _ == \"Mon\")", weekdays.σ ("day",  _ == "Mon"))
    test ("weekdays.σ (\"day\",  _ > \"Mon\")",  weekdays.σ ("day",  _ > "Mon"))
    test ("weekdays.σ (\"day\",  _ < \"Wed\")",  weekdays.σ ("day",  _ < "Wed"))
    test ("weekdays.σ (\"time\", _ == 5.00)",    weekdays.σ ("time", _ == 5.00))
    test ("weekdays.σ (\"time\", _ > 5.00)",     weekdays.σ ("time", _ > 5.00))

    test ("weekdays.σ (\"day\", _ > \"Mon\").σ (\"time\", _ > 7.00)",
           weekdays.σ ("day", _ > "Mon").σ ("time", _ > 7.00))

    println (">>>>> union")
    val week = weekdays ⋃ weekend
    test ("weekdays ⋃ weekend)", week)

    println (">>>>> intersect")
    test ("week ⋂ weekend", week ⋂ weekend)

    println (">>>>> addm")
    test ("weekend.addm (\"Zday\", 1.00)", weekend.addm (Vector ("Zday", 1.00)))

    println (">>>>> minus")
    test ("week - weekend", week - weekend)

    println (">>>>> product")
    test ("week × weekend", week × weekend)

    println (">>>>> join")
    test ("week ⋈ weekend", week ⋈ weekend)

end relationTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `relationTest3` main function tests the operations provided by `Relation`.
 *  It test various aggregate/OLAP operations on a simple data warehouse fact table.
 *  @see www.codeproject.com/Articles/652108/Create-First-Data-WareHouse
 *  FIX - allow entering doubles as "13" rather than "13.0"
 *  > runMain scalation.database.relation.relationTest3
 *
@main def relationTest3 (): Unit =

    import Relation.{max, min}
    import Ex_ProductSales._

    val costVprice = productSales.project ("ProductActualCost", "SalesTotalCost")

    productSales.show ()

    println ("productSales = " + productSales)
    println ("productSales.project (\"ProductActualCost\", \"SalesTotalCost\") = " + costVprice)

    banner ("Test count")
    println ("count (productSales) = " + count (productSales))
    println ("-" * 60)
    println ("count (costVprice)   = " + count (costVprice))

    banner ("Test min")
    println ("min (productSales)   = " + min (productSales))
    println ("-" * 60)
    println ("min (costVprice)     = " + min (costVprice))

    banner ("Test max")
    println ("max (productSales)   = " + max (productSales))
    println ("-" * 60)
    println ("max (costVprice)     = " + max (costVprice))

    banner ("Test sum")
    println ("sum (productSales)   = " + sum (productSales))
    println ("-" * 60)
    println ("sum (costVprice)     = " + sum (costVprice))

    banner ("Test expectation/mean")
    println ("mean (productSales)  = " + mean (productSales))
    println ("-" * 60)
    println ("mean (costVprice)    = " + mean (costVprice))

    banner ("Test variance")
    println ("variance (productSales) = " + variance (productSales))
    println ("-" * 60)
    println ("variance (costVprice)   = " + variance (costVprice))

end relationTest3
 */


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `relationTest4` main function tests conversion `Relation` to a matrix.
 *  > runMain scalation.database.relation.relationTest4
 *
@main def relationTest4 (): Unit =

    import Ex_ProductSales._

    val (mat, vec) = productSales.toMatrixDD (Array.range (0, 11), 11)

    banner ("productSales")
    productSales.show ()

    banner ("mat and vec")
    println ("mat = " + mat)
    println ("vec = " + vec)

end relationTest4
 */


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `relationTest5` main function tests the interoperability between Relations and Matrices.
 *  > runMain scalation.database.relation.relationTest5
 */
@main def relationTest5 (): Unit =

    val sales_item1 = Relation ("Sales_Item1",
        Array ("Date", "FL", "GA", "NC", "SC"),
        Array ('S', 'I', 'I', 'I', 'I'), Array ("Date"),
        VEC [Row] (Vector ("20130101", 10, 5, 5, 4),
                   Vector ("20130102", 20, 30, 40, 25),
                   Vector ("20130103", 8, 6, 9, 9),
                   Vector ("20130104", 6, 7, 9, 10),
                   Vector ("20130105", 4, 7, 9, 10)))

    val price_item1 = Relation ("Price_Item1",
        Array ("Date", "FL", "GA", "NC", "SC"),
        Array ('S', 'D', 'D', 'D', 'D'), Array ("Date"),
        VEC [Row] (Vector ("20130101", 1.6, 1.6, 1.5, 1.3),
                   Vector ("20130102", 1.6, 1.6, 1.5, 1.2),
                   Vector ("20130103", 1.5, 1.6, 1.5, 1.4),
                   Vector ("20130104", 1.4, 1.7, 1.5, 1.4),
                   Vector ("20130105", 1.4, 1.7, 1.4, 1.4)))

    val revenue = Relation ("Revenue",
        Array ("Item", "FL", "GA", "NC", "SC"),
        Array ('S', 'D', 'D', 'D', 'D'), Array ("Item"))

    sales_item1.show ()
    price_item1.show ()

    val x = sales_item1.toMatrix (Array.range (1, 5))
    val y = price_item1.toMatrix (Array.range (1, 5))
    println (s"x = $x")
    println (s"y = $y")
    val z = x *~ y
    println (s"z = $z")
    val colsums = z.sumV
    val row: Row = "Item1" +: colsums.toVector
    println (s"row = $row")
    revenue.add (row)      // FIX
    revenue.materialize ()

    banner ("revenue")
    revenue.show ()

end relationTest5


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `relationTest6` main function tests indexjoin, groupby and eproject
 *  (aggregate operator).
 *  > runMain scalation.database.relation.relationTest6
 *
@main def relationTest6 (): Unit =

    banner ("database")

    val professor = Relation ("professor", 0, "ISS", "pid", "name", "prodeptid")
    TableGen.popTable (professor, 50)  // 10
    professor.create_index ()
    professor.show ()

    val course = Relation ("course", 0, "ISS", "cid","crsname", "descr")
    TableGen.popTable (course, 50)   // 20
    course.create_index ()
    course.show ()

    val teaching = Relation ("teaching", 0, "IISI", "tid", "cid", "semester", "pid")
    teaching.fKeys = VEC (("cid", "course", 0), ("pid", "professor", 0))
    TableGen.popTable (teaching, 50, VEC (course, professor))
    teaching.create_index ()
    teaching.show ()
    teaching.showFkey ()

// FIX - fails when 10, 20 < 50

//  def count1 (r: Table, c: String): Vect = ???

    banner ("joinindex")
    teaching.joinindex (Array ("pid"), Array ("pid"), professor).show ()
    banner ("groupBy.eproject")
//  teaching.groupBy ("cid").eproject ((count, "pid_count", "pid"))("tid", "semester").show ()   // FIX
//  teaching.groupBy ("cid").eproject ("count", "cid_count", "cid").show ()
//  teaching.groupBy ("cid").eproject (count1, "cid_count", "cid").show ()

    val a = (count1, "cid_count", "cid").asInstanceOf [AggColumn]
    teaching.groupBy ("cid").eproject (a).show ()

end relationTest6
 */


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `relationTest7` main function tests join method.
 *  > runMain scalation.database.relation.relationTest7
 */
@main def relationTest7 (): Unit =

    val professor = Relation ("professor",
        Array ("pid", "name", "department", "title"),
        Array ('I', 'S', 'S', 'I'), Array ("pid"),
        VEC [Row] (Vector (1, "jackson", "pharm", 4),
                   Vector (2, "ken", "cs", 2),
                   Vector (3, "pan", "pharm", 0),
                   Vector (4, "yang", "gis", 3),
                   Vector (5, "zhang", "cs", 0),
                   Vector (6, "Yu", "cs", 0)))

    val professor2 = Relation ("professor",
        Array ("pid", "name", "department", "title"),
        Array ('I', 'S', 'S', 'I'), Array ("pid"),
        VEC [Row] (Vector (7, "LiLy", "gis", 5),
                   Vector (8, "Marry", "gis", 5),
                   Vector (0, "Kate", "cs", 5)))

    professor.create_index ()
    professor2.create_index ()

    banner ("professor")
    professor.show ()
    banner ("professor2")
    professor2.show ()

    banner ("join")
    //professor.join (professor2, ("pid", "pid", (x: Int, y: Int) => x < y)).show ()   // FIX
    //professor.join (professor2, ("pid", "pid", _ < _)).show ()                       // FIX

end relationTest7


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `relationTest8` main function tests save method.
 *  > runMain scalation.database.relation.relationTest8
 */
@main def relationTest8 (): Unit =

    val professor = Relation ("professor",
        Array ("pid", "name", "department", "title"),
        Array ('I', 'S', 'S', 'I'), Array ("pid"),
        VEC [Row] (Vector (1, "jackson", "pharm", 4),
                   Vector (2, "ken", "cs", 2),
                   Vector (3, "pan", "pharm", 0),
                   Vector (4, "yang", "gis", 3),
                   Vector (5, "zhang", "cs", 0),
                   Vector (6, "Yu", "cs", 0)))

    val professor2 = Relation ("professor2",
        Array("pid", "name", "department", "title"),
        Array ('I', 'S', 'S', 'I'), Array ("pid"),
        VEC [Row] (Vector (1, "jackson", "pharm", 4),
                   Vector (2, "ken", "cs", 2)))

    professor.show ()
    professor.save ()
    professor2.show ()
    professor2.save ()

end relationTest8


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `relationTest9` main function tests apply method to load a saved relation.
 *  > runMain scalation.database.relation.relationTest9
 */
@main def relationTest9 (): Unit =

    Relation ("professor").show ()

end relationTest9


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `relationTest10` main function tests the orderBy method.
 *  > runMain scalation.database.relation.relationTest10
 *
@main def relationTest10 (): Unit =

    import Ex_ProductSales._

    productSales.orderBy ("SalesTotalCost", "Deviation").show ()

end relationTest10
 */


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `relationTest11` main function tests the `Relation` on the traffic schema.
 *  > runMain scalation.database.relation.relationTest11
 */
@main def relationTest11 (): Unit =

    val sensor  = Relation ("sensor",
                            Array ("sensorID", "model", "latitude", "longitude", "on"),
                            Array ('I', 'S', 'D', 'D', 'I'), Array ("sensorID"),
                            VEC [Row] ())

    val road    = Relation ("road",
                            Array ("roadID", "rdName", "lat1", "long1", "lat2", "long2"),
                            Array ('I', 'S', 'D', 'D', 'D', 'D'), Array ("roadID"),
                            VEC [Row] ())

    val mroad   = Relation ("road",
                            Array ("roadID", "rdName", "lanes", "lat1", "long1", "lat2", "long2"),
                            Array ('I', 'S', 'I', 'D', 'D', 'D', 'D'), Array ("roadID"),
                            VEC [Row] ())

    val traffic = Relation ("traffic",
                            Array ("time", "sensorID", "count", "speed"),
                            Array ('L', 'I', 'I', 'D'), Array ("time", "sensorID"),
                            VEC [Row] ())

    val wsensor = Relation ("sensor",
                            Array ("sensorID", "model", "latitude", "longitude"),
                            Array ('I', 'S', 'D', 'D'), Array ("sensorID"),
                            VEC [Row] ())

    val weather = Relation ("weather",
                            Array ("time", "sensorID", "precipitation", "wind"),
                            Array ('L', 'I', 'I', 'D'), Array ("time", "sensorID"),
                            VEC [Row] ())

    sensor.show ()
    road.show ()
    mroad.show ()
    traffic.show ()
    wsensor.show ()
    weather.show ()

end relationTest11


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `relationTest12` main function tests the `Relation` class on JSON data.
 *  @see www.learningcontainer.com/sample-json-file
 *  FIX - does not work for Scala 2.13
 *  > runMain scalation.database.relation.relationTest12
 *
@main def relationTest12 (): Unit =

    val fname    = BASE_DIR + "employee.json"
    println (s"fname = $fname")
    val employee = Relation (fname, "employee")

    employee.show ()

end relationTest12
 */

