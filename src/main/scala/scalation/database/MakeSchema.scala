
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Hao Peng
 *  @version 2.0
 *  @date    Tue Oct 6 12:27:00 EDT 2015
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Helper Methods for Making Database Schema
 */

package scalation
package database

import java.time.DateTimeException

import scala.collection.mutable.{ArrayBuffer => Bag}
import scala.math.max
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat.{MatrixD, VectorS}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MakeSchema` object provides helper methods for making database schema.
 *  It attempts to infers the domain (data-type) for a column of data, e.g.,
 *  from a csv file or database table.  The domain for a column is inferred by
 *  examining the data elements in a sample from its vector (`VectorS`).
 *  @see `makeSchemaTest2` for an example of making a schema for a datafile.
 */
object MakeSchema:

    /** Transition matrix: state = function (state, datatype)
     */
    private val tMatrix = MatrixD ((5, 5), 0, 1, 2, 4, 4,    // Int
                                           1, 1, 2, 4, 4,    // Long
                                           2, 2, 2, 4, 4,    // Double
                                           4, 4, 4, 3, 4,    // TimeNum
                                           4, 4, 4, 4, 4)    // String

    /** Regular expression for integers
     */
    private val iPattern = "[\\-\\+]?\\d+".r.pattern

    /** Regular expression for floating point numbers
     */
    private val dPattern = "[\\-\\+]?\\d*(\\.\\d+)?".r.pattern

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether the str argument has a date-time format.
     *  @see `scalation.TimeNum`
     *  @param str  the string argument to be checked
     */
    def isDateTime (str: String): Boolean =
        var correct = true
        try
            TimeNum (str)
        catch
            case _ : DateTimeException => correct = false
        correct
    end isDateTime

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Analyze the type of a given column as a `VectorS` by using regular expression matching.
     *  Types can be Double (D), Integer (I), Long (L), String (S), or TimeNum (T).
     *  @param c           the column/VectorS to have its type analyzed
     *  @param samplePerc  the percentage of c used to analyze the type of column c
     */
    def analyzeType (c: VectorS, samplePerc: Int = 100): Char =
        val ns    = max (1.0, c.dim * (samplePerc / 100.0)).toInt         // number of samples
        var state = 0

        var (go, i) = (true, 0)
        cfor (go && i < ns, i += 1) {
            val typ =                                                     // determine type of i-th value
            if iPattern.matcher (c(i).toString).matches () then
                if c(i).toLong out (Int.MinValue, Int.MaxValue) then 1    // it's a `Long`
                else 0                                                    // it's an `Int`
            else if dPattern.matcher (c(i).toString).matches () then 2    // it's a `Double
            else if isDateTime (c(i).toString) then 3                     // it's a `TimeNum`
            else 4                                                        // it's a `String`

            state = if i == 0 then typ                                    // no previous values yet
                    else tMatrix(state, typ).toInt                        // f (state from previous values, typ)
            if state == 4 then go = false                                 // once a string, always a string, so quit
        } // cfor

        state match                                                       // convert to domain spec. char
            case 0 => 'I'
            case 1 => 'L'
            case 2 => 'D'
            case 3 => 'T'
            case 4 => 'S'
    end analyzeType

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Analyze the type of several COLUMNS by calling the analyzeType method on each one.
     *  @param cols        the columns to have their types analyzed
     *  @param samplePerc  the percentage for each column in cols to analyze
     */
    def analyzeColumns (cols: Array [VectorS], samplePerc: Int = 100): Array [Char] =
        cols.map (analyzeType (_, samplePerc))
    end analyzeColumns

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Analyze the type for several ROWS by creating corresponding columns and calling
     *  the analyzeColumns method.
     *  @param rows        the rows to have their types analyzed
     *  @param samplePerc  the percentage for each column in cols to analyze
     */
    def analyzeRows (rows: Bag [Array [String]], samplePerc: Int = 100): Array [Char] =
        val (nRows, nCols) = (rows.size, rows(0).size)
        val cols  = Array.fill (nCols)(new VectorS (nRows))
        for i <- 0 until nRows; j <- 0 until nCols do cols(j)(i) = rows(i)(j)
        analyzeColumns (cols, samplePerc)
    end analyzeRows

end MakeSchema


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `makeSchemaTest` main function tests the `MakeSchema` object on several
 *  columns of data.
 *  > runMain scalation.database.makeSchemaTest
 */
@main def makeSchemaTest (): Unit =

    val columns = Array (VectorS ("5", "7", "9"),                         // I
                         VectorS ("9", "3304385400", "9"),                // L
                         VectorS ("9", "3.14", "8"),                      // D
                         VectorS ("07/26/2018 16:30:01"),                 // T    
                         VectorS ("sun", "earth", "moon"),                // S
                         VectorS ("3.14", "Circle", "2"))                 // S

    banner ("Test analyzeType")
    for c <- columns do println (s"domain of $c is ${MakeSchema.analyzeType (c)}")

    banner ("Test analyzeColumns")
    println (stringOf (MakeSchema.analyzeColumns (columns)))

end makeSchemaTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `makeSchemaTest2` main function tests the `MakeSchema` object on several rows
 *  of data from a datafile with the goal of making a database derived from `Tabular`
 *  @see github.com/scalation/data/blob/master/traffic/d04_text_station_5min_2022_10_04.csv#L7
 *  @see `Tabular`, and `table.Table`
 *  @param name    the name of the table
 *  @param schema  the attributes for the table
 *  @param domain  the domains/data-types for the attributes ('D', 'I', 'L', 'S', 'X', 'T')
 *  @param key     the attributes forming the primary key
 *  trait Tabular [T <: Tabular [T]] (val name: String, val schema: Schema, val domain: Domain, val key: Schema)
 *  > runMain scalation.database.makeSchemaTest2
 */
@main def makeSchemaTest2 (): Unit =

    // Given a dataset in a datafile
    //   name:   derived from filename or user specified
    //   schema: attributes from meta-data or header in datafile (some Table.load methods do this automatically)
    //   domain: data types for each attribute/column (look at the data, or copy-paste rows for the analyzeRows method)
    //   key:    attribute or attribute combination that is unique for all rows (serves as primary key)

    val name = "trafficData"

    val schema = Array ("Timestamp", "Station", "District", "Freeway #", "Direction of Travel", "Lane Type",
                        "Station Length", "Samples", "% Observed", "Total Flow", "Avg Occupancy", "Avg Speed")

    val rows = Bag [Array [String]] ()
    rows += Array ("10/04/2022 00:00:00", "401833", "4", "101", "N", "ML", ".445", "50",  "80", "88", ".0129", "70")
         += Array ("10/04/2022 00:00:00", "401834", "4", "101", "N", "ML", ".225", "40",   "0", "82", ".0137", "66.2")
         += Array ("10/04/2022 00:00:00", "403205", "4", "101", "N", "FR", "0",    "10", "100",  "9", ".0068", "10")
         += Array ("10/04/2022 00:00:00", "403207", "4", "101", "N", "OR", "0",    "20", "100",  "9", ".0067",  "0")
         += Array ("10/04/2022 00:00:00", "403311", "4", "101", "N", "OR", "0",    "20", "100",  "1", ".0006",  "0")

    val domain = MakeSchema.analyzeRows (rows)

    val key = Array ("Timestamp", "Station")

    banner ("Make Schema")
    println (s"name   = $name")
    println (s"schema = ${stringOf (schema)}")
    println (s"domain = ${stringOf (domain)}")
    println (s"key    = ${stringOf (key)}")

    banner ("Check Schema: no message => okay")
    val (ns, nd) = (schema.size, domain.size)
    if ns != nd then println (s"require schema ($ns) and domain ($nd) sizes to be the same")

end makeSchemaTest2

