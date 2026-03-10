
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Sahil Varma
 *  @version 2.0
 *  @date    Sun Apr 27 21:04:18 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Comparing Various Join Algorithms including Link Joins
 */

package scalation
package database
package table

import java.io._

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

import scalation.mathstat.{Plot, VectorD}
//import scalation.scala2d.writeImage

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `my_timer_function` generates Tables and Ltables of specified size and
 *  performs various joins iteratively reducing the size of tables.
 *  > runMain scalation.database.table.my_timer_function
 */
@main def my_timer_function (): Unit =

    val totalSize = 100000
    val stepSize  = 10000
    val steps     = totalSize / stepSize
    val tSize     = new VectorD (steps)

    // NLJs take too much time to run, so if running on large table size, comment the NLJs
    val selectedJoins = Set (
//                            "NatJoinNLJ",
//                            "NatJoinUI",
//                            "NatJoinNUI",
//                            "NatJoinLink",
//                            "EquiJoinNLJ",
                              "EquiJoinUI",
//                            "EquiJoinNUI",
//                            "EquiJoinLink",
//                            "PredJoinNLJ",
                              "SortMergeJoin")

    val t_NatJoinNLJ    = new VectorD (steps)
    val t_NatJoinUI     = new VectorD (steps)
    val t_NatJoinNUI    = new VectorD (steps)
    val t_NatJoinLink   = new VectorD (steps)

    val t_EquiJoinNLJ   = new VectorD (steps)
    val t_EquiJoinUI    = new VectorD (steps)
    val t_EquiJoinNUI   = new VectorD (steps)
    val t_EquiJoinLink  = new VectorD (steps)

    val t_PredJoinNLJ   = new VectorD (steps)
    val t_SortMergeJoin = new VectorD (steps)

    banner ("create Tables: customerT and depositT")
    val customerT = Table ("customer", "cname, street, ccity", "S, S, S", "cname")
    val depositT  = Table ("deposit", "accno, balance, cname, bname", "I, D, S, S", "accno")
    depositT.addLinkage ("cname", customerT)

    TableGen.popTable (customerT, totalSize)
    TableGen.popTable (depositT, totalSize * 2)

    depositT.create_mindex ("cname")

    banner ("create LTables: customerLT and depositLT")
    val customerLT = LTable ("customer", "cname, street, ccity", "S, S, S", "cname")
    val depositLT  = LTable ("deposit", "accno, balance, cname, bname", "I, D, S, S", "accno")

    customerLT.tuples ++= customerT.tuples
    depositLT.tuples  ++= depositT.tuples

    depositLT.addLinkage ("cname", customerLT)
    depositLT.create_mindex ("cname")

    banner ("Perform Joins")
    val rep     = 10
    val a_cname = Array ("cname")
    val pred    = (t: Tuple, u: Tuple) => t(customerT.on("cname")) == u(depositT.on("cname"))
    var joinedTab1: Table = null
    var joinedTab2: Table = null

    for j <- 0 until steps do
        val sz = totalSize - (j * stepSize)

        tSize(j) = sz
        println (s"for j = $j: sz = $sz")

        // NATURAL JOINS
        if selectedJoins contains "NatJoinlNLJ" then
            t_NatJoinNLJ(j) = timedX (rep) { joinedTab1 = customerT join depositT }._2

        if selectedJoins contains "NatJoinUI" then
            t_NatJoinUI(j) = timedX (rep) { joinedTab1 = depositT join_ customerT }._2

        if selectedJoins contains "NatJoinNUI" then
            t_NatJoinNUI(j) = timedX (rep) { joinedTab2 = depositT _join customerT }._2

        if selectedJoins contains "NaturalLink" then
            t_NatJoinLink(j) = timedX (rep) { joinedTab1 = depositLT join customerLT }._2

        // EQUI JOINS
        if selectedJoins contains "EquiJoinNLJ" then
            t_EquiJoinNLJ(j) = timedX (rep) { joinedTab1 = depositT.join (a_cname, a_cname, customerT) }._2

        if selectedJoins contains "EquiJoinUI" then
            t_EquiJoinUI(j) = timedX (rep) { joinedTab1 = depositT.join ("cname", customerT) }._2

        if selectedJoins contains "EquiJoinNUI" then
            t_EquiJoinNUI(j) = timedX (rep) { joinedTab2 = depositT._join ("cname", customerT) }._2

        if selectedJoins contains "EquiJoinLink" then
            t_EquiJoinLink(j) = timedX (rep) { joinedTab2 = depositLT.join ("cname", customerLT) }._2

        // PREDICATE NLJ
        if selectedJoins contains "PredJoinNLJ" then
            t_PredJoinNLJ(j) = timedX (rep) { joinedTab1 = customerT.join (pred, depositT) }._2

        // SORT MERGE JOIN
        if selectedJoins contains "SortMergeJoin" then
            t_SortMergeJoin(j) = timedX (rep) { joinedTab2 = depositT._join_ ("cname", customerT) }._2

        customerT.deleteLast (stepSize)
        customerLT.deleteLast (stepSize)
    end for

//  joinedTab1.show ()
//  joinedTab2.show ()
    assert (checkTuples (joinedTab1.tuples, joinedTab2.tuples))
    // If assertion fails, generated join tables are correct but may be sorted and unsorted, due to nature of joins

    banner ("Show Timing Plots")
    println ("SIZE VECTOR:\n" + tSize)

    def plotTimings (joinName: String, timeVec: VectorD): Unit =
        println (s"joinName = $joinName \n $timeVec")
        new Plot (tSize / stepSize, timeVec, null, "$joinName elapsedTime", lines = true)
//      val plot = new Plot (tSize / stepSize, timeVec, null, "$joinName elapsedTime", lines = true)
//      writeImage (joinName + ".png", plot)
//      writeVectors2CSV (tSize, timeVec, joinName, "myResults.csv")
    end plotTimings

    // NATURAL JOINS
    if selectedJoins contains "NatJoinNLJ"  then plotTimings ("NatJoinNLJ",  t_NatJoinNLJ)
    if selectedJoins contains "NatJoinUI"   then plotTimings ("NatJoinUI",   t_NatJoinUI)
    if selectedJoins contains "NatJoinNUI"  then plotTimings ("NatJoinNUI",  t_NatJoinNUI)
    if selectedJoins contains "NatJoinLink" then plotTimings ("NatJoinLink", t_NatJoinLink)

    // EQUI JOINS
    if selectedJoins contains "EquiJoinNLJ"  then plotTimings ("EquiJoinNLJ",  t_EquiJoinNLJ)
    if selectedJoins contains "EquiJoinUI"   then plotTimings ("EquiJoinUI",   t_EquiJoinUI)
    if selectedJoins contains "EquiJoinNUI"  then plotTimings ("EquiJoinNUI",  t_EquiJoinNUI)
    if selectedJoins contains "EquiJoinLink" then plotTimings ("EquiJoinLink", t_EquiJoinLink)

    // Predicate NLJ
    if selectedJoins contains "PredJoinNLJ" then plotTimings ("PredJoinNLJ", t_PredJoinNLJ)

    // SORT MERGE JOIN
    if selectedJoins contains "SortMergeJoin" then plotTimings ("SortMergeJoin", t_SortMergeJoin)

end my_timer_function


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def writeVectors2CSV (tSize: VectorD, valueVector: VectorD, valueRowHeading: String, fileName: String): Unit =
    val file = new File (fileName)
    var lines = List [String] ()
    var header = ""

    if file.exists () then
        lines = Source.fromFile (fileName).getLines().toList
    else
        header = s"tSize,${tSize.mkString(",")}\n"

    val joinMethodExists = lines.exists (line => line.startsWith (s"$valueRowHeading,"))

    val clippedValues = valueVector.map (value => f"$value%.5f")
    val newRow = s"$valueRowHeading,${clippedValues.mkString(",")}"

    if joinMethodExists then
        // If the join method exists, overwrite it
        val updatedLines = lines.map {
            case line if line.startsWith (s"$valueRowHeading,") => newRow + "\n"  // replace the old line with the new one
            case line => line + "\n"                                              // keep the other lines unchanged
        }
        // Write all the updated lines back to the file, including the header
        val writer = new BufferedWriter (new FileWriter (fileName))
        updatedLines.foreach(writer.write)
        writer.close()
//      println(s"Data for $valueRowHeading updated in $fileName")

    else
        // If the join method does not exist, append the new row
        val writer = new BufferedWriter (new FileWriter (fileName, true))
        if lines.isEmpty then
            writer.write (s"$header\n")
        writer.write (s"$newRow\n")
        writer.close ()
//  println (s"Data for $valueRowHeading appended to $fileName")
end writeVectors2CSV


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def checkTuples (t1: ArrayBuffer [Tuple], t2: ArrayBuffer [Tuple]): Boolean =

    if t1.size != t2.size then
        println("Table Size not Match")
        return false

    var i = 0
    while i < t1.size do
        if ! (t1(i) sameElements t2(i)) then
            println ("TABLES NOT SAME AT: row index" + i)
            return false
        i += 1

    println ("Both Tables have same elements, Join Successful")
    true
end checkTuples

