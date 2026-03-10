
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Venkatasai Krishna Gude
 *  @version 2.0
 *  @date    Mon Dec  2 23:39:39 EST 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Time Comparison of Several join Algorithms
 */

package scalation
package database
package table

import scalation.mathstat.{Plot, VectorD}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `timer_function` main method is used to compare the speed of several join
 *  algorithms.
 *  > runMain scalation.database.table.timer_function
 */
@main def timer_function (): Unit =

    val customer = Table ("customer", "cname, street, ccity", "S, S, S", "cname")
    val deposit  = Table ("deposit", "accno, balance, cname, bname", "I, D, S, S", "accno")

    deposit.addLinkage ("cname", customer)

    val selectedJoins = Set ("SortMerge", "Natural", "Theta", "Index", "Equi", "Predicate")

    val tSize = new VectorD (10)
    val rTime_SortMergeJoin = new VectorD (10)
    val rTime_NaturalJoin   = new VectorD (10)
    val rTime_ThetaJoin     = new VectorD (10)
    val rTime_IndexJoin     = new VectorD (10)
    val rTime_EquiJoin      = new VectorD (10)
    val rTime_PredicateJoin = new VectorD (10)

    var i = 0 
    val s = 90000
    TableGen.popTable (customer, s)
    TableGen.popTable (deposit, s)

    for sz <- 10000 to s by 10000 do
        tSize(i) = sz
        println (s"sz = $sz")
        val limitedCustomer = customer.limit (sz)
        val limitedDeposit  = deposit.limit (sz)
        limitedCustomer.create_index()

        if selectedJoins.contains ("SortMerge") then
            println ("SortMerge Join")
            rTime_SortMergeJoin(i) = timed (5, true) { limitedDeposit._join_ (("cname", limitedCustomer)) }._2

        if selectedJoins.contains ("Natural") then
            println ("Natural Join")
            rTime_NaturalJoin(i) = timed (5, true) {
                limitedDeposit.join (limitedCustomer) }._2

        if selectedJoins.contains ("Theta") then
            println ("Theta Join")
            rTime_ThetaJoin(i) = timed (5, true) { limitedCustomer.join (("cname == cname"), limitedDeposit) }._2

        if selectedJoins.contains ("Index") then
            println ("Equi join with Index")
            rTime_IndexJoin(i) = timed (5, true) { limitedDeposit.join (("cname", limitedCustomer)) }._2

        if selectedJoins.contains ("Equi") then
            println ("Equi join")
            rTime_EquiJoin(i) = timed (5, true) { limitedCustomer.join (Array("cname"), Array("cname"), limitedDeposit) }._2

        if selectedJoins.contains ("Predicate") then
            println ("Predicate Join")
            rTime_PredicateJoin(i) = timed (5, true) {
                limitedCustomer.join ((t, u) => t(customer.on("cname")) == u(deposit.on("cname")), limitedDeposit) }._2

        i += 1
    end for

    println ("Size Vector" + tSize)

    println ("SortMerge Join vector" + rTime_SortMergeJoin)
//  val sortMergePlot =
    new Plot (tSize/10000, rTime_SortMergeJoin, null, "EquiJoin with SortMerge elapsedTime", lines = true)

    println ("Natural Join vector" + rTime_ThetaJoin)
//  val naturalJoinPlot =
    new Plot (tSize/10000, rTime_NaturalJoin, null, "Natural Join Elapsed Time", lines = true)

    println ("Theta Join vector" + rTime_ThetaJoin)
//  val thetaJoinPlot =
    new Plot (tSize/10000, rTime_ThetaJoin, null, "Theta Join Elapsed Time", lines = true)

    println ("Index Join vector" + rTime_IndexJoin)
//  val indexJoinPlot =
    new Plot(tSize/10000, rTime_IndexJoin, null, "Equi-join with Index Elapsed Time", lines = true)

    println ("Equi Join vector" + rTime_ThetaJoin)
//  val equiJoinPlot =
    new Plot(tSize/10000, rTime_EquiJoin, null, "Equi-join Elapsed Time", lines = true)

    println ("Predicate Join vector" + rTime_PredicateJoin)
//  val predicateJoinPlot =
    new Plot(tSize, rTime_PredicateJoin, null, "Predicate Join Elapsed Time", lines = true)

end timer_function

/*
   sortMergePlot.savePlot("/Users/krishna/Documents/COURSE_WORK/major/Results/Sort-MergePlot.png")
   naturalJoinPlot.savePlot("/Users/krishna/Documents/COURSE_WORK/major/naturalJoinPlot.png")
   thetaJoinPlot.savePlot("/Users/krishna/Documents/COURSE_WORK/major/ThetaJoinPlot.png")
   indexJoinPlot.savePlot("/Users/krishna/Documents/COURSE_WORK/major/Results/EquiJoinIndex.png")
   equiJoinPlot.savePlot("/Users/krishna/Documents/COURSE_WORK/major/EquiJoin.png")
   predicateJoinPlot.savePlot("/Users/krishna/Documents/COURSE_WORK/major/PredicateJoinPlot.png")

   val joinTimes = Map (
       "Theta Join Join Time (ms)" -> rTime_ThetaJoin,
       "Predicate Join Time (ms)"  -> rTime_PredicateJoin,
       "Theta Join Time (ms)"      -> rTime_ThetaJoin,
       "Natural Join Time (ms)"    -> rTime_NaturalJoin,
       "Predicate Join Time (ms)"  -> rTime_PredicateJoin,
       "Equi-Join without Index Time (ms)" -> runTimeEquiJoin,
   )

    // Export to Excel
    exportToExcelWithJoins(tSize, joinTimes, "/Users/krishna/Documents/COURSE_WORK/major/JoinPerformanceData.xlsx")

//import org.apache.poi.ss.usermodel._
import org.apache.poi.xssf.usermodel.XSSFWorkbook
import java.io.FileOutputStream

def exportToExcelWithJoins (tSize: VectorD,
                            joinTimes: Map [String, VectorD],
                            filePath: String): Unit = 

  // Create a new workbook and a sheet
  val workbook = new XSSFWorkbook()
  val sheet = workbook.createSheet("Join Performance Data")

  // Create the header row
  val headerRow = sheet.createRow(0)
  headerRow.createCell(0).setCellValue("Tuple Size (Tuples)")

  // Add join type headers (columns)
  val joinTypes = joinTimes.keys.toArray
  for (j <- joinTypes.indices) {
    headerRow.createCell(j + 1).setCellValue(joinTypes(j))
  }

  // Populate rows with data
  for (i <- tSize.indices) {
    val row = sheet.createRow(i + 1)
    row.createCell(0).setCellValue(tSize(i)) // Tuple size

    // Populate execution times for each join type
    for (j <- joinTypes.indices) {
      val joinType = joinTypes(j)
      val timeVector = joinTimes(joinType)
      row.createCell(j + 1).setCellValue(timeVector(i))
    }
  }

  // Adjust column width for readability
  for (col <- 0 to joinTypes.length) sheet.autoSizeColumn(col)

  // Write the output to an Excel file
  val fos = new FileOutputStream(filePath)
  workbook.write(fos)
  fos.close()
  workbook.close()

  println(s"Data successfully exported to Excel file: $filePath")
end exportToExcelWithJoins

*/

