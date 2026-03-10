
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Feb 26 18:20:39 EST 2014
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Basic UI Table for Displaying Statistics
 */

package scalation
package mathstat

import scala.collection.mutable.{ArrayBuffer => VEC}
//import scala.collection.mutable.{ListBuffer => VEC}

import scalation.scala2d.{Frame, ScrollPane, Table}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `StatTable` class display statistical results in a frame's table.
 *  @param _title  the title of the frame
 *  @param stats   the statistics to be displayed in the table.
 */
class StatTable (_title: String, stats: VEC [Statistic])
      extends Frame (_title):

    val rows  = stats.size + 1
    val cols  = Statistic.label.length
    val table = new Table (rows, cols)
    for j <- 0 until cols do table.setValueAt (Statistic.label(j), 0, j)
    for i <- 0 until rows - 1 do
        val st = stats(i).statRow
        for j <- 0 until cols do table.setValueAt (st(j), i+1, j)
    end for

    getContentPane.add (new ScrollPane (table))
    pack ()
    setVisible (true)

end StatTable


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `statTableTest` main function is used to test the `StatTable` class.
 *  > runMain scalation.mathstat.statTableTest
 */
@main def statTableTest (): Unit =

     println ("Create a StatTable called Test")
     val stats = VEC [Statistic] ()
     cfor (0, 50) { _ => stats += new Statistic () }
     for j <- 0 until 50; i <- 0 until 50 do stats(j).tally (i)
     new StatTable ("Test", stats)

end statTableTest

