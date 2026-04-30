
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Tue Jun 29 18:56:40 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note   Untility for Scalation Developers to Make VectorI.scala from VectorD.scala
 */

package scalation

import scala.io.StdIn.readLine

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `makeVectorI` main function make a draft version of `VectorI` from `VectorD`.
 *  It reads from standard input and writes to standard output.
 *  > sbt run <infile >outfile
 */
@main def makeVectorI (): Unit =

    println ("start makeVectorI")

    var line = ""

    var cont = true
    while cont do
        line = readLine ()
//      println (line)
        if line != null then
           line = line.replace ("Double", "Int")
           line = line.replace ("VectorD", "VectorI")
           println (line)
        else
           cont = false
    end while

end makeVectorI

