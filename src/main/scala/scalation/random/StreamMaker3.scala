
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Mar 28 14:14:24 EDT 2014
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Code Generator: 1000 Seeds for LCG RNG
 */

package scalation
package random

import java.util.Date

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `streamMaker3` main function finds seeds for the `Random3` random number generator.
 *  This generator has a period length around 2^31.  Each seed is a 32-bit integer.
 *  > runMain scalation.random.streamMaker3
 */
@main def streamMaker3 (): Unit =

    val r = Random3 ()
    printSeeds (RandomSeeds.N_STREAMS)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print out 'k' seed values to initialize the streams for the Random3 generator.
     *  @param k  the number of seeds to print
     */
    def printSeeds (k: Int): Unit =
        val streamLen = ((r.getM - 1) / k).toInt
        var seed      = 123
        val INDENT    = "                       "
        val COLS      = 5
        val ROWS      = k / COLS

        println ("\n" +
                 "//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n" +
                 "/** @author  John Miller (code generated)\n" +
                 " *  @version 2.0\n" +
                 " *  @date    " + new Date () + "\n" +
                 " *  @see     LICENSE (MIT style license file).\n" +
                 " *\n" +
                 " *  @note    1000 Seeds for LCG Random Number Generator\n" +
                 " */\n" +
                 "\n" +
                 "package scalation.random\n" +
                 "\n" +
                 "//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n" +
                 "/** The first " + k + " seeds for the LCG random number generator.\n" +
                 " */\n" +
                 "object RandomSeeds3\n" +
                 "{")
        print ("    val seeds = Array (")

        for i <- 0 until ROWS; j <- 0 until COLS do
            if i > 0 && j == 0 then print (INDENT)
            if j < COLS-1 then      print ("\t" + seed + ","); if seed < 1000000 then print ("\t")
            else if i < ROWS-1 then println ("\t" + seed + ",")
            else                    println ("\t" + seed + ")")
            cfor (0, streamLen) { _ => seed = r.igen }                // iterate for the next seed
        end for

        println ("\n} // RandomSeeds3 object" +
                 "\n")
    end printSeeds

end streamMaker3

